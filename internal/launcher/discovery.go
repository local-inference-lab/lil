// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package launcher

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/charmbracelet/x/ansi"
)

const gpuRuntimeProbe = `
import json
import torch

print(json.dumps([
    {
        "id": index,
        "memory": torch.cuda.get_device_properties(index).total_memory,
        "major": torch.cuda.get_device_capability(index)[0],
        "minor": torch.cuda.get_device_capability(index)[1],
    }
    for index in range(torch.cuda.device_count())
]))
`

type LocalDiscoveryOptions struct {
	Name                 string
	Host                 string
	Port                 int
	GPUMemoryUtilization float64
	DefaultTP            string
	RepoRoot             string
	Python               string
	B12XRoot             string
	CUDAHome             string
}

type SparkDiscoveryOptions struct {
	Name                  string
	Host                  string
	Port                  int
	GPUMemoryUtilization  float64
	DefaultTP             string
	RepoRoot              string
	Python                string
	B12XRoot              string
	RuntimeRepoRoot       string
	RuntimePython         string
	VLLMBin               string
	RuntimeB12XRoot       string
	Nodes                 []string
	MasterPort            int
	Image                 string
	ContainerNamePrefix   string
	ContainerMemoryGB     int
	ContainerMemorySwapGB int
	ContainerShmGB        int
	ContainerPidsLimit    int
	ContainerNofileLimit  int
	CUDAHome              string
	NCCLDebug             string
	DeviceID              int
}

const roceGIDProbe = `
import json
import sys
from pathlib import Path

hcas = json.loads(sys.argv[1])
common = None
for hca in hcas:
    root = Path("/sys/class/infiniband") / hca / "ports" / "1"
    indices = set()
    for path in (root / "gid_attrs" / "types").iterdir():
        try:
            gid_type = path.read_text().strip()
            gid = (root / "gids" / path.name).read_text().strip().lower()
        except OSError:
            continue
        if gid_type == "RoCE v2" and gid.startswith("0000:0000:0000:0000:0000:ffff:"):
            indices.add(int(path.name))
    common = indices if common is None else common & indices
print(json.dumps(sorted(common or [])))
`

type gpuObservation struct {
	ID     int   `json:"id"`
	Memory int64 `json:"memory"`
	Major  int   `json:"major"`
	Minor  int   `json:"minor"`
}

func discoverPath(path, root, fallback string) (string, error) {
	if path == "" {
		path = fallback
	}
	return absolutePath(path, root)
}

func discoverRepoRoot(explicit string) (string, error) {
	if explicit != "" {
		return filepath.Abs(explicit)
	}
	directory, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		if info, statErr := os.Stat(filepath.Join(directory, "vllm", "__init__.py")); statErr == nil && !info.IsDir() {
			return directory, nil
		}
		parent := filepath.Dir(directory)
		if parent == directory {
			break
		}
		directory = parent
	}
	return "", fmt.Errorf("cannot identify vLLM source from the working directory; pass --repo-root")
}

func validateLocalPath(path, description string, executable bool) error {
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("%s %s: %w", description, path, err)
	}
	if executable && info.Mode().Perm()&0o111 == 0 {
		return fmt.Errorf("%s is not executable: %s", description, path)
	}
	return nil
}

func parseGPUObservations(data []byte) ([]gpuObservation, error) {
	var observations []gpuObservation
	if err := json.Unmarshal(data, &observations); err != nil {
		return nil, fmt.Errorf("invalid GPU probe output: %w", err)
	}
	if len(observations) == 0 {
		return nil, fmt.Errorf("GPU probe found no CUDA devices")
	}
	seen := map[int]bool{}
	for _, gpu := range observations {
		if gpu.ID < 0 || gpu.Memory <= 0 || gpu.Major <= 0 || seen[gpu.ID] {
			return nil, fmt.Errorf("GPU probe returned an invalid device: %+v", gpu)
		}
		seen[gpu.ID] = true
	}
	sort.Slice(observations, func(i, j int) bool { return observations[i].ID < observations[j].ID })
	return observations, nil
}

func minimumGPUMemory(observations []gpuObservation) int64 {
	result := observations[0].Memory
	for _, gpu := range observations[1:] {
		if gpu.Memory < result {
			result = gpu.Memory
		}
	}
	return result
}

func commonComputeCapability(observations []gpuObservation) (int, int, error) {
	major, minor := observations[0].Major, observations[0].Minor
	for _, gpu := range observations[1:] {
		if gpu.Major != major || gpu.Minor != minor {
			return 0, 0, fmt.Errorf(
				"mixed GPU compute capabilities are unsupported: %d.%d and %d.%d",
				major, minor, gpu.Major, gpu.Minor,
			)
		}
	}
	return major, minor, nil
}

func cuteDSLArch(major, minor int) string {
	return fmt.Sprintf("sm_%d%da", major, minor)
}

func gpuProbeEnvironment() []string {
	environment := environmentMap(os.Environ())
	delete(environment, "CUDA_VISIBLE_DEVICES")
	return environmentList(environment)
}

func runGPUProbe(ctx context.Context, python, directory string) ([]gpuObservation, error) {
	output, status, err := runCommand(
		ctx, 30*time.Second, directory, gpuProbeEnvironment(),
		[]string{python, "-c", gpuRuntimeProbe},
	)
	if err != nil {
		return nil, err
	}
	if status != 0 {
		return nil, fmt.Errorf("GPU runtime probe failed: %s", lastOutputLine(output, "no output"))
	}
	return parseGPUObservations(output)
}

func inferCUDAHome(explicit string) (string, error) {
	if explicit != "" {
		path, err := filepath.Abs(explicit)
		if err != nil {
			return "", err
		}
		if err := validateLocalPath(filepath.Join(path, "bin", "ptxas"), "CUDA ptxas", true); err != nil {
			return "", err
		}
		return path, nil
	}
	if ptxas, err := exec.LookPath("ptxas"); err == nil {
		path, err := filepath.EvalSymlinks(ptxas)
		if err == nil {
			return filepath.Dir(filepath.Dir(path)), nil
		}
	}
	for _, path := range []string{"/opt/cuda", "/usr/local/cuda"} {
		if validateLocalPath(filepath.Join(path, "bin", "ptxas"), "CUDA ptxas", true) == nil {
			return path, nil
		}
	}
	return "", fmt.Errorf("CUDA ptxas was not found; pass --cuda-home")
}

func parseNVIDIATopology(data []byte, deviceIDs []int) [][]int {
	known := map[int]bool{}
	parent := map[int]int{}
	for _, id := range deviceIDs {
		known[id] = true
		parent[id] = id
	}
	var find func(int) int
	find = func(value int) int {
		if parent[value] != value {
			parent[value] = find(parent[value])
		}
		return parent[value]
	}
	join := func(left, right int) {
		left, right = find(left), find(right)
		if left != right {
			parent[right] = left
		}
	}
	lines := strings.Split(ansi.Strip(string(data)), "\n")
	var columns []int
	for _, line := range lines {
		fields := strings.Fields(line)
		if len(fields) == 0 {
			continue
		}
		if columns == nil {
			for _, field := range fields {
				if strings.HasPrefix(field, "GPU") {
					id, err := strconv.Atoi(strings.TrimPrefix(field, "GPU"))
					if err == nil && known[id] {
						columns = append(columns, id)
					}
				}
			}
			continue
		}
		if !strings.HasPrefix(fields[0], "GPU") {
			continue
		}
		row, err := strconv.Atoi(strings.TrimPrefix(fields[0], "GPU"))
		if err != nil || !known[row] {
			continue
		}
		for index, relation := range fields[1:] {
			if index >= len(columns) {
				break
			}
			if relation == "PIX" || relation == "PXB" || strings.HasPrefix(relation, "NV") {
				join(row, columns[index])
			}
		}
	}
	components := map[int][]int{}
	for _, id := range deviceIDs {
		components[find(id)] = append(components[find(id)], id)
	}
	groups := make([][]int, 0, len(components))
	for _, component := range components {
		sort.Ints(component)
		groups = append(groups, component)
	}
	sort.Slice(groups, func(i, j int) bool { return groups[i][0] < groups[j][0] })
	pools := make([][]int, 0, len(groups)*2)
	for _, group := range groups {
		pools = append(pools, append([]int(nil), group...))
	}
	combined := []int{}
	for _, group := range groups {
		combined = append(combined, group...)
		if len(group) < len(deviceIDs) && len(combined) > len(group) {
			pools = append(pools, append([]int(nil), combined...))
		}
	}
	return pools
}

func DiscoverLocalTopology(ctx context.Context, options LocalDiscoveryOptions) (Topology, error) {
	if options.DefaultTP == "" {
		options.DefaultTP = DefaultTPFit
	}
	if _, err := resolveDefaultTP(options.DefaultTP, "discovery"); err != nil {
		return Topology{}, err
	}
	if options.Port == 0 {
		options.Port = 8000
	}
	if options.Host == "" {
		options.Host = "0.0.0.0"
	}
	if options.Name == "" {
		name, err := os.Hostname()
		if err != nil {
			return Topology{}, fmt.Errorf("discover hostname: %w", err)
		}
		options.Name = name
	}
	repoRoot, err := discoverRepoRoot(options.RepoRoot)
	if err != nil {
		return Topology{}, err
	}
	python, err := discoverPath(options.Python, repoRoot, ".venv/bin/python")
	if err != nil {
		return Topology{}, err
	}
	b12xRoot, err := discoverPath(options.B12XRoot, repoRoot, filepath.Join(filepath.Dir(repoRoot), "b12x"))
	if err != nil {
		return Topology{}, err
	}
	for _, path := range []struct {
		value, description string
		executable         bool
	}{
		{filepath.Join(repoRoot, "vllm", "__init__.py"), "vLLM source", false},
		{python, "vLLM Python", true},
		{filepath.Join(b12xRoot, "b12x", "__init__.py"), "B12X source", false},
	} {
		if err := validateLocalPath(path.value, path.description, path.executable); err != nil {
			return Topology{}, err
		}
	}
	cudaHome, err := inferCUDAHome(options.CUDAHome)
	if err != nil {
		return Topology{}, err
	}
	gpus, err := runGPUProbe(ctx, python, repoRoot)
	if err != nil {
		return Topology{}, err
	}
	major, minor, err := commonComputeCapability(gpus)
	if err != nil {
		return Topology{}, err
	}
	deviceIDs := make([]int, len(gpus))
	for index, gpu := range gpus {
		deviceIDs[index] = gpu.ID
	}
	topoOutput, status, commandErr := runCommand(
		ctx, 10*time.Second, "", nil, []string{"nvidia-smi", "topo", "-m"},
	)
	devicePools := [][]int{append([]int(nil), deviceIDs...)}
	if commandErr == nil && status == 0 {
		devicePools = parseNVIDIATopology(topoOutput, deviceIDs)
	}
	return Topology{
		Kind: "local", GPUMemoryUtilization: options.GPUMemoryUtilization,
		DefaultTP: options.DefaultTP,
		Local: &LocalTopology{
			Name: options.Name, Host: options.Host, Port: options.Port,
			DeviceMemoryBytes: minimumGPUMemory(gpus),
			RepoRoot:          repoRoot, Python: python, B12XRoot: b12xRoot,
			CUDAHome: cudaHome, CuteDSLArch: cuteDSLArch(major, minor),
			DevicePools: devicePools, Environment: DefaultLocalEnvironment(),
		},
	}, nil
}

type ipAddressObservation struct {
	Family    string `json:"family"`
	Local     string `json:"local"`
	PrefixLen int    `json:"prefixlen"`
	Scope     string `json:"scope"`
}

type ipInterfaceObservation struct {
	Name      string                 `json:"ifname"`
	State     string                 `json:"operstate"`
	Addresses []ipAddressObservation `json:"addr_info"`
}

type rdmaObservation struct {
	Name          string `json:"ifname"`
	State         string `json:"state"`
	PhysicalState string `json:"physical_state"`
	NetDevice     string `json:"netdev"`
}

type remoteNetworkObservation struct {
	SSHHost    string
	Address    string
	PrefixLen  int
	Network    string
	Ethernet   string
	RDMADevice string
}

func networkKey(address string, prefix int) (string, bool) {
	ip := net.ParseIP(address)
	if ip == nil || ip.To4() == nil || !ip.IsPrivate() || prefix < 1 || prefix > 32 {
		return "", false
	}
	mask := net.CIDRMask(prefix, 32)
	return (&net.IPNet{IP: ip.Mask(mask), Mask: mask}).String(), true
}

func parseRemoteNetworks(
	host string,
	ipData []byte,
	rdmaData []byte,
) ([]remoteNetworkObservation, error) {
	var interfaces []ipInterfaceObservation
	if err := json.Unmarshal(ipData, &interfaces); err != nil {
		return nil, fmt.Errorf("%s returned invalid ip JSON: %w", host, err)
	}
	var rdma []rdmaObservation
	if err := json.Unmarshal(rdmaData, &rdma); err != nil {
		return nil, fmt.Errorf("%s returned invalid RDMA JSON: %w", host, err)
	}
	active := map[string]string{}
	for _, device := range rdma {
		if device.State == "ACTIVE" && device.PhysicalState == "LINK_UP" &&
			device.NetDevice != "" && device.Name != "" {
			active[device.NetDevice] = device.Name
		}
	}
	observations := []remoteNetworkObservation{}
	for _, iface := range interfaces {
		rdmaDevice, ok := active[iface.Name]
		if !ok || iface.State != "UP" {
			continue
		}
		for _, address := range iface.Addresses {
			key, ok := networkKey(address.Local, address.PrefixLen)
			if !ok || address.Family != "inet" || address.Scope != "global" {
				continue
			}
			observations = append(observations, remoteNetworkObservation{
				SSHHost: host, Address: address.Local, PrefixLen: address.PrefixLen,
				Network: key, Ethernet: iface.Name, RDMADevice: rdmaDevice,
			})
		}
	}
	if len(observations) == 0 {
		return nil, fmt.Errorf("%s has no active private IPv4 RDMA network", host)
	}
	return observations, nil
}

func selectSparkNetworks(
	hosts []string,
	observations map[string][]remoteNetworkObservation,
) ([]SparkNode, error) {
	common := map[string]bool{}
	for _, observation := range observations[hosts[0]] {
		common[observation.Network] = true
	}
	for _, host := range hosts[1:] {
		present := map[string]bool{}
		for _, observation := range observations[host] {
			present[observation.Network] = true
		}
		for network := range common {
			if !present[network] {
				delete(common, network)
			}
		}
	}
	if len(common) == 0 {
		return nil, fmt.Errorf("nodes have no common active private IPv4 RDMA subnet")
	}
	networks := make([]string, 0, len(common))
	for network := range common {
		networks = append(networks, network)
	}
	sort.Slice(networks, func(i, j int) bool {
		left, _, _ := net.ParseCIDR(networks[i])
		right, _, _ := net.ParseCIDR(networks[j])
		return bytesCompare(left.To4(), right.To4()) < 0
	})
	nodes := make([]SparkNode, 0, len(hosts))
	for _, host := range hosts {
		byNetwork := map[string]remoteNetworkObservation{}
		for _, observation := range observations[host] {
			if common[observation.Network] {
				byNetwork[observation.Network] = observation
			}
		}
		primary := byNetwork[networks[0]]
		rdmaInterfaces := make([]string, 0, len(networks))
		seen := map[string]bool{}
		for _, network := range networks {
			device := byNetwork[network].RDMADevice
			if !seen[device] {
				rdmaInterfaces = append(rdmaInterfaces, device)
				seen[device] = true
			}
		}
		nodes = append(nodes, SparkNode{
			SSHHost: host, Address: primary.Address,
			EthernetInterface: primary.Ethernet,
			RDMAInterfaces:    rdmaInterfaces,
		})
	}
	return nodes, nil
}

func bytesCompare(left, right []byte) int {
	for index := range left {
		if left[index] < right[index] {
			return -1
		}
		if left[index] > right[index] {
			return 1
		}
	}
	return 0
}

func remoteOutput(ctx context.Context, host string, argv []string, timeout time.Duration, description string) ([]byte, error) {
	output, status, err := remoteRun(ctx, host, argv, timeout)
	if err != nil {
		return nil, fmt.Errorf("%s %s: %w", host, description, err)
	}
	if status != 0 {
		return nil, fmt.Errorf("%s %s failed: %s", host, description, lastOutputLine(output, "no output"))
	}
	return output, nil
}

func remotePathCheck(ctx context.Context, host, path, mode string) error {
	if _, err := remoteOutput(ctx, host, []string{"test", mode, path}, 10*time.Second, "path check"); err != nil {
		return fmt.Errorf("required remote path is unavailable: %s: %w", path, err)
	}
	return nil
}

func remoteGPUProbe(ctx context.Context, host, python string, deviceID int) ([]gpuObservation, error) {
	output, err := remoteOutput(
		ctx, host,
		[]string{"env", "CUDA_VISIBLE_DEVICES=" + strconv.Itoa(deviceID), python, "-c", gpuRuntimeProbe},
		30*time.Second,
		"GPU runtime probe",
	)
	if err != nil {
		return nil, err
	}
	return parseGPUObservations(output)
}

func inferRemoteCUDAHome(ctx context.Context, hosts []string, explicit string) (string, error) {
	candidates := []string{explicit}
	if explicit == "" {
		candidates = []string{"/opt/cuda", "/usr/local/cuda"}
	}
	available := []string{}
	for _, candidate := range candidates {
		present := true
		for _, host := range hosts {
			if remotePathCheck(ctx, host, filepath.Join(candidate, "bin", "ptxas"), "-x") != nil {
				present = false
				break
			}
		}
		if present {
			available = append(available, candidate)
		}
	}
	if len(available) == 0 {
		if explicit != "" {
			return "", fmt.Errorf("CUDA ptxas is not executable under %s on every node", explicit)
		}
		return "", fmt.Errorf("common CUDA ptxas was not found; pass --cuda-home")
	}
	return available[0], nil
}

func commonRemoteImage(ctx context.Context, hosts []string, explicit string) (string, error) {
	if explicit != "" {
		for _, host := range hosts {
			if _, err := remoteOutput(ctx, host, []string{"docker", "image", "inspect", explicit}, 15*time.Second, "Docker image check"); err != nil {
				return "", err
			}
		}
		return explicit, nil
	}
	var common map[string]bool
	for _, host := range hosts {
		output, err := remoteOutput(
			ctx, host,
			[]string{"docker", "image", "ls", "--format", "{{.Repository}}:{{.Tag}}"},
			15*time.Second,
			"Docker image inventory",
		)
		if err != nil {
			return "", err
		}
		present := map[string]bool{}
		for _, image := range strings.Fields(string(output)) {
			if !strings.Contains(image, "<none>") {
				present[image] = true
			}
		}
		if common == nil {
			common = present
			continue
		}
		for image := range common {
			if !present[image] {
				delete(common, image)
			}
		}
	}
	images := make([]string, 0, len(common))
	for image := range common {
		images = append(images, image)
	}
	sort.Strings(images)
	if len(images) == 0 {
		return "", fmt.Errorf("nodes have no common tagged Docker image; pass --image")
	}
	if len(images) != 1 {
		return "", fmt.Errorf("Docker image is ambiguous (%s); pass --image", strings.Join(images, ", "))
	}
	return images[0], nil
}

func discoverRoCEGIDIndex(ctx context.Context, nodes []SparkNode, runtimePython string) (int, error) {
	var common map[int]bool
	for _, node := range nodes {
		hcas, _ := json.Marshal(node.RDMAInterfaces)
		output, err := remoteOutput(
			ctx, node.SSHHost,
			[]string{runtimePython, "-c", roceGIDProbe, string(hcas)},
			10*time.Second,
			"RoCE GID probe",
		)
		if err != nil {
			return 0, err
		}
		var indices []int
		if err := json.Unmarshal(output, &indices); err != nil {
			return 0, fmt.Errorf("%s returned invalid RoCE GID data: %w", node.SSHHost, err)
		}
		present := map[int]bool{}
		for _, index := range indices {
			present[index] = true
		}
		if common == nil {
			common = present
			continue
		}
		for index := range common {
			if !present[index] {
				delete(common, index)
			}
		}
	}
	indices := make([]int, 0, len(common))
	for index := range common {
		indices = append(indices, index)
	}
	sort.Ints(indices)
	if len(indices) == 0 {
		return 0, fmt.Errorf("Spark nodes have no common IPv4 RoCE v2 GID index")
	}
	return indices[0], nil
}

func DiscoverSparkTopology(ctx context.Context, options SparkDiscoveryOptions) (Topology, error) {
	if options.DefaultTP == "" {
		options.DefaultTP = DefaultTPAll
	}
	if _, err := resolveDefaultTP(options.DefaultTP, "discovery"); err != nil {
		return Topology{}, err
	}
	if len(options.Nodes) == 0 {
		return Topology{}, fmt.Errorf("Spark/RDMA discovery requires at least one --node")
	}
	seenHosts := map[string]bool{}
	for _, host := range options.Nodes {
		if host == "" || seenHosts[host] {
			return Topology{}, fmt.Errorf("--node values must be unique and non-empty")
		}
		seenHosts[host] = true
	}
	if options.Name == "" {
		options.Name = "spark-rdma"
	}
	if options.Host == "" {
		options.Host = "0.0.0.0"
	}
	if options.Port == 0 {
		options.Port = 8000
	}
	if options.MasterPort == 0 {
		options.MasterPort = 29638
	}
	if options.ContainerNamePrefix == "" {
		options.ContainerNamePrefix = "vllm-fleet"
	}
	if options.ContainerMemoryGB == 0 {
		options.ContainerMemoryGB = 108
	}
	if options.ContainerMemorySwapGB == 0 {
		options.ContainerMemorySwapGB = options.ContainerMemoryGB + 4
	}
	if options.ContainerShmGB == 0 {
		options.ContainerShmGB = 64
	}
	if options.ContainerPidsLimit == 0 {
		options.ContainerPidsLimit = 4096
	}
	if options.ContainerNofileLimit == 0 {
		options.ContainerNofileLimit = 1048576
	}
	if options.NCCLDebug == "" {
		options.NCCLDebug = "INFO"
	}
	repoRoot, err := discoverRepoRoot(options.RepoRoot)
	if err != nil {
		return Topology{}, err
	}
	python, err := discoverPath(options.Python, repoRoot, ".venv/bin/python")
	if err != nil {
		return Topology{}, err
	}
	b12xRoot, err := discoverPath(options.B12XRoot, repoRoot, filepath.Join(filepath.Dir(repoRoot), "b12x"))
	if err != nil {
		return Topology{}, err
	}
	for _, path := range []struct {
		value, description string
		executable         bool
	}{
		{filepath.Join(repoRoot, "vllm", "__init__.py"), "vLLM source", false},
		{python, "vLLM Python", true},
		{filepath.Join(b12xRoot, "b12x", "__init__.py"), "B12X source", false},
	} {
		if err := validateLocalPath(path.value, path.description, path.executable); err != nil {
			return Topology{}, err
		}
	}
	firstHomeData, err := remoteOutput(ctx, options.Nodes[0], []string{"pwd"}, 10*time.Second, "home-directory probe")
	if err != nil {
		return Topology{}, err
	}
	remoteHome := strings.TrimSpace(string(firstHomeData))
	if !filepath.IsAbs(remoteHome) {
		return Topology{}, fmt.Errorf("%s returned invalid home directory %q", options.Nodes[0], remoteHome)
	}
	if options.RuntimeRepoRoot == "" {
		options.RuntimeRepoRoot = filepath.Join(remoteHome, "projects", "vllm")
	}
	if options.RuntimePython == "" {
		options.RuntimePython = filepath.Join(options.RuntimeRepoRoot, ".venv", "bin", "python")
	}
	if options.VLLMBin == "" {
		options.VLLMBin = filepath.Join(options.RuntimeRepoRoot, ".venv", "bin", "vllm")
	}
	if options.RuntimeB12XRoot == "" {
		options.RuntimeB12XRoot = filepath.Join(remoteHome, "projects", "b12x")
	}
	options.CUDAHome, err = inferRemoteCUDAHome(ctx, options.Nodes, options.CUDAHome)
	if err != nil {
		return Topology{}, err
	}
	remotePaths := []struct {
		path, mode string
	}{
		{filepath.Join(options.RuntimeRepoRoot, "vllm", "__init__.py"), "-f"},
		{options.RuntimePython, "-x"},
		{options.VLLMBin, "-x"},
		{filepath.Join(options.RuntimeB12XRoot, "b12x", "__init__.py"), "-f"},
		{filepath.Join(options.CUDAHome, "bin", "ptxas"), "-x"},
	}
	networkObservations := map[string][]remoteNetworkObservation{}
	allGPUs := []gpuObservation{}
	for _, host := range options.Nodes {
		for _, path := range remotePaths {
			if err := remotePathCheck(ctx, host, path.path, path.mode); err != nil {
				return Topology{}, err
			}
		}
		ipData, err := remoteOutput(ctx, host, []string{"ip", "-j", "-4", "addr", "show"}, 10*time.Second, "network inventory")
		if err != nil {
			return Topology{}, err
		}
		rdmaData, err := remoteOutput(ctx, host, []string{"rdma", "-j", "link", "show"}, 10*time.Second, "RDMA inventory")
		if err != nil {
			return Topology{}, err
		}
		networkObservations[host], err = parseRemoteNetworks(host, ipData, rdmaData)
		if err != nil {
			return Topology{}, err
		}
		gpus, err := remoteGPUProbe(ctx, host, options.RuntimePython, options.DeviceID)
		if err != nil {
			return Topology{}, err
		}
		if len(gpus) != 1 {
			return Topology{}, fmt.Errorf("%s device %d probe returned %d GPUs", host, options.DeviceID, len(gpus))
		}
		allGPUs = append(allGPUs, gpus[0])
	}
	nodes, err := selectSparkNetworks(options.Nodes, networkObservations)
	if err != nil {
		return Topology{}, err
	}
	gidIndex, err := discoverRoCEGIDIndex(ctx, nodes, options.RuntimePython)
	if err != nil {
		return Topology{}, err
	}
	mergeNICs := false
	for _, node := range nodes {
		mergeNICs = mergeNICs || len(node.RDMAInterfaces) > 1
	}
	major, minor, err := commonComputeCapability(allGPUs)
	if err != nil {
		return Topology{}, err
	}
	image, err := commonRemoteImage(ctx, options.Nodes, options.Image)
	if err != nil {
		return Topology{}, err
	}
	cacheCandidates := []struct{ source, target string }{
		{filepath.Join(remoteHome, ".cache", "huggingface"), "/root/.cache/huggingface"},
		{filepath.Join(remoteHome, ".cache", "vllm"), "/root/.cache/vllm"},
		{filepath.Join(remoteHome, ".cache", "flashinfer"), "/root/.cache/flashinfer"},
		{filepath.Join(remoteHome, ".triton"), "/root/.triton"},
		{filepath.Join(remoteHome, ".tilelang"), "/root/.tilelang"},
	}
	cacheMounts := []CacheMount{}
	for _, candidate := range cacheCandidates {
		present := true
		for _, host := range options.Nodes {
			if remotePathCheck(ctx, host, candidate.source, "-d") != nil {
				present = false
				break
			}
		}
		if present {
			cacheMounts = append(cacheMounts, CacheMount{candidate.source, candidate.target})
		}
	}
	deviceMemory := minimumGPUMemory(allGPUs)
	containerLimit := int64(options.ContainerMemoryGB) << 30
	if containerLimit < deviceMemory {
		deviceMemory = containerLimit
	}
	return Topology{
		Kind: "spark_rdma", GPUMemoryUtilization: options.GPUMemoryUtilization,
		DefaultTP: options.DefaultTP,
		Spark: &SparkRDMATopology{
			Name: options.Name, Host: options.Host, Port: options.Port,
			DeviceMemoryBytes: deviceMemory,
			RepoRoot:          repoRoot, Python: python,
			RuntimeRepoRoot:       options.RuntimeRepoRoot,
			RuntimePython:         options.RuntimePython,
			VLLMBin:               options.VLLMBin,
			B12XRoot:              b12xRoot,
			RuntimeB12XRoot:       options.RuntimeB12XRoot,
			Nodes:                 nodes,
			MasterPort:            options.MasterPort,
			Image:                 image,
			ContainerNamePrefix:   options.ContainerNamePrefix,
			ContainerMemoryGB:     options.ContainerMemoryGB,
			ContainerMemorySwapGB: options.ContainerMemorySwapGB,
			ContainerShmGB:        options.ContainerShmGB,
			ContainerPidsLimit:    options.ContainerPidsLimit,
			ContainerNofileLimit:  options.ContainerNofileLimit,
			CacheMounts:           cacheMounts,
			CUDAHome:              options.CUDAHome,
			CuteDSLArch:           cuteDSLArch(major, minor),
			NCCLDebug:             options.NCCLDebug,
			NCCLIBGIDIndex:        gidIndex,
			NCCLIBMergeNICs:       mergeNICs,
			DeviceID:              options.DeviceID,
			Environment:           DefaultSparkEnvironment(),
		},
	}, nil
}
