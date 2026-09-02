// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package launcher

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

const schemaVersion = 1

type topologyMeta struct {
	SchemaVersion int    `yaml:"schema_version"`
	Kind          string `yaml:"kind"`
}

type localTopologyYAML struct {
	SchemaVersion        int               `yaml:"schema_version"`
	Kind                 string            `yaml:"kind"`
	Name                 string            `yaml:"name"`
	Host                 string            `yaml:"host"`
	Port                 int               `yaml:"port"`
	GPUMemoryUtilization *float64          `yaml:"gpu_memory_utilization,omitempty"`
	DefaultTP            string            `yaml:"default_tp,omitempty"`
	DeviceMemoryBytes    int64             `yaml:"device_memory_bytes"`
	RepoRoot             string            `yaml:"repo_root"`
	Python               string            `yaml:"python"`
	B12XRoot             string            `yaml:"b12x_root"`
	CUDAHome             string            `yaml:"cuda_home"`
	CuteDSLArch          string            `yaml:"cute_dsl_arch"`
	NVRTCLibraryDir      string            `yaml:"nvrtc_library_dir,omitempty"`
	DevicePools          [][]int           `yaml:"device_pools"`
	Environment          map[string]string `yaml:"environment"`
}

type sparkNodeYAML struct {
	SSHHost           string   `yaml:"ssh_host"`
	Address           string   `yaml:"address"`
	EthernetInterface string   `yaml:"ethernet_interface"`
	RDMAInterfaces    []string `yaml:"rdma_interfaces"`
}

type cacheMountYAML struct {
	Source string `yaml:"source"`
	Target string `yaml:"target"`
}

type sparkTopologyYAML struct {
	SchemaVersion         int               `yaml:"schema_version"`
	Kind                  string            `yaml:"kind"`
	Name                  string            `yaml:"name"`
	Host                  string            `yaml:"host"`
	Port                  int               `yaml:"port"`
	GPUMemoryUtilization  *float64          `yaml:"gpu_memory_utilization,omitempty"`
	DefaultTP             string            `yaml:"default_tp,omitempty"`
	DeviceMemoryBytes     int64             `yaml:"device_memory_bytes"`
	RepoRoot              string            `yaml:"repo_root"`
	Python                string            `yaml:"python"`
	RuntimeRepoRoot       string            `yaml:"runtime_repo_root"`
	RuntimePython         string            `yaml:"runtime_python"`
	VLLMBin               string            `yaml:"vllm_bin"`
	B12XRoot              string            `yaml:"b12x_root"`
	RuntimeB12XRoot       string            `yaml:"runtime_b12x_root"`
	Nodes                 []sparkNodeYAML   `yaml:"nodes"`
	MasterPort            int               `yaml:"master_port"`
	Image                 string            `yaml:"image"`
	ContainerNamePrefix   string            `yaml:"container_name_prefix"`
	ContainerMemoryGB     int               `yaml:"container_memory_gb"`
	ContainerMemorySwapGB int               `yaml:"container_memory_swap_gb"`
	ContainerShmGB        int               `yaml:"container_shm_gb"`
	ContainerPidsLimit    int               `yaml:"container_pids_limit"`
	ContainerNofileLimit  int               `yaml:"container_nofile_limit"`
	CacheMounts           []cacheMountYAML  `yaml:"cache_mounts"`
	CUDAHome              string            `yaml:"cuda_home"`
	CuteDSLArch           string            `yaml:"cute_dsl_arch"`
	NCCLDebug             string            `yaml:"nccl_debug"`
	NCCLIBGIDIndex        int               `yaml:"nccl_ib_gid_index"`
	NCCLIBMergeNICs       bool              `yaml:"nccl_ib_merge_nics"`
	DeviceID              int               `yaml:"device_id"`
	NVRTCLibraryDir       string            `yaml:"nvrtc_library_dir,omitempty"`
	Environment           map[string]string `yaml:"environment"`
}

func strictYAML(data []byte, target any) error {
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(target); err != nil {
		return err
	}
	return nil
}

func expandPath(path string) (string, error) {
	if path == "" {
		return "", fmt.Errorf("path must not be empty")
	}
	if path == "~" || len(path) > 1 && path[:2] == "~/" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		if path == "~" {
			path = home
		} else {
			path = filepath.Join(home, path[2:])
		}
	}
	return path, nil
}

func absolutePath(path, base string) (string, error) {
	expanded, err := expandPath(path)
	if err != nil {
		return "", err
	}
	if !filepath.IsAbs(expanded) {
		expanded = filepath.Join(base, expanded)
	}
	return filepath.Abs(expanded)
}

func requiredAbsolutePath(path string) (string, error) {
	expanded, err := expandPath(path)
	if err != nil {
		return "", err
	}
	if !filepath.IsAbs(expanded) {
		return "", fmt.Errorf("must be an absolute path: %s", path)
	}
	return filepath.Clean(expanded), nil
}

func optionalAbsolutePath(path, context string) (string, error) {
	if path == "" {
		return "", nil
	}
	resolved, err := requiredAbsolutePath(path)
	if err != nil {
		return "", fmt.Errorf("%s %w", context, err)
	}
	return resolved, nil
}

func nonEmpty(values ...string) bool {
	for _, value := range values {
		if value == "" {
			return false
		}
	}
	return true
}

func validatePort(port int, name string) error {
	if port < 1 || port > 65535 {
		return fmt.Errorf("%s must be in 1..65535", name)
	}
	return nil
}

func resolveGPUMemoryUtilization(value *float64, context string) (float64, error) {
	if value == nil {
		return DefaultGPUMemoryUtilization, nil
	}
	if *value <= 0 || *value > 1 {
		return 0, fmt.Errorf("%s.gpu_memory_utilization must be in (0, 1]", context)
	}
	return *value, nil
}

func resolveDefaultTP(value, context string) (string, error) {
	switch value {
	case "":
		return DefaultTPFit, nil
	case DefaultTPFit, DefaultTPAll:
		return value, nil
	default:
		return "", fmt.Errorf("%s.default_tp must be fit or all", context)
	}
}

func validateTopologyEnvironment(environment map[string]string, context string) error {
	if environment == nil {
		return fmt.Errorf(
			"%s.environment is required; run lil discover again or add the host tuning block",
			context,
		)
	}
	for name := range environment {
		if !environmentName.MatchString(name) {
			return fmt.Errorf("%s.environment contains an invalid variable name: %q", context, name)
		}
		if derivedEnvironment[name] {
			return fmt.Errorf("%s.environment.%s is derived by the launcher and cannot be configured", context, name)
		}
	}
	return nil
}

func LoadTopology(data []byte, label, baseDir string) (Topology, error) {
	var meta topologyMeta
	if err := yaml.Unmarshal(data, &meta); err != nil {
		return Topology{}, fmt.Errorf("invalid YAML in %s: %w", label, err)
	}
	if meta.SchemaVersion != schemaVersion {
		return Topology{}, fmt.Errorf(
			"%s.schema_version must be %d; got %d",
			label, schemaVersion, meta.SchemaVersion,
		)
	}
	switch meta.Kind {
	case "local":
		return loadLocalTopology(data, label, baseDir)
	case "spark_rdma":
		return loadSparkTopology(data, label, baseDir)
	default:
		return Topology{}, fmt.Errorf(
			"%s.kind must be 'local' or 'spark_rdma'; got %q", label, meta.Kind,
		)
	}
}

func MarshalTopology(topology Topology) ([]byte, error) {
	var value any
	defaultTP, err := resolveDefaultTP(topology.DefaultTP, "topology")
	if err != nil {
		return nil, err
	}
	switch topology.Kind {
	case "local":
		if topology.Local == nil || topology.Spark != nil {
			return nil, fmt.Errorf("local topology has inconsistent payload")
		}
		local := topology.Local
		utilization := topology.MemoryUtilization()
		value = localTopologyYAML{
			SchemaVersion:        schemaVersion,
			Kind:                 "local",
			Name:                 local.Name,
			Host:                 local.Host,
			Port:                 local.Port,
			GPUMemoryUtilization: &utilization,
			DefaultTP:            defaultTP,
			DeviceMemoryBytes:    local.DeviceMemoryBytes,
			RepoRoot:             local.RepoRoot,
			Python:               local.Python,
			B12XRoot:             local.B12XRoot,
			CUDAHome:             local.CUDAHome,
			CuteDSLArch:          local.CuteDSLArch,
			NVRTCLibraryDir:      local.NVRTCLibraryDir,
			DevicePools:          local.DevicePools,
			Environment:          emptyIfNil(local.Environment),
		}
	case "spark_rdma":
		if topology.Spark == nil || topology.Local != nil {
			return nil, fmt.Errorf("Spark/RDMA topology has inconsistent payload")
		}
		spark := topology.Spark
		utilization := topology.MemoryUtilization()
		nodes := make([]sparkNodeYAML, len(spark.Nodes))
		for index, node := range spark.Nodes {
			nodes[index] = sparkNodeYAML{
				SSHHost: node.SSHHost, Address: node.Address,
				EthernetInterface: node.EthernetInterface,
				RDMAInterfaces:    node.RDMAInterfaces,
			}
		}
		mounts := make([]cacheMountYAML, len(spark.CacheMounts))
		for index, mount := range spark.CacheMounts {
			mounts[index] = cacheMountYAML{Source: mount.Source, Target: mount.Target}
		}
		value = sparkTopologyYAML{
			SchemaVersion:         schemaVersion,
			Kind:                  "spark_rdma",
			Name:                  spark.Name,
			Host:                  spark.Host,
			Port:                  spark.Port,
			GPUMemoryUtilization:  &utilization,
			DefaultTP:             defaultTP,
			DeviceMemoryBytes:     spark.DeviceMemoryBytes,
			RepoRoot:              spark.RepoRoot,
			Python:                spark.Python,
			RuntimeRepoRoot:       spark.RuntimeRepoRoot,
			RuntimePython:         spark.RuntimePython,
			VLLMBin:               spark.VLLMBin,
			B12XRoot:              spark.B12XRoot,
			RuntimeB12XRoot:       spark.RuntimeB12XRoot,
			Nodes:                 nodes,
			MasterPort:            spark.MasterPort,
			Image:                 spark.Image,
			ContainerNamePrefix:   spark.ContainerNamePrefix,
			ContainerMemoryGB:     spark.ContainerMemoryGB,
			ContainerMemorySwapGB: spark.ContainerMemorySwapGB,
			ContainerShmGB:        spark.ContainerShmGB,
			ContainerPidsLimit:    spark.ContainerPidsLimit,
			ContainerNofileLimit:  spark.ContainerNofileLimit,
			CacheMounts:           mounts,
			CUDAHome:              spark.CUDAHome,
			CuteDSLArch:           spark.CuteDSLArch,
			NCCLDebug:             spark.NCCLDebug,
			NCCLIBGIDIndex:        spark.NCCLIBGIDIndex,
			NCCLIBMergeNICs:       spark.NCCLIBMergeNICs,
			DeviceID:              spark.DeviceID,
			NVRTCLibraryDir:       spark.NVRTCLibraryDir,
			Environment:           emptyIfNil(spark.Environment),
		}
	default:
		return nil, fmt.Errorf("unsupported topology kind: %q", topology.Kind)
	}
	var node yaml.Node
	if err := node.Encode(value); err != nil {
		return nil, err
	}
	setSequenceItemsFlowStyle(&node, "device_pools")
	var encoded bytes.Buffer
	encoder := yaml.NewEncoder(&encoded)
	encoder.SetIndent(2)
	if err := encoder.Encode(&node); err != nil {
		return nil, err
	}
	if err := encoder.Close(); err != nil {
		return nil, err
	}
	header := "# SPDX-License-Identifier: Apache-2.0\n" +
		"# SPDX-FileCopyrightText: Copyright contributors to the lil project\n\n"
	return append([]byte(header), encoded.Bytes()...), nil
}

func emptyIfNil(environment map[string]string) map[string]string {
	if environment == nil {
		return map[string]string{}
	}
	return environment
}

func setSequenceItemsFlowStyle(node *yaml.Node, key string) {
	if node.Kind == yaml.MappingNode {
		for index := 0; index+1 < len(node.Content); index += 2 {
			if node.Content[index].Value == key && node.Content[index+1].Kind == yaml.SequenceNode {
				for _, item := range node.Content[index+1].Content {
					item.Style = yaml.FlowStyle
				}
			}
		}
	}
	for _, child := range node.Content {
		setSequenceItemsFlowStyle(child, key)
	}
}

func loadLocalTopology(data []byte, label, baseDir string) (Topology, error) {
	var raw localTopologyYAML
	if err := strictYAML(data, &raw); err != nil {
		return Topology{}, fmt.Errorf("invalid topology %s: %w", label, err)
	}
	if !nonEmpty(raw.Name, raw.Host, raw.RepoRoot, raw.Python, raw.B12XRoot,
		raw.CUDAHome, raw.CuteDSLArch) {
		return Topology{}, fmt.Errorf("%s contains an empty required string", label)
	}
	if err := validatePort(raw.Port, label+".port"); err != nil {
		return Topology{}, err
	}
	if raw.DeviceMemoryBytes <= 0 {
		return Topology{}, fmt.Errorf("%s.device_memory_bytes must be positive", label)
	}
	utilization, err := resolveGPUMemoryUtilization(raw.GPUMemoryUtilization, label)
	if err != nil {
		return Topology{}, err
	}
	defaultTP, err := resolveDefaultTP(raw.DefaultTP, label)
	if err != nil {
		return Topology{}, err
	}
	if err := validateTopologyEnvironment(raw.Environment, label); err != nil {
		return Topology{}, err
	}
	if len(raw.DevicePools) == 0 {
		return Topology{}, fmt.Errorf("%s.device_pools must not be empty", label)
	}
	for index, pool := range raw.DevicePools {
		if len(pool) == 0 {
			return Topology{}, fmt.Errorf(
				"%s.device_pools[%d] must be a non-empty integer list", label, index,
			)
		}
		seen := map[int]bool{}
		for _, device := range pool {
			if device < 0 || seen[device] {
				return Topology{}, fmt.Errorf(
					"%s.device_pools[%d] must contain unique non-negative IDs",
					label, index,
				)
			}
			seen[device] = true
		}
	}
	repoRoot, err := absolutePath(raw.RepoRoot, baseDir)
	if err != nil {
		return Topology{}, err
	}
	python, err := absolutePath(raw.Python, repoRoot)
	if err != nil {
		return Topology{}, err
	}
	b12xRoot, err := absolutePath(raw.B12XRoot, repoRoot)
	if err != nil {
		return Topology{}, err
	}
	cudaHome, err := absolutePath(raw.CUDAHome, repoRoot)
	if err != nil {
		return Topology{}, err
	}
	nvrtc, err := optionalAbsolutePath(raw.NVRTCLibraryDir, label+".nvrtc_library_dir")
	if err != nil {
		return Topology{}, err
	}
	return Topology{
		Kind: "local", GPUMemoryUtilization: utilization, DefaultTP: defaultTP,
		Local: &LocalTopology{
			Name: raw.Name, Host: raw.Host, Port: raw.Port,
			DeviceMemoryBytes: raw.DeviceMemoryBytes,
			RepoRoot:          repoRoot, Python: python, B12XRoot: b12xRoot,
			CUDAHome: cudaHome, CuteDSLArch: raw.CuteDSLArch, NVRTCLibraryDir: nvrtc,
			DevicePools: raw.DevicePools, Environment: raw.Environment,
		},
	}, nil
}

func loadSparkTopology(data []byte, label, baseDir string) (Topology, error) {
	var raw sparkTopologyYAML
	if err := strictYAML(data, &raw); err != nil {
		return Topology{}, fmt.Errorf("invalid topology %s: %w", label, err)
	}
	if !nonEmpty(raw.Name, raw.Host, raw.RepoRoot, raw.Python,
		raw.RuntimeRepoRoot, raw.RuntimePython, raw.VLLMBin, raw.B12XRoot,
		raw.RuntimeB12XRoot, raw.Image, raw.ContainerNamePrefix, raw.CUDAHome,
		raw.CuteDSLArch, raw.NCCLDebug) {
		return Topology{}, fmt.Errorf("%s contains an empty required string", label)
	}
	if err := validatePort(raw.Port, label+".port"); err != nil {
		return Topology{}, err
	}
	if err := validatePort(raw.MasterPort, label+".master_port"); err != nil {
		return Topology{}, err
	}
	if raw.DeviceMemoryBytes <= 0 {
		return Topology{}, fmt.Errorf("%s.device_memory_bytes must be positive", label)
	}
	utilization, err := resolveGPUMemoryUtilization(raw.GPUMemoryUtilization, label)
	if err != nil {
		return Topology{}, err
	}
	defaultTP, err := resolveDefaultTP(raw.DefaultTP, label)
	if err != nil {
		return Topology{}, err
	}
	if err := validateTopologyEnvironment(raw.Environment, label); err != nil {
		return Topology{}, err
	}
	if raw.ContainerMemoryGB <= 0 || raw.ContainerMemorySwapGB < raw.ContainerMemoryGB {
		return Topology{}, fmt.Errorf(
			"%s requires positive container memory and swap >= memory", label,
		)
	}
	if raw.ContainerShmGB <= 0 || raw.ContainerPidsLimit <= 0 || raw.ContainerNofileLimit <= 0 {
		return Topology{}, fmt.Errorf(
			"%s requires positive shared-memory, PID, and nofile limits", label,
		)
	}
	if raw.DeviceID < 0 {
		return Topology{}, fmt.Errorf("%s.device_id must be non-negative", label)
	}
	if raw.NCCLIBGIDIndex < 0 {
		return Topology{}, fmt.Errorf("%s.nccl_ib_gid_index must be non-negative", label)
	}
	validDebug := map[string]bool{"VERSION": true, "WARN": true, "INFO": true, "TRACE": true}
	if !validDebug[raw.NCCLDebug] {
		return Topology{}, fmt.Errorf(
			"%s.nccl_debug must be VERSION, WARN, INFO, or TRACE", label,
		)
	}
	if len(raw.Nodes) == 0 {
		return Topology{}, fmt.Errorf("%s.nodes must contain at least one node", label)
	}
	sshHosts := map[string]bool{}
	addresses := map[string]bool{}
	nodes := make([]SparkNode, 0, len(raw.Nodes))
	for index, node := range raw.Nodes {
		if !nonEmpty(node.SSHHost, node.Address, node.EthernetInterface) || len(node.RDMAInterfaces) == 0 {
			return Topology{}, fmt.Errorf("%s.nodes[%d] is incomplete", label, index)
		}
		if sshHosts[node.SSHHost] || addresses[node.Address] {
			return Topology{}, fmt.Errorf("%s.nodes contains duplicate hosts or addresses", label)
		}
		sshHosts[node.SSHHost], addresses[node.Address] = true, true
		interfaces := map[string]bool{}
		for _, iface := range node.RDMAInterfaces {
			if iface == "" || interfaces[iface] {
				return Topology{}, fmt.Errorf("%s.nodes[%d].rdma_interfaces contains an empty or duplicate value", label, index)
			}
			interfaces[iface] = true
		}
		nodes = append(nodes, SparkNode{node.SSHHost, node.Address, node.EthernetInterface, node.RDMAInterfaces})
	}
	targets := map[string]bool{}
	mounts := make([]CacheMount, 0, len(raw.CacheMounts))
	for index, mount := range raw.CacheMounts {
		source, err := requiredAbsolutePath(mount.Source)
		if err != nil {
			return Topology{}, fmt.Errorf("%s.cache_mounts[%d].source %w", label, index, err)
		}
		target, err := requiredAbsolutePath(mount.Target)
		if err != nil {
			return Topology{}, fmt.Errorf("%s.cache_mounts[%d].target %w", label, index, err)
		}
		if targets[target] {
			return Topology{}, fmt.Errorf("%s.cache_mounts contains duplicate container targets", label)
		}
		targets[target] = true
		mounts = append(mounts, CacheMount{source, target})
	}
	repoRoot, err := absolutePath(raw.RepoRoot, baseDir)
	if err != nil {
		return Topology{}, err
	}
	python, err := absolutePath(raw.Python, repoRoot)
	if err != nil {
		return Topology{}, err
	}
	b12xRoot, err := absolutePath(raw.B12XRoot, repoRoot)
	if err != nil {
		return Topology{}, err
	}
	abs := func(value string) (string, error) { return requiredAbsolutePath(value) }
	runtimeRepo, err := abs(raw.RuntimeRepoRoot)
	if err != nil {
		return Topology{}, err
	}
	runtimePython, err := abs(raw.RuntimePython)
	if err != nil {
		return Topology{}, err
	}
	vllmBin, err := abs(raw.VLLMBin)
	if err != nil {
		return Topology{}, err
	}
	runtimeB12X, err := abs(raw.RuntimeB12XRoot)
	if err != nil {
		return Topology{}, err
	}
	cudaHome, err := abs(raw.CUDAHome)
	if err != nil {
		return Topology{}, err
	}
	nvrtc, err := optionalAbsolutePath(raw.NVRTCLibraryDir, label+".nvrtc_library_dir")
	if err != nil {
		return Topology{}, err
	}
	return Topology{
		Kind: "spark_rdma", GPUMemoryUtilization: utilization, DefaultTP: defaultTP,
		Spark: &SparkRDMATopology{
			Name: raw.Name, Host: raw.Host, Port: raw.Port,
			DeviceMemoryBytes: raw.DeviceMemoryBytes,
			RepoRoot:          repoRoot, Python: python,
			RuntimeRepoRoot: runtimeRepo, RuntimePython: runtimePython,
			VLLMBin: vllmBin, B12XRoot: b12xRoot, RuntimeB12XRoot: runtimeB12X,
			Nodes: nodes, MasterPort: raw.MasterPort, Image: raw.Image,
			ContainerNamePrefix:   raw.ContainerNamePrefix,
			ContainerMemoryGB:     raw.ContainerMemoryGB,
			ContainerMemorySwapGB: raw.ContainerMemorySwapGB,
			ContainerShmGB:        raw.ContainerShmGB,
			ContainerPidsLimit:    raw.ContainerPidsLimit,
			ContainerNofileLimit:  raw.ContainerNofileLimit,
			CacheMounts:           mounts, CUDAHome: cudaHome,
			CuteDSLArch: raw.CuteDSLArch, NCCLDebug: raw.NCCLDebug,
			NCCLIBGIDIndex:  raw.NCCLIBGIDIndex,
			NCCLIBMergeNICs: raw.NCCLIBMergeNICs,
			DeviceID:        raw.DeviceID,
			NVRTCLibraryDir: nvrtc,
			Environment:     raw.Environment,
		},
	}, nil
}
