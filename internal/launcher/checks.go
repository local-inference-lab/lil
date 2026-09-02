// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package launcher

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

const parseVLLMArgv = `
import json
import sys

try:
    from vllm.entrypoints.launchers.cli_args import (
        create_parser_for_docs,
        validate_parsed_serve_args,
    )
except ModuleNotFoundError:
    from vllm.entrypoints.openai.cli_args import (
        create_parser_for_docs,
        validate_parsed_serve_args,
    )

parser = create_parser_for_docs()
args = parser.parse_args(json.loads(sys.argv[1]))
validate_parsed_serve_args(args)
`

const sparkNodeProbe = `
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

config = json.loads(sys.argv[1])
errors = []
notes = []

for path, kind in config["paths"]:
    candidate = Path(path)
    if kind == "executable" and not os.access(candidate, os.X_OK):
        errors.append(f"not executable: {candidate}")
    elif kind == "file" and not candidate.is_file():
        errors.append(f"missing file: {candidate}")
    elif kind == "directory" and not candidate.is_dir():
        errors.append(f"missing directory: {candidate}")

interface = config["ethernet_interface"]
if not Path("/sys/class/net", interface).exists():
    errors.append(f"missing network interface: {interface}")
else:
    address = subprocess.run(
        ["ip", "-o", "-4", "addr", "show", "dev", interface],
        check=False, capture_output=True, text=True,
    )
    if config["address"] not in address.stdout:
        errors.append(f"{interface} does not own {config['address']}")

for peer in config["peer_addresses"]:
    reply = subprocess.run(
        ["ping", "-c", "1", "-W", "2", peer],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if reply.returncode:
        errors.append(f"cannot reach RDMA peer address: {peer}")

rdma = subprocess.run(["rdma", "link", "show"], check=False, capture_output=True, text=True)
for hca in config["rdma_interfaces"]:
    if not Path("/sys/class/infiniband", hca).exists():
        errors.append(f"missing RDMA interface: {hca}")
    elif f"link {hca}/" not in rdma.stdout or "state ACTIVE" not in next(
        (line for line in rdma.stdout.splitlines() if f"link {hca}/" in line), ""
    ):
        errors.append(f"RDMA interface is not active: {hca}")

gpu = subprocess.run(
    ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
    check=False, capture_output=True, text=True,
)
gpu_ids = {line.strip() for line in gpu.stdout.splitlines()}
if gpu.returncode or str(config["device_id"]) not in gpu_ids:
    errors.append(f"GPU {config['device_id']} is unavailable")

image = subprocess.run(
    ["docker", "image", "inspect", config["image"]],
    check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
if image.returncode:
    errors.append(f"missing Docker image: {config['image']}")

container = subprocess.run(
    ["docker", "container", "inspect", "--format", "{{.State.Running}}", config["container_name"]],
    check=False, capture_output=True, text=True,
)
if not container.returncode:
    if container.stdout.strip() == "true":
        errors.append(f"container is already running: {config['container_name']}")
    else:
        notes.append(f"exited container {config['container_name']} will be removed before launch")

for port in config["ports"]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("0.0.0.0", port))
        except OSError as exc:
            errors.append(f"port {port} is unavailable: {exc}")

print(json.dumps({"errors": errors, "notes": notes}))
raise SystemExit(bool(errors))
`

const sourceDigestScript = `
import hashlib
import os
import sys
from pathlib import Path

digest = hashlib.sha256()
for root_index, raw_root in enumerate(sys.argv[1:]):
    root = Path(raw_root)
    for path in sorted(root.rglob("*"), key=lambda item: os.fsencode(item)):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix in {
            ".a", ".o", ".pyc", ".pyo", ".so",
        }:
            continue
        digest.update(f"{root_index}:{relative}".encode())
        if path.is_symlink():
            digest.update(b"L")
            digest.update(os.readlink(path).encode())
        elif path.is_file():
            digest.update(b"F")
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
print(digest.hexdigest())
`

func pathCheck(name, path string, executable bool) CheckResult {
	info, err := os.Stat(path)
	if err != nil {
		return CheckResult{name, "FAIL", "missing: " + path}
	}
	if executable && info.Mode().Perm()&0111 == 0 {
		return CheckResult{name, "FAIL", "not executable: " + path}
	}
	return CheckResult{name, "PASS", path}
}

func modelFactsCheck(spec LaunchSpec) CheckResult {
	facts, _ := spec.Metadata["checkpoint_facts"].(map[string]any)
	architectures, _ := facts["architectures"].([]string)
	heads, _ := facts["attention_heads"].(int)
	source, _ := facts["source"].(string)
	if spec.CheckpointPath != nil {
		local, err := LoadCheckpointFacts(*spec.CheckpointPath)
		if err != nil {
			return CheckResult{"model architecture", "FAIL", err.Error()}
		}
		if !sameArchitectures(local.Architectures, architectures) || local.AttentionHeads != heads {
			return CheckResult{"model architecture", "FAIL", fmt.Sprintf("checkpoint changed since the launch was resolved: %v with %d heads", local.Architectures, local.AttentionHeads)}
		}
	}
	return CheckResult{"model architecture", "PASS", fmt.Sprintf("%s, %d heads, TP=%d (%s)", strings.Join(architectures, ","), heads, spec.TPSize, source)}
}

func environmentForSpec(spec LaunchSpec) []string {
	environment := environmentMap(os.Environ())
	for name, value := range spec.RuntimeEnvironment {
		environment[name] = value
	}
	if spec.Topology.Spark != nil {
		environment["PYTHONPATH"] = strings.Join([]string{spec.Topology.Spark.RepoRoot, spec.Topology.Spark.B12XRoot}, string(os.PathListSeparator))
	}
	for _, name := range spec.UnsetEnvironment {
		delete(environment, name)
	}
	return environmentList(environment)
}

func environmentMap(values []string) map[string]string {
	result := map[string]string{}
	for _, value := range values {
		name, contents, ok := strings.Cut(value, "=")
		if ok {
			result[name] = contents
		}
	}
	return result
}

func environmentList(values map[string]string) []string {
	keys := sortedMapKeys(values)
	result := make([]string, 0, len(keys))
	for _, key := range keys {
		result = append(result, key+"="+values[key])
	}
	return result
}

func importCheck(ctx context.Context, spec LaunchSpec) CheckResult {
	argv := []string{spec.Topology.Python(), "-c", "import b12x, vllm, yaml"}
	output, status, err := runCommand(ctx, 30*time.Second, spec.Topology.RepoRoot(), environmentForSpec(spec), argv)
	if err != nil {
		return CheckResult{"Python imports", "FAIL", err.Error()}
	}
	if status != 0 {
		return CheckResult{"Python imports", "FAIL", lastOutputLine(output, fmt.Sprintf("exit status %d", status))}
	}
	return CheckResult{"Python imports", "PASS", "b12x, vllm, and yaml"}
}

func serveArguments(argv []string) ([]string, bool) {
	for index, argument := range argv {
		if argument == "serve" {
			return argv[index+1:], true
		}
	}
	return nil, false
}

func vllmArgvCheck(ctx context.Context, spec LaunchSpec) CheckResult {
	arguments, ok := serveArguments(spec.VLLMArgv)
	if !ok {
		return CheckResult{"vLLM argv", "FAIL", "serve subcommand is absent"}
	}
	encoded, _ := json.Marshal(arguments)
	argv := []string{spec.Topology.Python(), "-c", parseVLLMArgv, string(encoded)}
	output, status, err := runCommand(ctx, 30*time.Second, spec.Topology.RepoRoot(), environmentForSpec(spec), argv)
	if err != nil {
		return CheckResult{"vLLM argv", "FAIL", err.Error()}
	}
	if status != 0 {
		return CheckResult{"vLLM argv", "FAIL", lastOutputLine(output, fmt.Sprintf("exit status %d", status))}
	}
	return CheckResult{"vLLM argv", "PASS", fmt.Sprintf("%d arguments accepted by the installed serve parser", len(arguments))}
}

func gpuCheck(ctx context.Context, spec LaunchSpec) CheckResult {
	if spec.Topology.Local == nil || spec.DeviceIDs == nil {
		return CheckResult{"GPU inventory", "PASS", "one configured GPU per Spark/RDMA node"}
	}
	nvidiaSMI, err := exec.LookPath("nvidia-smi")
	if err != nil {
		return CheckResult{"GPU inventory", "FAIL", "nvidia-smi is not on PATH"}
	}
	output, status, err := runCommand(ctx, 10*time.Second, "", nil, []string{nvidiaSMI, "--query-gpu=index,name", "--format=csv,noheader"})
	if err != nil {
		return CheckResult{"GPU inventory", "FAIL", err.Error()}
	}
	if status != 0 {
		return CheckResult{"GPU inventory", "FAIL", lastOutputLine(output, "nvidia-smi failed")}
	}
	available := map[int]bool{}
	for _, line := range strings.Split(string(output), "\n") {
		field := strings.TrimSpace(strings.SplitN(line, ",", 2)[0])
		if value, err := strconv.Atoi(field); err == nil {
			available[value] = true
		}
	}
	var missing []int
	for _, id := range spec.DeviceIDs {
		if !available[id] {
			missing = append(missing, id)
		}
	}
	if len(missing) > 0 {
		return CheckResult{"GPU inventory", "FAIL", fmt.Sprintf("GPU IDs are unavailable: %v", missing)}
	}
	parts := make([]string, len(spec.DeviceIDs))
	for index, id := range spec.DeviceIDs {
		parts[index] = strconv.Itoa(id)
	}
	return CheckResult{"GPU inventory", "PASS", "physical devices " + strings.Join(parts, ",")}
}

func portCheck(spec LaunchSpec) CheckResult {
	listener, err := net.Listen("tcp", net.JoinHostPort(spec.Host, strconv.Itoa(spec.Port)))
	if err != nil {
		return CheckResult{"API port", "FAIL", fmt.Sprintf("cannot bind %s:%d: %v", spec.Host, spec.Port, err)}
	}
	listener.Close()
	return CheckResult{"API port", "PASS", fmt.Sprintf("%s:%d is available", spec.Host, spec.Port)}
}

type nodeProbeResult struct {
	Errors []string `json:"errors"`
	Notes  []string `json:"notes"`
}

func sparkNodeCheck(ctx context.Context, spec LaunchSpec, index int) CheckResult {
	topology := spec.Topology.Spark
	launch := spec.SparkNodes[index]
	name := fmt.Sprintf("Spark node %d", index)
	peers := []string{}
	for _, candidate := range spec.SparkNodes {
		if candidate.Rank != launch.Rank {
			peers = append(peers, candidate.Node.Address)
		}
	}
	paths := [][]string{
		{topology.RuntimePython, "executable"}, {topology.VLLMBin, "executable"},
		{filepath.Join(topology.RuntimeRepoRoot, "vllm", "__init__.py"), "file"},
		{filepath.Join(topology.RuntimeB12XRoot, "b12x", "__init__.py"), "file"},
		{filepath.Join(topology.CUDAHome, "bin", "ptxas"), "executable"},
		{"/dev/infiniband", "directory"},
	}
	for _, mount := range topology.CacheMounts {
		paths = append(paths, []string{mount.Source, "directory"})
	}
	if profileDir, ok := ProfilerOutputDir(launch.VLLMArgv); ok {
		paths = append(paths, []string{profileDir, "directory"})
	}
	if filepath.IsAbs(spec.ModelSource) {
		paths = append(paths, []string{filepath.Join(spec.ModelSource, "config.json"), "file"})
	}
	ports := []int{}
	if index == 0 {
		ports = []int{spec.Port, topology.MasterPort}
	}
	payload := map[string]any{
		"address": launch.Node.Address, "peer_addresses": peers,
		"ethernet_interface": launch.Node.EthernetInterface,
		"rdma_interfaces":    launch.Node.RDMAInterfaces, "device_id": topology.DeviceID,
		"image": topology.Image, "container_name": launch.ContainerName,
		"ports": ports, "paths": paths,
	}
	encoded, _ := json.Marshal(payload)
	output, status, err := remoteRun(ctx, launch.Node.SSHHost, []string{topology.RuntimePython, "-c", sparkNodeProbe, string(encoded)}, 60*time.Second)
	if err != nil {
		return CheckResult{name, "FAIL", fmt.Sprintf("%s: %v", launch.Node.SSHHost, err)}
	}
	var result nodeProbeResult
	if decodeErr := json.Unmarshal([]byte(lastOutputLine(output, "{}")), &result); decodeErr != nil || status != 0 && len(result.Errors) == 0 {
		return CheckResult{name, "FAIL", fmt.Sprintf("%s: %s", launch.Node.SSHHost, lastOutputLine(output, "probe failed"))}
	}
	if len(result.Errors) > 0 {
		return CheckResult{name, "FAIL", fmt.Sprintf("%s: %s", launch.Node.SSHHost, strings.Join(result.Errors, "; "))}
	}
	detail := fmt.Sprintf("%s (%s), GPU %d, %s", launch.Node.SSHHost, launch.Node.Address, topology.DeviceID, strings.Join(launch.Node.RDMAInterfaces, ","))
	if len(result.Notes) > 0 {
		detail += "; " + strings.Join(result.Notes, "; ")
	}
	return CheckResult{name, "PASS", detail}
}

// sparkRuntimeCheck parses the rank's argv inside the launch image with the
// launch mounts and environment, so the interpreter, imports, and parser that
// are validated are the ones the container will run.
func sparkRuntimeCheck(ctx context.Context, spec LaunchSpec, index int) CheckResult {
	topology := spec.Topology.Spark
	launch := spec.SparkNodes[index]
	name := fmt.Sprintf("Spark runtime %d", index)
	arguments, ok := serveArguments(launch.VLLMArgv)
	if !ok {
		return CheckResult{name, "FAIL", "serve subcommand is absent"}
	}
	encoded, _ := json.Marshal(arguments)
	argv := sparkDockerRun(topology, sparkMounts(topology, spec.CheckpointPath, launch.VLLMArgv), launch.RuntimeEnvironment, "", nil)
	argv = append(argv, topology.RuntimePython, "-c", "import b12x, yaml\n"+parseVLLMArgv, string(encoded))
	output, status, err := remoteRun(ctx, launch.Node.SSHHost, argv, 5*time.Minute)
	if err != nil {
		return CheckResult{name, "FAIL", fmt.Sprintf("%s: %v", launch.Node.SSHHost, err)}
	}
	if status != 0 {
		return CheckResult{name, "FAIL", fmt.Sprintf("%s: %s", launch.Node.SSHHost, lastOutputLine(output, "container runtime check failed"))}
	}
	return CheckResult{name, "PASS", fmt.Sprintf("%s: imports and rank %d argv accepted inside %s", launch.Node.SSHHost, index, topology.Image)}
}

func ignoredDigestPath(path string) bool {
	for _, part := range strings.Split(filepath.Clean(path), string(os.PathSeparator)) {
		if part == "__pycache__" {
			return true
		}
	}
	switch filepath.Ext(path) {
	case ".a", ".o", ".pyc", ".pyo", ".so":
		return true
	}
	return false
}

func localSourceDigest(roots []string) (string, error) {
	digest := sha256.New()
	for rootIndex, root := range roots {
		err := filepath.WalkDir(root, func(path string, entry os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if path == root {
				return nil
			}
			relative, err := filepath.Rel(root, path)
			if err != nil {
				return err
			}
			if ignoredDigestPath(relative) {
				if entry.IsDir() {
					return filepath.SkipDir
				}
				return nil
			}
			fmt.Fprintf(digest, "%d:%s", rootIndex, relative)
			info, err := entry.Info()
			if err != nil {
				return err
			}
			if info.Mode()&os.ModeSymlink != 0 {
				target, err := os.Readlink(path)
				if err != nil {
					return err
				}
				digest.Write([]byte("L"))
				digest.Write([]byte(target))
				return nil
			}
			if info.Mode().IsRegular() {
				digest.Write([]byte("F"))
				source, err := os.Open(path)
				if err != nil {
					return err
				}
				_, copyErr := io.Copy(digest, source)
				closeErr := source.Close()
				if copyErr != nil {
					return copyErr
				}
				return closeErr
			}
			return nil
		})
		if err != nil {
			return "", err
		}
	}
	return hex.EncodeToString(digest.Sum(nil)), nil
}

func sparkSourceCheck(ctx context.Context, spec LaunchSpec) CheckResult {
	topology := spec.Topology.Spark
	digest, err := localSourceDigest([]string{filepath.Join(topology.RepoRoot, "vllm"), filepath.Join(topology.B12XRoot, "b12x")})
	if err != nil {
		return CheckResult{"Spark source parity", "FAIL", err.Error()}
	}
	type hostDigest struct{ host, digest string }
	digests := []hostDigest{{"controller", digest}}
	for _, launch := range spec.SparkNodes {
		output, status, err := remoteRun(ctx, launch.Node.SSHHost, []string{topology.RuntimePython, "-c", sourceDigestScript, filepath.Join(topology.RuntimeRepoRoot, "vllm"), filepath.Join(topology.RuntimeB12XRoot, "b12x")}, 120*time.Second)
		if err != nil {
			return CheckResult{"Spark source parity", "FAIL", fmt.Sprintf("%s: %v", launch.Node.SSHHost, err)}
		}
		value := strings.TrimSpace(string(output))
		if status != 0 || len(value) != 64 {
			return CheckResult{"Spark source parity", "FAIL", fmt.Sprintf("%s: %s", launch.Node.SSHHost, lastOutputLine(output, "digest failed"))}
		}
		digests = append(digests, hostDigest{launch.Node.SSHHost, value})
	}
	unique := map[string]bool{}
	for _, item := range digests {
		unique[item.digest] = true
	}
	if len(unique) != 1 {
		parts := make([]string, len(digests))
		for index, item := range digests {
			parts[index] = item.host + "=" + item.digest[:12]
		}
		return CheckResult{"Spark source parity", "FAIL", strings.Join(parts, ", ")}
	}
	return CheckResult{"Spark source parity", "PASS", "vLLM+B12X Python digest " + digest[:12] + " on all nodes (native extensions are not compared)"}
}

func sparkChecks(ctx context.Context, spec LaunchSpec) []CheckResult {
	var results []CheckResult
	nodesReady := true
	for index := range spec.SparkNodes {
		node := sparkNodeCheck(ctx, spec, index)
		results = append(results, node)
		nodesReady = nodesReady && !node.Failed()
		if !node.Failed() {
			results = append(results, sparkRuntimeCheck(ctx, spec, index))
		}
	}
	if nodesReady {
		results = append(results, sparkSourceCheck(ctx, spec))
	}
	return results
}

// RecheckSparkNodes repeats only the per-node host probes. It runs after a
// long cache update so that port and container conflicts that appeared in
// the meantime are caught before containers start.
func RecheckSparkNodes(ctx context.Context, spec LaunchSpec) []CheckResult {
	results := make([]CheckResult, 0, len(spec.SparkNodes))
	for index := range spec.SparkNodes {
		results = append(results, sparkNodeCheck(ctx, spec, index))
	}
	return results
}

func RunChecks(ctx context.Context, spec LaunchSpec, checkPort bool) []CheckResult {
	results := []CheckResult{
		pathCheck("runtime executable", spec.Topology.Python(), true),
		pathCheck("vLLM source", filepath.Join(spec.Topology.RepoRoot(), "vllm", "__init__.py"), false),
		pathCheck("B12X source", filepath.Join(spec.Topology.B12XRoot(), "b12x", "__init__.py"), false),
	}
	if spec.Topology.Local != nil {
		results = append(results, pathCheck("CUDA ptxas", filepath.Join(spec.Topology.CUDAHome(), "bin", "ptxas"), true))
	}
	results = append(results, modelFactsCheck(spec), gpuCheck(ctx, spec), importCheck(ctx, spec), vllmArgvCheck(ctx, spec))
	if checkPort && spec.Topology.Local != nil {
		results = append(results, portCheck(spec))
	}
	if spec.Topology.Spark != nil {
		results = append(results, sparkChecks(ctx, spec)...)
	}
	return results
}

func SortCheckResults(results []CheckResult) {
	sort.SliceStable(results, func(i, j int) bool { return results[i].Name < results[j].Name })
}
