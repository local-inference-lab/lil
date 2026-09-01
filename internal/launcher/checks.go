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

if config["peer_address"] is not None:
    peer = subprocess.run(
        ["ping", "-c", "1", "-W", "2", config["peer_address"]],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if peer.returncode:
        errors.append(f"cannot reach RDMA peer address: {config['peer_address']}")

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
    ["docker", "container", "inspect", config["container_name"]],
    check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
if not container.returncode:
    errors.append(f"container already exists: {config['container_name']}")

for port in config["ports"]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("0.0.0.0", port))
        except OSError as exc:
            errors.append(f"port {port} is unavailable: {exc}")

print(json.dumps(errors))
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

func modelConfigCheck(spec LaunchSpec) CheckResult {
	if spec.CheckpointPath == nil {
		return CheckResult{"model config", "PASS", fmt.Sprintf("%q will be updated in the Hugging Face cache by lil before launch", spec.ModelSource)}
	}
	path := filepath.Join(*spec.CheckpointPath, "config.json")
	data, err := os.ReadFile(path)
	if err != nil {
		return CheckResult{"model config", "FAIL", fmt.Sprintf("cannot read %s: %v", path, err)}
	}
	var config map[string]any
	if err := json.Unmarshal(data, &config); err != nil {
		return CheckResult{"model config", "FAIL", fmt.Sprintf("cannot read %s: %v", path, err)}
	}
	rawArchitectures, ok := config["architectures"].([]any)
	if !ok {
		return CheckResult{"model architecture", "FAIL", "architectures is absent in " + path}
	}
	architectures := make([]string, 0, len(rawArchitectures))
	for _, raw := range rawArchitectures {
		if value, ok := raw.(string); ok {
			architectures = append(architectures, value)
		}
	}
	expected := stringSet(spec.Model.ExpectedArchitectures...)
	match := false
	for _, architecture := range architectures {
		if !expected[architecture] {
			return CheckResult{"model architecture", "FAIL", fmt.Sprintf("expected %v, got %v", spec.Model.ExpectedArchitectures, architectures)}
		}
		match = true
	}
	if !match {
		return CheckResult{"model architecture", "FAIL", fmt.Sprintf("expected %v, got %v", spec.Model.ExpectedArchitectures, architectures)}
	}
	model := modelConfig(config)
	heads, ok := jsonInteger(model["num_attention_heads"])
	if !ok || heads != int64(spec.Model.AttentionHeads) {
		return CheckResult{"model architecture", "FAIL", fmt.Sprintf("expected %d attention heads, got %v", spec.Model.AttentionHeads, model["num_attention_heads"])}
	}
	if heads%int64(spec.TPSize) != 0 {
		return CheckResult{"model architecture", "FAIL", fmt.Sprintf("%d attention heads are not divisible by TP=%d", heads, spec.TPSize)}
	}
	return CheckResult{"model architecture", "PASS", fmt.Sprintf("%s, %d heads, TP=%d", architectures[0], heads, spec.TPSize)}
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

func runCommand(timeout time.Duration, directory string, environment []string, argv []string) ([]byte, int, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	command := exec.CommandContext(ctx, argv[0], argv[1:]...)
	command.Dir = directory
	if environment != nil {
		command.Env = environment
	}
	output, err := command.CombinedOutput()
	if ctx.Err() == context.DeadlineExceeded {
		return output, -1, fmt.Errorf("command timed out after %s", timeout)
	}
	if err == nil {
		return output, 0, nil
	}
	if exit, ok := err.(*exec.ExitError); ok {
		return output, exit.ExitCode(), nil
	}
	return output, -1, err
}

func lastOutputLine(output []byte, fallback string) string {
	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	if len(lines) == 0 || lines[0] == "" {
		return fallback
	}
	return lines[len(lines)-1]
}

func importCheck(spec LaunchSpec) CheckResult {
	argv := []string{spec.Topology.Python(), "-c", "import b12x, vllm, yaml"}
	output, status, err := runCommand(30*time.Second, spec.Topology.RepoRoot(), environmentForSpec(spec), argv)
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

func vllmArgvCheck(spec LaunchSpec) CheckResult {
	arguments, ok := serveArguments(spec.VLLMArgv)
	if !ok {
		return CheckResult{"vLLM argv", "FAIL", "serve subcommand is absent"}
	}
	encoded, _ := json.Marshal(arguments)
	argv := []string{spec.Topology.Python(), "-c", parseVLLMArgv, string(encoded)}
	output, status, err := runCommand(30*time.Second, spec.Topology.RepoRoot(), environmentForSpec(spec), argv)
	if err != nil {
		return CheckResult{"vLLM argv", "FAIL", err.Error()}
	}
	if status != 0 {
		return CheckResult{"vLLM argv", "FAIL", lastOutputLine(output, fmt.Sprintf("exit status %d", status))}
	}
	return CheckResult{"vLLM argv", "PASS", fmt.Sprintf("%d arguments accepted by the installed serve parser", len(arguments))}
}

func gpuCheck(spec LaunchSpec) CheckResult {
	if spec.Topology.Local == nil || spec.DeviceIDs == nil {
		return CheckResult{"GPU inventory", "PASS", "one configured GPU per Spark/RDMA node"}
	}
	nvidiaSMI, err := exec.LookPath("nvidia-smi")
	if err != nil {
		return CheckResult{"GPU inventory", "FAIL", "nvidia-smi is not on PATH"}
	}
	output, status, err := runCommand(10*time.Second, "", nil, []string{nvidiaSMI, "--query-gpu=index,name", "--format=csv,noheader"})
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

func remoteRun(host string, argv []string, timeout time.Duration) ([]byte, int, error) {
	return runCommand(timeout, "", nil, RemoteArgv(host, argv))
}

func sparkNodeCheck(spec LaunchSpec, index int) CheckResult {
	topology := spec.Topology.Spark
	launch := spec.SparkNodes[index]
	var peer any
	for _, candidate := range spec.SparkNodes {
		if candidate.Rank != launch.Rank {
			peer = candidate.Node.Address
			break
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
		"address": launch.Node.Address, "peer_address": peer,
		"ethernet_interface": launch.Node.EthernetInterface,
		"rdma_interfaces":    launch.Node.RDMAInterfaces, "device_id": topology.DeviceID,
		"image": topology.Image, "container_name": launch.ContainerName,
		"ports": ports, "paths": paths,
	}
	encoded, _ := json.Marshal(payload)
	output, status, err := remoteRun(launch.Node.SSHHost, []string{topology.RuntimePython, "-c", sparkNodeProbe, string(encoded)}, 60*time.Second)
	if err != nil {
		return CheckResult{fmt.Sprintf("Spark node %d", index), "FAIL", fmt.Sprintf("%s: %v", launch.Node.SSHHost, err)}
	}
	if status != 0 {
		return CheckResult{fmt.Sprintf("Spark node %d", index), "FAIL", fmt.Sprintf("%s: %s", launch.Node.SSHHost, lastOutputLine(output, "probe failed"))}
	}
	return CheckResult{fmt.Sprintf("Spark node %d", index), "PASS", fmt.Sprintf("%s (%s), GPU %d, %s", launch.Node.SSHHost, launch.Node.Address, topology.DeviceID, strings.Join(launch.Node.RDMAInterfaces, ","))}
}

func sparkRuntimeCheck(spec LaunchSpec, index int) CheckResult {
	topology := spec.Topology.Spark
	launch := spec.SparkNodes[index]
	arguments, ok := serveArguments(launch.VLLMArgv)
	if !ok {
		return CheckResult{fmt.Sprintf("Spark runtime %d", index), "FAIL", "serve subcommand is absent"}
	}
	encoded, _ := json.Marshal(arguments)
	environment := []string{"/usr/bin/env"}
	for _, name := range sortedMapKeys(launch.RuntimeEnvironment) {
		environment = append(environment, name+"="+launch.RuntimeEnvironment[name])
	}
	environment = append(environment, topology.RuntimePython, "-c", "import b12x, yaml\n"+parseVLLMArgv, string(encoded))
	output, status, err := remoteRun(launch.Node.SSHHost, environment, 60*time.Second)
	if err != nil {
		return CheckResult{fmt.Sprintf("Spark runtime %d", index), "FAIL", fmt.Sprintf("%s: %v", launch.Node.SSHHost, err)}
	}
	if status != 0 {
		return CheckResult{fmt.Sprintf("Spark runtime %d", index), "FAIL", fmt.Sprintf("%s: %s", launch.Node.SSHHost, lastOutputLine(output, "runtime parser failed"))}
	}
	return CheckResult{fmt.Sprintf("Spark runtime %d", index), "PASS", fmt.Sprintf("%s: imports and rank %d argv accepted", launch.Node.SSHHost, index)}
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

func sparkSourceCheck(spec LaunchSpec) CheckResult {
	topology := spec.Topology.Spark
	digest, err := localSourceDigest([]string{filepath.Join(topology.RepoRoot, "vllm"), filepath.Join(topology.B12XRoot, "b12x")})
	if err != nil {
		return CheckResult{"Spark source parity", "FAIL", err.Error()}
	}
	type hostDigest struct{ host, digest string }
	digests := []hostDigest{{"controller", digest}}
	for _, launch := range spec.SparkNodes {
		output, status, err := remoteRun(launch.Node.SSHHost, []string{topology.RuntimePython, "-c", sourceDigestScript, filepath.Join(topology.RuntimeRepoRoot, "vllm"), filepath.Join(topology.RuntimeB12XRoot, "b12x")}, 120*time.Second)
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
	return CheckResult{"Spark source parity", "PASS", "vLLM+B12X digest " + digest[:12] + " on all nodes"}
}

func sparkChecks(spec LaunchSpec) []CheckResult {
	var results []CheckResult
	nodesReady := true
	for index := range spec.SparkNodes {
		node := sparkNodeCheck(spec, index)
		results = append(results, node)
		nodesReady = nodesReady && !node.Failed()
		if !node.Failed() {
			results = append(results, sparkRuntimeCheck(spec, index))
		}
	}
	if nodesReady {
		results = append(results, sparkSourceCheck(spec))
	}
	return results
}

func RunChecks(spec LaunchSpec, checkPort bool) []CheckResult {
	results := []CheckResult{
		pathCheck("runtime executable", spec.Topology.Python(), true),
		pathCheck("vLLM source", filepath.Join(spec.Topology.RepoRoot(), "vllm", "__init__.py"), false),
		pathCheck("B12X source", filepath.Join(spec.Topology.B12XRoot(), "b12x", "__init__.py"), false),
	}
	if spec.Topology.Local != nil {
		results = append(results, pathCheck("CUDA ptxas", filepath.Join(spec.Topology.CUDAHome(), "bin", "ptxas"), true))
	}
	results = append(results, modelConfigCheck(spec), gpuCheck(spec), importCheck(spec), vllmArgvCheck(spec))
	if checkPort && spec.Topology.Local != nil {
		results = append(results, portCheck(spec))
	}
	if spec.Topology.Spark != nil {
		results = append(results, sparkChecks(spec)...)
	}
	return results
}

func SortCheckResults(results []CheckResult) {
	sort.SliceStable(results, func(i, j int) bool { return results[i].Name < results[j].Name })
}
