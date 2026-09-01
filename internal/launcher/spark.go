// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package launcher

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

func ResolveSparkTarget(profile ModelProfile, topology Topology, requestedTP *int, requestedPort *int) (SparkTarget, error) {
	if topology.Spark == nil {
		return SparkTarget{}, fmt.Errorf("cluster operations require a Spark/RDMA topology")
	}
	tpSize, err := resolveTPSize(profile, topology, requestedTP)
	if err != nil {
		return SparkTarget{}, err
	}
	port := topology.Port()
	if requestedPort != nil {
		port = *requestedPort
	}
	if err := validatePort(port, "port"); err != nil {
		return SparkTarget{}, err
	}
	return SparkTarget{
		Topology: topology.Spark, ProfileName: profile.Name, TPSize: tpSize,
		Port: port, ContainerName: sparkContainerName(topology.Spark, profile.Name, tpSize),
		Nodes: append([]SparkNode(nil), topology.Spark.Nodes[:tpSize]...),
	}, nil
}

type dockerContainerState struct {
	Running   bool   `json:"Running"`
	Status    string `json:"Status"`
	StartedAt string `json:"StartedAt"`
	ExitCode  int    `json:"ExitCode"`
}

func InspectSparkTarget(target SparkTarget) ([]SparkNodeStatus, error) {
	statuses := make([]SparkNodeStatus, 0, len(target.Nodes))
	for rank, node := range target.Nodes {
		output, status, err := remoteRun(
			node.SSHHost,
			[]string{"docker", "container", "inspect", "--format", "{{json .State}}", target.ContainerName},
			15*time.Second,
		)
		if err != nil {
			return nil, fmt.Errorf("rank %d on %s: %w", rank, node.SSHHost, err)
		}
		if status != 0 {
			detail := strings.TrimSpace(string(output))
			if strings.Contains(detail, "No such object") || strings.Contains(detail, "No such container") {
				statuses = append(statuses, SparkNodeStatus{Node: node, Rank: rank, Status: "absent"})
				continue
			}
			return nil, fmt.Errorf("rank %d on %s: %s", rank, node.SSHHost, detail)
		}
		var state dockerContainerState
		if err := json.Unmarshal(output, &state); err != nil {
			return nil, fmt.Errorf("rank %d on %s returned invalid Docker state: %w", rank, node.SSHHost, err)
		}
		statuses = append(statuses, SparkNodeStatus{
			Node: node, Rank: rank, Exists: true, Running: state.Running,
			Status: state.Status, StartedAt: state.StartedAt, ExitCode: state.ExitCode,
		})
	}
	return statuses, nil
}

func StopSparkTarget(target SparkTarget) error {
	statuses, err := InspectSparkTarget(target)
	if err != nil {
		return err
	}
	var failures []string
	for _, status := range statuses {
		if !status.Exists {
			fmt.Fprintf(os.Stderr, "rank %d on %s: absent\n", status.Rank, status.Node.SSHHost)
			continue
		}
		output, exitStatus, runErr := remoteRun(
			status.Node.SSHHost,
			[]string{"docker", "stop", target.ContainerName},
			2*time.Minute,
		)
		if runErr != nil {
			failures = append(failures, fmt.Sprintf("%s: %v", status.Node.SSHHost, runErr))
		} else if exitStatus != 0 {
			failures = append(failures, fmt.Sprintf("%s: %s", status.Node.SSHHost, strings.TrimSpace(string(output))))
		} else {
			fmt.Fprintf(os.Stderr, "rank %d stopped on %s\n", status.Rank, status.Node.SSHHost)
		}
	}
	if len(failures) > 0 {
		return fmt.Errorf("cluster stop failed: %s", strings.Join(failures, "; "))
	}
	return nil
}

func RunSparkLogs(target SparkTarget, follow bool, tail int) (int, error) {
	if tail < 0 {
		return 2, fmt.Errorf("log tail must be non-negative")
	}
	argv := []string{"docker", "logs", "--tail", strconv.Itoa(tail)}
	if follow {
		argv = append(argv, "--follow")
	}
	argv = append(argv, target.ContainerName)
	remote := RemoteArgv(target.Nodes[0].SSHHost, argv)
	command := exec.Command(remote[0], remote[1:]...)
	command.Stdout, command.Stderr = os.Stdout, os.Stderr
	err := command.Run()
	if err == nil {
		return 0, nil
	}
	if exit, ok := err.(*exec.ExitError); ok {
		return exit.ExitCode(), nil
	}
	return 1, err
}

func WaitSparkReady(target SparkTarget, timeout time.Duration) error {
	if timeout <= 0 {
		return fmt.Errorf("readiness timeout must be positive")
	}
	deadline := time.Now().Add(timeout)
	for {
		statuses, err := InspectSparkTarget(target)
		if err != nil {
			return err
		}
		for _, status := range statuses {
			if !status.Exists {
				return fmt.Errorf("rank %d container is absent on %s", status.Rank, status.Node.SSHHost)
			}
			if !status.Running {
				return fmt.Errorf("rank %d container exited on %s with status %s and code %d", status.Rank, status.Node.SSHHost, status.Status, status.ExitCode)
			}
		}
		_, healthStatus, healthErr := remoteRun(
			target.Nodes[0].SSHHost,
			[]string{"curl", "--fail", "--silent", "--show-error", "--max-time", "5", fmt.Sprintf("http://127.0.0.1:%d/health", target.Port)},
			10*time.Second,
		)
		if healthErr == nil && healthStatus == 0 {
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("API did not become healthy on %s:%d within %s", target.Nodes[0].SSHHost, target.Port, timeout)
		}
		time.Sleep(5 * time.Second)
	}
}

func SparkProfileRequest(target SparkTarget, start bool, timeout time.Duration) error {
	endpoint := "stop_profile"
	if start {
		endpoint = "start_profile"
	}
	output, status, err := remoteRun(
		target.Nodes[0].SSHHost,
		[]string{"curl", "--fail", "--silent", "--show-error", "--max-time", strconv.Itoa(max(1, int(timeout.Seconds()))), "--request", "POST", fmt.Sprintf("http://127.0.0.1:%d/%s", target.Port, endpoint)},
		timeout+5*time.Second,
	)
	if err != nil {
		return err
	}
	if status != 0 {
		return fmt.Errorf("%s failed: %s", endpoint, strings.TrimSpace(string(output)))
	}
	if response := strings.TrimSpace(string(output)); response != "" {
		fmt.Fprintln(os.Stdout, response)
	}
	return nil
}

func prepareRemoteDirectory(host, path string) error {
	output, status, err := remoteRun(host, []string{"mkdir", "-p", "--", path}, 30*time.Second)
	if err != nil {
		return err
	}
	if status != 0 {
		return fmt.Errorf("%s", strings.TrimSpace(string(output)))
	}
	return nil
}

func PrepareSparkProfilerDirectories(spec LaunchSpec) error {
	profileDir, ok := ProfilerOutputDir(spec.VLLMArgv)
	if !ok {
		return nil
	}
	for _, launch := range spec.SparkNodes {
		if err := prepareRemoteDirectory(launch.Node.SSHHost, profileDir); err != nil {
			return fmt.Errorf("cannot create profiler directory on %s: %w", launch.Node.SSHHost, err)
		}
	}
	return nil
}

func SyncSparkModel(spec LaunchSpec) error {
	if spec.Topology.Spark == nil {
		return fmt.Errorf("model synchronization requires a Spark/RDMA launch spec")
	}
	if !filepath.IsAbs(spec.ModelSource) {
		return fmt.Errorf("--sync-model requires an explicit --model-path")
	}
	info, err := os.Stat(spec.ModelSource)
	if err != nil || !info.IsDir() {
		return fmt.Errorf("model source is not a directory: %s", spec.ModelSource)
	}
	for _, launch := range spec.SparkNodes {
		if err := prepareRemoteDirectory(launch.Node.SSHHost, spec.ModelSource); err != nil {
			return fmt.Errorf("cannot prepare model directory on %s: %w", launch.Node.SSHHost, err)
		}
		fmt.Fprintf(os.Stderr, "syncing model to %s:%s\n", launch.Node.SSHHost, spec.ModelSource)
		command := exec.Command(
			"rsync", "--archive", "--partial", "--protect-args", "--info=progress2",
			spec.ModelSource+"/", launch.Node.SSHHost+":"+spec.ModelSource+"/",
		)
		command.Stdout, command.Stderr = os.Stdout, os.Stderr
		if err := command.Run(); err != nil {
			return fmt.Errorf("model sync failed for %s: %w", launch.Node.SSHHost, err)
		}
	}
	return nil
}

func stopContainer(node SparkNodeLaunch, containerID string) {
	_, _, _ = remoteRun(node.Node.SSHHost, []string{"docker", "stop", containerID}, 30*time.Second)
}

func startContainer(node SparkNodeLaunch) (string, error) {
	output, status, err := remoteRun(node.Node.SSHHost, node.DockerArgv, 10*time.Minute)
	if err != nil {
		return "", fmt.Errorf("rank %d container failed on %s: %w", node.Rank, node.Node.SSHHost, err)
	}
	if status != 0 {
		return "", fmt.Errorf("rank %d container failed on %s: %s", node.Rank, node.Node.SSHHost, strings.TrimSpace(string(output)))
	}
	containerID := strings.TrimSpace(string(output))
	if containerID == "" {
		return "", fmt.Errorf("rank %d container on %s returned no ID", node.Rank, node.Node.SSHHost)
	}
	shortID := containerID
	if len(shortID) > 12 {
		shortID = shortID[:12]
	}
	fmt.Fprintf(os.Stderr, "rank %d started on %s (%s): %s\n", node.Rank, node.Node.SSHHost, node.Node.Address, shortID)
	return containerID, nil
}

func followHead(node SparkNodeLaunch, containerID string) (int, error) {
	logsArgv := RemoteArgv(node.Node.SSHHost, []string{"docker", "logs", "--follow", containerID})
	logs := exec.Command(logsArgv[0], logsArgv[1:]...)
	logs.Stdout, logs.Stderr = os.Stdout, os.Stderr
	if err := logs.Start(); err != nil {
		return 1, err
	}
	output, status, err := remoteRun(node.Node.SSHHost, []string{"docker", "wait", containerID}, 30*24*time.Hour)
	if logs.Process != nil {
		_ = logs.Process.Kill()
	}
	_ = logs.Wait()
	if err != nil {
		return 1, fmt.Errorf("cannot wait for head container: %w", err)
	}
	if status != 0 {
		return 1, fmt.Errorf("cannot wait for head container: %s", strings.TrimSpace(string(output)))
	}
	exitStatus, err := strconv.Atoi(strings.TrimSpace(string(output)))
	if err != nil {
		return 1, fmt.Errorf("head container returned invalid exit status: %q", string(output))
	}
	return exitStatus, nil
}

func SyncSparkSources(spec LaunchSpec) bool {
	if spec.Topology.Spark == nil {
		panic("source synchronization requires a Spark/RDMA launch spec")
	}
	topology := spec.Topology.Spark
	type sourceTarget struct{ source, target string }
	sources := []sourceTarget{
		{filepath.Join(topology.RepoRoot, "vllm"), filepath.Join(topology.RuntimeRepoRoot, "vllm")},
		{filepath.Join(topology.B12XRoot, "b12x"), filepath.Join(topology.RuntimeB12XRoot, "b12x")},
	}
	for _, launch := range spec.SparkNodes {
		for _, pair := range sources {
			if err := prepareRemoteDirectory(launch.Node.SSHHost, pair.target); err != nil {
				fmt.Fprintf(os.Stderr, "cannot prepare source directory on %s: %v\n", launch.Node.SSHHost, err)
				return false
			}
			name := pair.source[strings.LastIndex(pair.source, "/")+1:]
			fmt.Fprintf(os.Stderr, "syncing %s/ to %s:%s\n", name, launch.Node.SSHHost, pair.target)
			command := exec.Command(
				"rsync", "--archive", "--delete", "--exclude=__pycache__/",
				"--exclude=*.a", "--exclude=*.o", "--exclude=*.py[co]",
				"--exclude=*.so", pair.source+"/", launch.Node.SSHHost+":"+pair.target+"/",
			)
			command.Stdout, command.Stderr = os.Stdout, os.Stderr
			if err := command.Run(); err != nil {
				fmt.Fprintf(os.Stderr, "source sync failed for %s:%s\n", launch.Node.SSHHost, pair.target)
				return false
			}
		}
	}
	return true
}

func RunSparkCluster(spec LaunchSpec) int {
	if spec.Topology.Spark == nil || len(spec.SparkNodes) == 0 {
		panic("Spark execution requires a Spark/RDMA launch spec")
	}
	head := spec.SparkNodes[0]
	launchOrder := make([]SparkNodeLaunch, 0, len(spec.SparkNodes))
	for index := len(spec.SparkNodes) - 1; index >= 1; index-- {
		launchOrder = append(launchOrder, spec.SparkNodes[index])
	}
	launchOrder = append(launchOrder, head)
	type startedNode struct {
		node SparkNodeLaunch
		id   string
	}
	var started []startedNode
	keepRunning := false
	defer func() {
		if !keepRunning {
			for index := len(started) - 1; index >= 0; index-- {
				stopContainer(started[index].node, started[index].id)
			}
		}
	}()
	for _, node := range launchOrder {
		containerID, err := startContainer(node)
		if err != nil {
			fmt.Fprintln(os.Stderr, "Spark/RDMA launch failed:", err)
			return 1
		}
		started = append(started, startedNode{node, containerID})
	}
	if spec.Detach {
		keepRunning = true
		fmt.Fprintf(os.Stderr, "head API: http://%s:%d\n", head.Node.SSHHost, spec.Port)
		return 0
	}
	status, err := followHead(head, started[len(started)-1].id)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Spark/RDMA launch failed:", err)
		return 1
	}
	return status
}
