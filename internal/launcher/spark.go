// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package launcher

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Tunables for the cluster monitor. They are variables so tests can shrink
// them.
var (
	sparkPollInterval    = 5 * time.Second
	sparkDisconnectGrace = 10 * time.Minute
	sparkLogDrainTimeout = 10 * time.Second
	sparkStartupSettle   = 3 * time.Second
)

func ResolveSparkTarget(profile ModelProfile, topology Topology, requestedTP *int, requestedPort *int) (SparkTarget, error) {
	if topology.Spark == nil {
		return SparkTarget{}, fmt.Errorf("cluster operations require a Spark/RDMA topology")
	}
	if profile.Facts == nil {
		return SparkTarget{}, fmt.Errorf("model %q has no checkpoint facts; config.json was not resolved", profile.Name)
	}
	tpSize, _, err := resolveTPSize(profile.Facts, topology, requestedTP, topology.MemoryUtilization())
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

// inspectContainer reports whether a container exists and, if so, its state.
// A transport failure is returned as an error so callers can tell "absent"
// from "unreachable".
func inspectContainer(ctx context.Context, host, containerName string) (dockerContainerState, bool, error) {
	output, status, err := remoteRun(
		ctx, host,
		[]string{"docker", "container", "inspect", "--format", "{{json .State}}", containerName},
		15*time.Second,
	)
	if err != nil {
		return dockerContainerState{}, false, err
	}
	if status != 0 {
		detail := strings.TrimSpace(string(output))
		if strings.Contains(detail, "No such object") || strings.Contains(detail, "No such container") {
			return dockerContainerState{}, false, nil
		}
		return dockerContainerState{}, false, fmt.Errorf("%s", detail)
	}
	var state dockerContainerState
	if err := json.Unmarshal([]byte(lastOutputLine(output, "{}")), &state); err != nil {
		return dockerContainerState{}, false, fmt.Errorf("invalid Docker state: %w", err)
	}
	return state, true, nil
}

func InspectSparkTarget(ctx context.Context, target SparkTarget) ([]SparkNodeStatus, error) {
	statuses := make([]SparkNodeStatus, 0, len(target.Nodes))
	for rank, node := range target.Nodes {
		state, exists, err := inspectContainer(ctx, node.SSHHost, target.ContainerName)
		if err != nil {
			return nil, fmt.Errorf("rank %d on %s: %w", rank, node.SSHHost, err)
		}
		if !exists {
			statuses = append(statuses, SparkNodeStatus{Node: node, Rank: rank, Status: "absent"})
			continue
		}
		statuses = append(statuses, SparkNodeStatus{
			Node: node, Rank: rank, Exists: true, Running: state.Running,
			Status: state.Status, StartedAt: state.StartedAt, ExitCode: state.ExitCode,
		})
	}
	return statuses, nil
}

func removeContainer(ctx context.Context, host, containerName string) error {
	output, status, err := remoteRun(ctx, host, []string{"docker", "rm", containerName}, 60*time.Second)
	if err != nil {
		return err
	}
	if status != 0 {
		detail := strings.TrimSpace(string(output))
		if strings.Contains(detail, "No such container") {
			return nil
		}
		return fmt.Errorf("%s", detail)
	}
	return nil
}

func stopContainer(ctx context.Context, host, containerName string) error {
	output, status, err := remoteRun(ctx, host, []string{"docker", "stop", containerName}, 2*time.Minute)
	if err != nil {
		return err
	}
	if status != 0 {
		detail := strings.TrimSpace(string(output))
		if strings.Contains(detail, "No such container") {
			return nil
		}
		return fmt.Errorf("%s", detail)
	}
	return nil
}

// StopSparkTarget stops every rank and removes its container. Absent ranks
// are reported, not treated as failures.
func StopSparkTarget(ctx context.Context, target SparkTarget) error {
	statuses, err := InspectSparkTarget(ctx, target)
	if err != nil {
		return err
	}
	var failures []string
	for _, status := range statuses {
		if !status.Exists {
			fmt.Fprintf(os.Stderr, "rank %d on %s: absent\n", status.Rank, status.Node.SSHHost)
			continue
		}
		if status.Running {
			if err := stopContainer(ctx, status.Node.SSHHost, target.ContainerName); err != nil {
				failures = append(failures, fmt.Sprintf("%s: %v", status.Node.SSHHost, err))
				continue
			}
		}
		if err := removeContainer(ctx, status.Node.SSHHost, target.ContainerName); err != nil {
			failures = append(failures, fmt.Sprintf("%s: %v", status.Node.SSHHost, err))
			continue
		}
		fmt.Fprintf(os.Stderr, "rank %d stopped and removed on %s\n", status.Rank, status.Node.SSHHost)
	}
	if len(failures) > 0 {
		return fmt.Errorf("cluster stop failed: %s", strings.Join(failures, "; "))
	}
	return nil
}

func RunSparkLogs(ctx context.Context, target SparkTarget, rank int, follow bool, tail int) (int, error) {
	if tail < 0 {
		return 2, fmt.Errorf("log tail must be non-negative")
	}
	if rank < 0 || rank >= len(target.Nodes) {
		return 2, fmt.Errorf("rank must be in 0..%d", len(target.Nodes)-1)
	}
	argv := []string{"docker", "logs", "--tail", strconv.Itoa(tail)}
	if follow {
		argv = append(argv, "--follow")
	}
	argv = append(argv, target.ContainerName)
	wait, err := startStream(ctx, RemoteArgv(target.Nodes[rank].SSHHost, argv))
	if err != nil {
		return 1, err
	}
	err = wait()
	if err == nil {
		return 0, nil
	}
	if exit, ok := err.(*exec.ExitError); ok {
		return exit.ExitCode(), nil
	}
	return 1, err
}

func WaitSparkReady(ctx context.Context, target SparkTarget, timeout time.Duration) error {
	if timeout <= 0 {
		return fmt.Errorf("readiness timeout must be positive")
	}
	deadline := time.Now().Add(timeout)
	for {
		statuses, err := InspectSparkTarget(ctx, target)
		if err != nil {
			return err
		}
		for _, status := range statuses {
			if !status.Exists {
				return fmt.Errorf("rank %d container is absent on %s; it was never started or has been removed", status.Rank, status.Node.SSHHost)
			}
			if !status.Running {
				return fmt.Errorf("rank %d container exited on %s with status %s and code %d; inspect it with lil cluster logs --rank %d", status.Rank, status.Node.SSHHost, status.Status, status.ExitCode, status.Rank)
			}
		}
		_, healthStatus, healthErr := remoteRun(
			ctx, target.Nodes[0].SSHHost,
			[]string{"curl", "--fail", "--silent", "--show-error", "--max-time", "5", fmt.Sprintf("http://127.0.0.1:%d/health", target.Port)},
			10*time.Second,
		)
		if healthErr == nil && healthStatus == 0 {
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("API did not become healthy on %s:%d within %s", target.Nodes[0].SSHHost, target.Port, timeout)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(sparkPollInterval):
		}
	}
}

func SparkProfileRequest(ctx context.Context, target SparkTarget, start bool, timeout time.Duration) error {
	endpoint := "stop_profile"
	if start {
		endpoint = "start_profile"
	}
	output, status, err := remoteRun(
		ctx, target.Nodes[0].SSHHost,
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

func prepareRemoteDirectory(ctx context.Context, host, path string) error {
	output, status, err := remoteRun(ctx, host, []string{"mkdir", "-p", "--", path}, 30*time.Second)
	if err != nil {
		return err
	}
	if status != 0 {
		return fmt.Errorf("%s", strings.TrimSpace(string(output)))
	}
	return nil
}

func PrepareSparkProfilerDirectories(ctx context.Context, spec LaunchSpec) error {
	profileDir, ok := ProfilerOutputDir(spec.VLLMArgv)
	if !ok {
		return nil
	}
	for _, launch := range spec.SparkNodes {
		if err := prepareRemoteDirectory(ctx, launch.Node.SSHHost, profileDir); err != nil {
			return fmt.Errorf("cannot create profiler directory on %s: %w", launch.Node.SSHHost, err)
		}
	}
	return nil
}

func SyncSparkModel(ctx context.Context, spec LaunchSpec) error {
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
		if err := prepareRemoteDirectory(ctx, launch.Node.SSHHost, spec.ModelSource); err != nil {
			return fmt.Errorf("cannot prepare model directory on %s: %w", launch.Node.SSHHost, err)
		}
		fmt.Fprintf(os.Stderr, "syncing model to %s:%s\n", launch.Node.SSHHost, spec.ModelSource)
		err := runVisible(ctx, []string{
			"rsync", "--archive", "--partial", "--protect-args", "--info=progress2",
			spec.ModelSource + "/", launch.Node.SSHHost + ":" + spec.ModelSource + "/",
		}, nil)
		if err != nil {
			return fmt.Errorf("model sync failed for %s: %w", launch.Node.SSHHost, err)
		}
	}
	return nil
}

func SyncSparkSources(ctx context.Context, spec LaunchSpec) bool {
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
			if err := prepareRemoteDirectory(ctx, launch.Node.SSHHost, pair.target); err != nil {
				fmt.Fprintf(os.Stderr, "cannot prepare source directory on %s: %v\n", launch.Node.SSHHost, err)
				return false
			}
			name := pair.source[strings.LastIndex(pair.source, "/")+1:]
			fmt.Fprintf(os.Stderr, "syncing %s/ to %s:%s\n", name, launch.Node.SSHHost, pair.target)
			err := runVisible(ctx, []string{
				"rsync", "--archive", "--delete", "--exclude=__pycache__/",
				"--exclude=*.a", "--exclude=*.o", "--exclude=*.py[co]",
				"--exclude=*.so", pair.source + "/", launch.Node.SSHHost + ":" + pair.target + "/",
			}, nil)
			if err != nil {
				fmt.Fprintf(os.Stderr, "source sync failed for %s:%s\n", launch.Node.SSHHost, pair.target)
				return false
			}
		}
	}
	return true
}

// removeStaleContainer clears an exited container that still holds the
// rank's name. A running container is a hard failure: preflight rejects it,
// and racing another launcher for the name must not stop its cluster.
func removeStaleContainer(ctx context.Context, node SparkNodeLaunch) error {
	state, exists, err := inspectContainer(ctx, node.Node.SSHHost, node.ContainerName)
	if err != nil {
		return fmt.Errorf("rank %d on %s: %w", node.Rank, node.Node.SSHHost, err)
	}
	if !exists {
		return nil
	}
	if state.Running {
		return fmt.Errorf("rank %d container %s is already running on %s", node.Rank, node.ContainerName, node.Node.SSHHost)
	}
	fmt.Fprintf(os.Stderr, "rank %d: removing exited container %s on %s (exit code %d)\n", node.Rank, node.ContainerName, node.Node.SSHHost, state.ExitCode)
	if err := removeContainer(ctx, node.Node.SSHHost, node.ContainerName); err != nil {
		return fmt.Errorf("rank %d on %s: %w", node.Rank, node.Node.SSHHost, err)
	}
	return nil
}

func startContainer(ctx context.Context, node SparkNodeLaunch) error {
	output, status, err := remoteRun(ctx, node.Node.SSHHost, node.DockerArgv, 10*time.Minute)
	if err != nil {
		return fmt.Errorf("rank %d container failed on %s: %w", node.Rank, node.Node.SSHHost, err)
	}
	if status != 0 {
		return fmt.Errorf("rank %d container failed on %s: %s", node.Rank, node.Node.SSHHost, strings.TrimSpace(string(output)))
	}
	containerID := strings.TrimSpace(lastOutputLine(output, ""))
	if containerID == "" {
		return fmt.Errorf("rank %d container on %s returned no ID", node.Rank, node.Node.SSHHost)
	}
	if len(containerID) > 12 {
		containerID = containerID[:12]
	}
	fmt.Fprintf(os.Stderr, "rank %d started on %s (%s): %s\n", node.Rank, node.Node.SSHHost, node.Node.Address, containerID)
	return nil
}

// printRankTail shows the last lines of an exited rank so the failure is
// visible without a second command.
func printRankTail(ctx context.Context, node SparkNodeLaunch, lines int) {
	output, _, err := remoteRun(ctx, node.Node.SSHHost, []string{"docker", "logs", "--tail", strconv.Itoa(lines), node.ContainerName}, 30*time.Second)
	if err != nil {
		fmt.Fprintf(os.Stderr, "rank %d: cannot read logs from %s: %v\n", node.Rank, node.Node.SSHHost, err)
		return
	}
	text := strings.TrimSpace(string(output))
	if text == "" {
		return
	}
	fmt.Fprintf(os.Stderr, "---- rank %d (%s) last %d log lines ----\n%s\n---- end rank %d ----\n", node.Rank, node.Node.SSHHost, lines, text, node.Rank)
}

// teardown stops and removes every started rank using a fresh context, so an
// interrupted launch still cleans up.
func teardown(started []SparkNodeLaunch, keepContainers bool) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	for index := len(started) - 1; index >= 0; index-- {
		node := started[index]
		if err := stopContainer(ctx, node.Node.SSHHost, node.ContainerName); err != nil {
			fmt.Fprintf(os.Stderr, "rank %d: stop failed on %s: %v\n", node.Rank, node.Node.SSHHost, err)
			continue
		}
		if keepContainers {
			continue
		}
		if err := removeContainer(ctx, node.Node.SSHHost, node.ContainerName); err != nil {
			fmt.Fprintf(os.Stderr, "rank %d: remove failed on %s: %v\n", node.Rank, node.Node.SSHHost, err)
		}
	}
}

type rankObservation struct {
	node   SparkNodeLaunch
	state  dockerContainerState
	exists bool
}

// observeRanks inspects every rank; an error means at least one rank could
// not be reached, not that a container exited.
func observeRanks(ctx context.Context, nodes []SparkNodeLaunch) ([]rankObservation, error) {
	observations := make([]rankObservation, 0, len(nodes))
	for _, node := range nodes {
		state, exists, err := inspectContainer(ctx, node.Node.SSHHost, node.ContainerName)
		if err != nil {
			return nil, fmt.Errorf("rank %d on %s: %w", node.Rank, node.Node.SSHHost, err)
		}
		observations = append(observations, rankObservation{node, state, exists})
	}
	return observations, nil
}

func exitedRank(observations []rankObservation) (rankObservation, bool) {
	for _, observation := range observations {
		if !observation.exists || !observation.state.Running {
			return observation, true
		}
	}
	return rankObservation{}, false
}

// logFollower streams the head container's logs and reconnects after a
// dropped session, resuming from the moment the previous session started.
type logFollower struct {
	head   SparkNodeLaunch
	cancel context.CancelFunc
	done   chan struct{}
}

func followHeadLogs(parent context.Context, head SparkNodeLaunch) *logFollower {
	ctx, cancel := context.WithCancel(parent)
	follower := &logFollower{head: head, cancel: cancel, done: make(chan struct{})}
	go func() {
		defer close(follower.done)
		since := ""
		for ctx.Err() == nil {
			argv := []string{"docker", "logs", "--follow"}
			if since != "" {
				argv = append(argv, "--since", since)
			}
			argv = append(argv, head.ContainerName)
			since = time.Now().UTC().Format(time.RFC3339Nano)
			wait, err := startStream(ctx, RemoteArgv(head.Node.SSHHost, argv))
			if err == nil {
				err = wait()
			}
			if ctx.Err() != nil {
				return
			}
			if err != nil {
				fmt.Fprintf(os.Stderr, "log stream from %s ended: %v; reconnecting\n", head.Node.SSHHost, err)
			}
			select {
			case <-ctx.Done():
				return
			case <-time.After(sparkPollInterval):
			}
		}
	}()
	return follower
}

// drain gives the stream a moment to flush the container's final lines, then
// stops it.
func (f *logFollower) drain() {
	select {
	case <-f.done:
	case <-time.After(sparkLogDrainTimeout):
	}
	f.cancel()
	<-f.done
}

// monitorCluster watches every rank until one exits, the context is
// cancelled, or the ranks stay unreachable past the disconnect grace. Only an
// observed exit or a cancellation returns with teardown requested.
func monitorCluster(ctx context.Context, nodes []SparkNodeLaunch, head SparkNodeLaunch) (status int, tearDown bool) {
	unreachableSince := time.Time{}
	unreachablePolls := 0
	for {
		observations, err := observeRanks(ctx, nodes)
		switch {
		case ctx.Err() != nil:
			fmt.Fprintln(os.Stderr, "interrupted; stopping the cluster")
			return 130, true
		case err != nil:
			if unreachableSince.IsZero() {
				unreachableSince = time.Now()
			}
			if unreachablePolls%12 == 0 {
				fmt.Fprintf(os.Stderr, "cannot observe the cluster (%v); containers are left running\n", err)
			}
			unreachablePolls++
			if time.Since(unreachableSince) > sparkDisconnectGrace {
				fmt.Fprintf(os.Stderr, "no contact with the cluster for %s; giving up. Use lil cluster status, logs, and stop to manage it\n", sparkDisconnectGrace)
				return 1, false
			}
		default:
			unreachableSince, unreachablePolls = time.Time{}, 0
			if observation, exited := exitedRank(observations); exited {
				if !observation.exists {
					fmt.Fprintf(os.Stderr, "rank %d container disappeared from %s\n", observation.node.Rank, observation.node.Node.SSHHost)
					return 1, true
				}
				if observation.node.Rank == head.Rank {
					return observation.state.ExitCode, true
				}
				fmt.Fprintf(os.Stderr, "rank %d exited on %s with code %d; stopping the cluster\n", observation.node.Rank, observation.node.Node.SSHHost, observation.state.ExitCode)
				printRankTail(context.Background(), observation.node, 40)
				return 1, true
			}
		}
		select {
		case <-ctx.Done():
			fmt.Fprintln(os.Stderr, "interrupted; stopping the cluster")
			return 130, true
		case <-time.After(sparkPollInterval):
		}
	}
}

// RunSparkCluster starts the workers, then the head, and supervises them.
// Containers are named and kept after exit, so a crashed rank leaves its
// logs and exit code behind. Without --detach the controller tears the
// cluster down only when it observes a rank exit or is interrupted; losing
// contact with the ranks never stops a running cluster.
func RunSparkCluster(ctx context.Context, spec LaunchSpec) int {
	if spec.Topology.Spark == nil || len(spec.SparkNodes) == 0 {
		panic("Spark execution requires a Spark/RDMA launch spec")
	}
	head := spec.SparkNodes[0]
	launchOrder := make([]SparkNodeLaunch, 0, len(spec.SparkNodes))
	for index := len(spec.SparkNodes) - 1; index >= 1; index-- {
		launchOrder = append(launchOrder, spec.SparkNodes[index])
	}
	launchOrder = append(launchOrder, head)
	for _, node := range launchOrder {
		if err := removeStaleContainer(ctx, node); err != nil {
			fmt.Fprintln(os.Stderr, "Spark/RDMA launch failed:", err)
			return 1
		}
	}
	var started []SparkNodeLaunch
	for _, node := range launchOrder {
		if ctx.Err() != nil {
			fmt.Fprintln(os.Stderr, "interrupted before every rank started; stopping the cluster")
			teardown(started, false)
			return 130
		}
		if err := startContainer(ctx, node); err != nil {
			fmt.Fprintln(os.Stderr, "Spark/RDMA launch failed:", err)
			teardown(started, false)
			return 1
		}
		started = append(started, node)
	}
	select {
	case <-ctx.Done():
		fmt.Fprintln(os.Stderr, "interrupted; stopping the cluster")
		teardown(started, false)
		return 130
	case <-time.After(sparkStartupSettle):
	}
	if observations, err := observeRanks(ctx, spec.SparkNodes); err == nil {
		if observation, exited := exitedRank(observations); exited {
			fmt.Fprintf(os.Stderr, "rank %d exited on %s with code %d during startup\n", observation.node.Rank, observation.node.Node.SSHHost, observation.state.ExitCode)
			printRankTail(context.Background(), observation.node, 40)
			teardown(started, true)
			fmt.Fprintln(os.Stderr, "containers were stopped but kept for inspection; lil cluster stop removes them")
			return 1
		}
	}
	if spec.Detach {
		fmt.Fprintf(os.Stderr, "head API: http://%s:%d\n", head.Node.SSHHost, spec.Port)
		return 0
	}
	follower := followHeadLogs(ctx, head)
	status, tearDown := monitorCluster(ctx, spec.SparkNodes, head)
	follower.drain()
	if tearDown {
		if status != 0 && status != 130 {
			printRankTail(context.Background(), head, 40)
		}
		teardown(started, false)
	}
	return status
}
