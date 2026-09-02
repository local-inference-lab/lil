// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package launcher

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"
)

// fakeCluster scripts docker command outcomes per host so the Spark
// lifecycle can be exercised without SSH.
type fakeCluster struct {
	mu          sync.Mutex
	containers  map[string]*fakeContainer
	calls       []string
	unreachable bool
	// exitAfter maps "host" to the number of inspections after which the
	// container reports an exit with exitCode.
	exitAfter map[string]int
	exitCode  map[string]int
	inspects  map[string]int
}

type fakeContainer struct {
	running  bool
	exitCode int
}

func newFakeCluster() *fakeCluster {
	return &fakeCluster{
		containers: map[string]*fakeContainer{}, exitAfter: map[string]int{},
		exitCode: map[string]int{}, inspects: map[string]int{},
	}
}

func (f *fakeCluster) run(ctx context.Context, timeout time.Duration, directory string, environment []string, argv []string) ([]byte, int, error) {
	if len(argv) == 0 || argv[0] != "ssh" {
		return nil, -1, fmt.Errorf("unexpected local command %v", argv)
	}
	host, command := argv[len(argv)-2], argv[len(argv)-1]
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, host+": "+command)
	if f.unreachable {
		return nil, -1, errors.New("ssh: connect to host failed")
	}
	switch {
	case strings.HasPrefix(command, "docker container inspect"):
		container, ok := f.containers[host]
		if !ok {
			return []byte("Error: No such container"), 1, nil
		}
		f.inspects[host]++
		if limit, scheduled := f.exitAfter[host]; scheduled && f.inspects[host] > limit && container.running {
			container.running = false
			container.exitCode = f.exitCode[host]
		}
		return []byte(fmt.Sprintf(`{"Running":%t,"Status":"%s","ExitCode":%d}`, container.running, map[bool]string{true: "running", false: "exited"}[container.running], container.exitCode)), 0, nil
	case strings.HasPrefix(command, "docker run"):
		if container, ok := f.containers[host]; ok && container.running {
			return []byte("Error: name is already in use"), 125, nil
		}
		f.containers[host] = &fakeContainer{running: true}
		return []byte("0123456789abcdef\n"), 0, nil
	case strings.HasPrefix(command, "docker stop"):
		if container, ok := f.containers[host]; ok {
			container.running = false
			return []byte("stopped"), 0, nil
		}
		return []byte("Error: No such container"), 1, nil
	case strings.HasPrefix(command, "docker rm"):
		if _, ok := f.containers[host]; ok {
			delete(f.containers, host)
			return []byte("removed"), 0, nil
		}
		return []byte("Error: No such container"), 1, nil
	case strings.HasPrefix(command, "docker logs"):
		return []byte("log line\n"), 0, nil
	}
	return nil, -1, fmt.Errorf("unscripted command %q on %s", command, host)
}

func (f *fakeCluster) count(prefix string) int {
	f.mu.Lock()
	defer f.mu.Unlock()
	total := 0
	for _, call := range f.calls {
		if strings.Contains(call, ": "+prefix) {
			total++
		}
	}
	return total
}

func installFakeCluster(t *testing.T) *fakeCluster {
	t.Helper()
	fake := newFakeCluster()
	previousRun, previousStream := runCommand, startStream
	previousPoll, previousGrace, previousDrain, previousSettle := sparkPollInterval, sparkDisconnectGrace, sparkLogDrainTimeout, sparkStartupSettle
	runCommand = fake.run
	startStream = func(ctx context.Context, argv []string) (func() error, error) {
		return func() error { <-ctx.Done(); return nil }, nil
	}
	sparkPollInterval, sparkDisconnectGrace, sparkLogDrainTimeout, sparkStartupSettle = 5*time.Millisecond, 60*time.Millisecond, 5*time.Millisecond, time.Millisecond
	t.Cleanup(func() {
		runCommand, startStream = previousRun, previousStream
		sparkPollInterval, sparkDisconnectGrace, sparkLogDrainTimeout, sparkStartupSettle = previousPoll, previousGrace, previousDrain, previousSettle
	})
	return fake
}

func sparkLaunchSpec(t *testing.T, detach bool) LaunchSpec {
	t.Helper()
	config := loadTestConfig(t)
	options := defaultOptions(2)
	options.Detach = detach
	spec, err := BuildLaunchSpec(config.profiles[qwenProfile], config.spark, options)
	if err != nil {
		t.Fatal(err)
	}
	return spec
}

func TestSparkRunStartsWorkersFirstAndTearsDownOnHeadExit(t *testing.T) {
	fake := installFakeCluster(t)
	spec := sparkLaunchSpec(t, false)
	fake.exitAfter["tachyon"] = 3
	fake.exitCode["tachyon"] = 0
	status := RunSparkCluster(context.Background(), spec)
	if status != 0 {
		t.Fatalf("status %d", status)
	}
	var starts []string
	for _, call := range fake.calls {
		if strings.Contains(call, ": docker run") {
			starts = append(starts, strings.SplitN(call, ":", 2)[0])
		}
	}
	if strings.Join(starts, ",") != "luxon,tachyon" {
		t.Fatalf("start order: %v", starts)
	}
	if fake.count("docker stop") != 2 || fake.count("docker rm") != 2 || len(fake.containers) != 0 {
		t.Fatalf("cluster was not torn down: stops=%d removes=%d remaining=%v", fake.count("docker stop"), fake.count("docker rm"), fake.containers)
	}
}

func TestSparkRunReportsWorkerExitAndStopsTheHead(t *testing.T) {
	fake := installFakeCluster(t)
	spec := sparkLaunchSpec(t, false)
	fake.exitAfter["luxon"] = 4
	fake.exitCode["luxon"] = 137
	status := RunSparkCluster(context.Background(), spec)
	if status != 1 {
		t.Fatalf("status %d", status)
	}
	if fake.count("docker logs --tail 40") < 1 {
		t.Fatalf("worker logs were not captured: %v", fake.calls)
	}
	if len(fake.containers) != 0 {
		t.Fatalf("cluster was not torn down: %v", fake.containers)
	}
}

func TestSparkRunNeverTearsDownWhileUnreachable(t *testing.T) {
	fake := installFakeCluster(t)
	spec := sparkLaunchSpec(t, false)
	go func() {
		time.Sleep(15 * time.Millisecond)
		fake.mu.Lock()
		fake.unreachable = true
		fake.mu.Unlock()
	}()
	status := RunSparkCluster(context.Background(), spec)
	if status != 1 {
		t.Fatalf("status %d", status)
	}
	if fake.count("docker stop") != 0 || fake.count("docker rm") != 0 {
		t.Fatalf("an unreachable cluster was torn down: %v", fake.calls)
	}
	fake.mu.Lock()
	defer fake.mu.Unlock()
	if len(fake.containers) != 2 || !fake.containers["tachyon"].running || !fake.containers["luxon"].running {
		t.Fatalf("containers should be left running: %v", fake.containers)
	}
}

func TestSparkRunInterruptStopsAndRemovesEveryRank(t *testing.T) {
	fake := installFakeCluster(t)
	spec := sparkLaunchSpec(t, false)
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(20 * time.Millisecond)
		cancel()
	}()
	status := RunSparkCluster(ctx, spec)
	if status != 130 {
		t.Fatalf("status %d", status)
	}
	if len(fake.containers) != 0 || fake.count("docker rm") != 2 {
		t.Fatalf("interrupt did not clean up: %v %v", fake.containers, fake.calls)
	}
}

func TestSparkRunRefusesRunningContainerAndRemovesExitedOne(t *testing.T) {
	fake := installFakeCluster(t)
	spec := sparkLaunchSpec(t, true)
	fake.containers["luxon"] = &fakeContainer{running: true}
	if status := RunSparkCluster(context.Background(), spec); status != 1 || fake.count("docker run") != 0 {
		t.Fatalf("running container was not refused: status=%d calls=%v", status, fake.calls)
	}
	fake.containers["luxon"] = &fakeContainer{running: false, exitCode: 3}
	if status := RunSparkCluster(context.Background(), spec); status != 0 {
		t.Fatalf("detached launch failed: %d", status)
	}
	if fake.count("docker rm") != 1 || fake.count("docker run") != 2 || fake.count("docker stop") != 0 {
		t.Fatalf("stale container handling: %v", fake.calls)
	}
	if len(fake.containers) != 2 || !fake.containers["luxon"].running || !fake.containers["tachyon"].running {
		t.Fatalf("detached cluster is not running: %v", fake.containers)
	}
}

func TestSparkDetachedLaunchReportsImmediateCrash(t *testing.T) {
	fake := installFakeCluster(t)
	spec := sparkLaunchSpec(t, true)
	fake.exitAfter["luxon"] = 0
	fake.exitCode["luxon"] = 2
	if status := RunSparkCluster(context.Background(), spec); status != 1 {
		t.Fatalf("status %d", status)
	}
	if fake.count("docker stop") != 2 || fake.count("docker rm") != 0 {
		t.Fatalf("crashed detached launch must stop but keep containers: %v", fake.calls)
	}
}

func TestClusterStopRemovesContainersAndSkipsAbsentRanks(t *testing.T) {
	fake := installFakeCluster(t)
	config := loadTestConfig(t)
	target, err := ResolveSparkTarget(config.profiles[qwenProfile], config.spark, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	fake.containers["tachyon"] = &fakeContainer{running: true}
	if err := StopSparkTarget(context.Background(), target); err != nil {
		t.Fatal(err)
	}
	if len(fake.containers) != 0 || fake.count("docker stop") != 1 || fake.count("docker rm") != 1 {
		t.Fatalf("stop did not remove the running rank only: %v", fake.calls)
	}
	statuses, err := InspectSparkTarget(context.Background(), target)
	if err != nil || len(statuses) != 2 || statuses[0].Exists || statuses[1].Exists {
		t.Fatalf("statuses after stop: %+v %v", statuses, err)
	}
}

func TestWaitSparkReadyReportsExitedRank(t *testing.T) {
	fake := installFakeCluster(t)
	config := loadTestConfig(t)
	target, err := ResolveSparkTarget(config.profiles[qwenProfile], config.spark, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	fake.containers["tachyon"] = &fakeContainer{running: true}
	fake.containers["luxon"] = &fakeContainer{running: false, exitCode: 9}
	err = WaitSparkReady(context.Background(), target, time.Second)
	if err == nil || !strings.Contains(err.Error(), "rank 1 container exited") || !strings.Contains(err.Error(), "--rank 1") {
		t.Fatalf("wait error: %v", err)
	}
}
