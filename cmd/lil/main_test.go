// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package main

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

func configureTestTopologies(t *testing.T) string {
	t.Helper()
	directory := t.TempDir()
	t.Setenv("LIL_TOPOLOGY_DIR", directory)
	for _, fixture := range []string{"local.yaml", "spark.yaml"} {
		data, err := os.ReadFile(filepath.Join("..", "..", "internal", "launcher", "testdata", fixture))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(directory, fixture), data, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return directory
}

func testModelsConfig() string {
	return filepath.Join(
		"..", "..", "internal", "launcher", "testdata", "model-manifests",
	)
}

func TestListUsesReadableMultilineCards(t *testing.T) {
	configureTestTopologies(t)
	profiles, err := loadProfiles(testModelsConfig())
	if err != nil {
		t.Fatal(err)
	}
	output := ansi.Strip(renderList(profiles))
	for _, want := range []string{
		"Models  3\n\n",
		"GLM-5.3-NVFP4",
		"GLM-5.3 with NVFP4 routed experts",
		"defaults  ·  local TP 8  ·  Spark/RDMA TP all",
		"Topologies  2\n\n",
		"local-test",
		"local  ·  TP 1–12  ·  95.6 GiB/GPU  ·  sm_120a",
		"spark-test",
		"Spark/RDMA  ·  TP 1–2  ·  108.0 GiB/rank  ·  sm_121a",
	} {
		if !strings.Contains(output, want) {
			t.Errorf("list output is missing %q:\n%s", want, output)
		}
	}
	for _, line := range strings.Split(strings.TrimSpace(output), "\n") {
		if width := lipgloss.Width(line); width > 80 {
			t.Errorf("list line is %d columns wide: %q", width, line)
		}
	}
}

func TestForwardedArgumentsRequireSeparatorAndRemainLast(t *testing.T) {
	configureTestTopologies(t)
	fs := newFlagSet("render")
	var values launchFlags
	configureLaunchFlags(fs, &values, false)
	values.modelsConfig = testModelsConfig()
	if err := fs.Parse([]string{
		"Qwen3.8-Flash-Next-NVFP4",
		"--speculator", "none",
		"--",
		"--disable-log-requests",
	}); err != nil {
		t.Fatal(err)
	}
	spec, err := buildSpec(fs, values)
	if err != nil {
		t.Fatal(err)
	}
	if got := spec.VLLMArgv[len(spec.VLLMArgv)-1]; got != "--disable-log-requests" {
		t.Fatalf("last vLLM argument: got %q", got)
	}
	if !slices.Contains(spec.VLLMArgv, "--max-model-len") {
		t.Fatalf("resolved argv is incomplete: %v", spec.VLLMArgv)
	}
}

func TestManagedForwardedArgumentFailsClosed(t *testing.T) {
	configureTestTopologies(t)
	fs := newFlagSet("render")
	var values launchFlags
	configureLaunchFlags(fs, &values, false)
	values.modelsConfig = testModelsConfig()
	if err := fs.Parse([]string{
		"Qwen3.8-Flash-Next-NVFP4",
		"--speculator", "none",
		"--",
		"--port", "9000",
	}); err != nil {
		t.Fatal(err)
	}
	_, err := buildSpec(fs, values)
	if err == nil || !strings.Contains(err.Error(), "launcher-managed") {
		t.Fatalf("managed argument error: %v", err)
	}
}

func TestConfigurationErrorsReturnUsageStatus(t *testing.T) {
	configureTestTopologies(t)
	status, err := execute([]string{
		"render",
		"Qwen3.8-Flash-Next-NVFP4",
		"--models-config", testModelsConfig(),
		"--model-path", "/does/not/exist",
	})
	if status != 2 || err == nil {
		t.Fatalf("status=%d error=%v", status, err)
	}

	status, err = execute([]string{"list", "unexpected"})
	if status != 2 || err == nil {
		t.Fatalf("list status=%d error=%v", status, err)
	}
}

func TestClusterHelpReturnsSuccess(t *testing.T) {
	for _, args := range [][]string{
		{"cluster", "--help"},
		{"cluster", "status", "--help"},
	} {
		status, err := execute(args)
		if status != 0 || err != nil {
			t.Errorf("%v: status=%d error=%v", args, status, err)
		}
	}
}
