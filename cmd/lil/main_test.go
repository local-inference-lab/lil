// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package main

import (
	"context"
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

func TestListShowsFactsAndPerTopologyDefaults(t *testing.T) {
	configureTestTopologies(t)
	loaded, err := loadCatalog(context.Background(), testModelsConfig())
	if err != nil {
		t.Fatal(err)
	}
	if len(loaded.drafts) != 2 {
		t.Fatalf("draft entries: %v", loaded.drafts)
	}
	output := ansi.Strip(renderList(loaded.models))
	for _, want := range []string{
		"Models  7\n\n",
		"DeepSeek-V4-Flash-0731",
		"155.4 GiB stored  ·  family deepseek-v4  ·  speculator dspark",
		"GLM-5.3-NVFP4",
		"GLM-5.3-NVFP4-Spark",
		"GLM-5.3 with NVFP4 routed experts",
		"432.9 GiB stored  ·  family glm  ·  speculator mtp",
		"defaults  ·  local-test: TP 8  ·  spark-test: TP 2",
		"98.6 GiB stored  ·  speculator mtp",
		"defaults  ·  local-test: TP 2  ·  spark-test: TP 2",
		"Topologies  2\n\n",
		"local-test",
		"local  ·  TP 1–12  ·  95.6 GiB/GPU  ·  sm_120a  ·  default TP fit",
		"spark-test",
		"Spark/RDMA  ·  TP 1–2  ·  108.0 GiB/rank  ·  sm_121a  ·  default TP all",
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
	spec, err := buildSpec(context.Background(), fs, values)
	if err != nil {
		t.Fatal(err)
	}
	if got := spec.VLLMArgv[len(spec.VLLMArgv)-1]; got != "--disable-log-requests" {
		t.Fatalf("last vLLM argument: got %q", got)
	}
	if !slices.Contains(spec.VLLMArgv, "--max-model-len") || spec.TPSize != 2 {
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
	_, err := buildSpec(context.Background(), fs, values)
	if err == nil || !strings.Contains(err.Error(), "launcher-managed") {
		t.Fatalf("managed argument error: %v", err)
	}
}

func TestConfigurationErrorsReturnUsageStatus(t *testing.T) {
	configureTestTopologies(t)
	status, err := execute(context.Background(), []string{
		"render",
		"Qwen3.8-Flash-Next-NVFP4",
		"--models-config", testModelsConfig(),
		"--model-path", "/does/not/exist",
	})
	if status != 2 || err == nil {
		t.Fatalf("status=%d error=%v", status, err)
	}

	status, err = execute(context.Background(), []string{"list", "unexpected"})
	if status != 2 || err == nil {
		t.Fatalf("list status=%d error=%v", status, err)
	}
}

func TestRenderJSONUsesSparkTopologyByKind(t *testing.T) {
	configureTestTopologies(t)
	fs := newFlagSet("render")
	var values launchFlags
	configureLaunchFlags(fs, &values, false)
	values.modelsConfig = testModelsConfig()
	values.config = "spark"
	if err := fs.Parse([]string{"GLM-5.3-Flash-NVFP4-Spark"}); err != nil {
		t.Fatal(err)
	}
	spec, err := buildSpec(context.Background(), fs, values)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Topology.Spark == nil || spec.TPSize != 2 || len(spec.SparkNodes) != 2 {
		t.Fatalf("unexpected Spark spec: %+v", spec)
	}
}

func TestClusterHelpReturnsSuccess(t *testing.T) {
	for _, args := range [][]string{
		{"cluster", "--help"},
		{"cluster", "status", "--help"},
	} {
		status, err := execute(context.Background(), args)
		if status != 0 || err != nil {
			t.Errorf("%v: status=%d error=%v", args, status, err)
		}
	}
}

func TestDFlashLaunchValidatesTheDraftEntry(t *testing.T) {
	configureTestTopologies(t)
	fs := newFlagSet("render")
	var values launchFlags
	configureLaunchFlags(fs, &values, false)
	values.modelsConfig = testModelsConfig()
	if err := fs.Parse([]string{"GLM-5.3-Flash-NVFP4", "--speculator", "dflash"}); err != nil {
		t.Fatal(err)
	}
	spec, err := buildSpec(context.Background(), fs, values)
	if err != nil {
		t.Fatal(err)
	}
	if spec.Metadata["speculator"] != "dflash" {
		t.Fatalf("speculator: %v", spec.Metadata["speculator"])
	}
	loaded, err := loadCatalog(context.Background(), testModelsConfig())
	if err != nil {
		t.Fatal(err)
	}
	profile := loaded.models["GLM-5.3-Flash-NVFP4"]
	profile.Model = "local-inference-lab/Other"
	if err := loaded.draftFor(profile); err == nil || !strings.Contains(err.Error(), "not compatible") {
		t.Fatalf("incompatible draft: %v", err)
	}
	profile.Speculators.DFlash.Model = "local-inference-lab/Nope"
	if err := loaded.draftFor(profile); err == nil || !strings.Contains(err.Error(), "no catalog draft entry") {
		t.Fatalf("missing draft: %v", err)
	}
}
