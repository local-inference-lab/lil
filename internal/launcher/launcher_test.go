// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package launcher

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"
)

const (
	qwenProfile     = "Qwen3.8-Flash-Next-NVFP4"
	glmFlashProfile = "GLM-5.3-Flash-NVFP4"
	glmProfile      = "GLM-5.3-NVFP4"
)

type testConfig struct {
	profiles map[string]ModelProfile
	local    Topology
	spark    Topology
}

func loadTestConfig(t *testing.T) testConfig {
	t.Helper()
	read := func(name string) []byte {
		t.Helper()
		data, err := os.ReadFile(filepath.Join("testdata", name))
		if err != nil {
			t.Fatal(err)
		}
		return data
	}
	basesPath := filepath.Join("..", "..", "configs", "models", "_bases.yaml")
	bases, err := os.ReadFile(basesPath)
	if err != nil {
		t.Fatal(err)
	}
	manifestRoot := filepath.Join("testdata", "model-manifests")
	entries, err := os.ReadDir(manifestRoot)
	if err != nil {
		t.Fatal(err)
	}
	profiles := map[string]ModelProfile{}
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		path := filepath.Join(manifestRoot, entry.Name(), "lil.yaml")
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		profile, err := LoadRepositoryModelProfile(
			bases, data, "local-inference-lab/"+entry.Name(), "", path,
		)
		if err != nil {
			t.Fatal(err)
		}
		profiles[profile.Name] = profile
	}
	local, err := LoadTopology(
		read("local.yaml"), "testdata/local.yaml", ".",
	)
	if err != nil {
		t.Fatal(err)
	}
	spark, err := LoadTopology(
		read("spark.yaml"), "testdata/spark.yaml", ".",
	)
	if err != nil {
		t.Fatal(err)
	}
	return testConfig{profiles: profiles, local: local, spark: spark}
}

func writeCheckpoint(t *testing.T, algorithm string, attentionHeads int) string {
	t.Helper()
	modelPath := t.TempDir()
	config := map[string]any{
		"architectures": []string{"TestArchitecture"},
		"text_config": map[string]any{
			"dtype":                    "bfloat16",
			"num_attention_heads":      attentionHeads,
			"num_hidden_layers":        45,
			"num_nextn_predict_layers": 1,
		},
	}
	if algorithm != "" {
		config["quantization_config"] = map[string]any{
			"quantized_layers": map[string]any{
				"model.language_model.layers.45.mlp.experts": map[string]any{
					"quant_algo": algorithm,
				},
			},
		}
	}
	writeJSON(t, filepath.Join(modelPath, "config.json"), config)
	writeJSON(t, filepath.Join(modelPath, "model.safetensors.index.json"), map[string]any{
		"metadata":   map[string]any{"total_size": "48000000000"},
		"weight_map": map[string]any{},
	})
	return modelPath
}

func writeJSON(t *testing.T, path string, value any) {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func intPointer(value int) *int              { return &value }
func floatPointer(value float64) *float64    { return &value }
func stringPointerTest(value string) *string { return &value }

func defaultOptions(modelPath string, tp int) LaunchOptions {
	return LaunchOptions{
		TPSize:         intPointer(tp),
		ModelPath:      stringPointerTest(modelPath),
		KVCacheDType:   "fp8",
		DCPSize:        1,
		DCPCommBackend: "a2a",
		AdaptiveWindow: 32,
		B12XPolicyMode: "auto",
	}
}

func optionValue(t *testing.T, argv []string, name string) string {
	t.Helper()
	index := slices.Index(argv, name)
	if index < 0 || index+1 == len(argv) {
		t.Fatalf("%s is absent or has no value in %v", name, argv)
	}
	if slices.Index(argv[index+1:], name) >= 0 {
		t.Fatalf("%s occurs more than once in %v", name, argv)
	}
	return argv[index+1]
}

func speculativeConfigValue(t *testing.T, spec LaunchSpec) map[string]any {
	t.Helper()
	var value map[string]any
	if err := json.Unmarshal(
		[]byte(optionValue(t, spec.VLLMArgv, "--speculative-config")), &value,
	); err != nil {
		t.Fatal(err)
	}
	return value
}

func TestProfileNamesMatchHuggingFaceRepositories(t *testing.T) {
	config := loadTestConfig(t)
	want := []string{glmFlashProfile, glmProfile, qwenProfile}
	if got := SortedProfileNames(config.profiles); !reflect.DeepEqual(got, want) {
		t.Fatalf("profile names: got %v, want %v", got, want)
	}
	for name, profile := range config.profiles {
		_, repository, ok := strings.Cut(profile.Model, "/")
		if !ok || repository != name {
			t.Errorf("profile %q has Hugging Face model %q", name, profile.Model)
		}
	}
}

func TestRepositoryManifestsInjectIdentityAndSeparateDrafts(t *testing.T) {
	bases, err := os.ReadFile(filepath.Join("..", "..", "configs", "models", "_bases.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	manifest := []byte(`schema_version: 1
kind: model
extends: glm
description: Repository model
served_model_name: served
expected_architectures: [GlmMoeDsaForCausalLM]
`)
	commit := "0123456789abcdef0123456789abcdef01234567"
	profile, err := LoadRepositoryModelProfile(
		bases, manifest, "local-inference-lab/Repository-Model", commit, "lil.yaml",
	)
	if err != nil {
		t.Fatal(err)
	}
	if profile.Name != "Repository-Model" || profile.Model != "local-inference-lab/Repository-Model" || profile.ManifestCommit != commit {
		t.Fatalf("repository identity was not injected: %+v", profile)
	}

	draftManifest := []byte(`schema_version: 1
kind: draft
description: DFlash draft
method: dflash
quantization: mxfp8
compatible_models:
  - local-inference-lab/Repository-Model
`)
	draft, err := LoadRepositoryDraftProfile(
		draftManifest, "local-inference-lab/Repository-Draft", commit, "lil.yaml",
	)
	if err != nil {
		t.Fatal(err)
	}
	if draft.Method != "dflash" || draft.CompatibleModels[0] != profile.Model {
		t.Fatalf("unexpected draft profile: %+v", draft)
	}
	if _, err := LoadRepositoryModelProfile(
		bases, draftManifest, draft.RepositoryID, commit, "lil.yaml",
	); err == nil || !strings.Contains(err.Error(), "kind must be model") {
		t.Fatalf("draft accepted as serving model: %v", err)
	}
}

func TestProfileInheritanceResolvesFamilyAndTPSettings(t *testing.T) {
	config := loadTestConfig(t)
	qwen := config.profiles[qwenProfile]
	glm := config.profiles[glmFlashProfile]
	qwenTP1 := qwen.Launch.Local.Resolve(1)
	if qwenTP1.Capacity.MaxModelLen != "auto" ||
		qwenTP1.Capacity.MaxNumSeqs != 8 ||
		qwenTP1.Capacity.MaxNumBatchedTokens != 2048 ||
		qwenTP1.Environment["VLLM_PLE_CPU_OFFLOAD"] != "1" {
		t.Fatalf("unexpected inherited Qwen TP=1 settings: %+v", qwenTP1)
	}
	if glm.DType != "bfloat16" || glm.ReasoningParser != "glm45" ||
		glm.Launch.SparkRDMA.Defaults.Capacity.MaxModelLen != "auto" {
		t.Fatalf("unexpected inherited GLM settings: %+v", glm)
	}
}

func TestProfileInheritanceAndStrictYAMLFailClosed(t *testing.T) {
	cycle := []byte(`
schema_version: 1
bases:
  first:
    extends: second
  second:
    extends: first
models:
  broken:
    extends: first
`)
	if _, err := LoadModelProfiles(cycle, "cycle.yaml"); err == nil ||
		!strings.Contains(err.Error(), "inheritance cycle") {
		t.Fatalf("cycle error: %v", err)
	}

	data := []byte("schema_version: 1\nbases: {}\nunexpected: true\n")
	if _, err := LoadModelProfiles(data, "invalid.yaml"); err == nil ||
		!strings.Contains(err.Error(), "unknown keys") {
		t.Fatalf("unknown-key error: %v", err)
	}
}

func TestEveryModelSupportsEveryValidLocalTP(t *testing.T) {
	config := loadTestConfig(t)
	tests := []struct {
		profile string
		tp      []int
		algo    string
	}{
		{qwenProfile, []int{1, 2, 3, 4, 6, 8, 12}, "W4A16_NVFP4"},
		{glmFlashProfile, []int{1, 2, 4, 8}, "MXFP8"},
		{glmProfile, []int{1, 2, 4, 8}, ""},
	}
	for _, test := range tests {
		t.Run(test.profile, func(t *testing.T) {
			profile := config.profiles[test.profile]
			modelPath := writeCheckpoint(t, test.algo, profile.AttentionHeads)
			for _, tp := range test.tp {
				spec, err := BuildLaunchSpec(
					profile, config.local, defaultOptions(modelPath, tp),
				)
				if err != nil {
					t.Fatalf("TP=%d: %v", tp, err)
				}
				if spec.TPSize != tp || len(spec.DeviceIDs) != tp ||
					optionValue(t, spec.VLLMArgv, "--tensor-parallel-size") != strconv.Itoa(tp) {
					t.Fatalf("bad TP=%d launch: %+v", tp, spec)
				}
			}
		})
	}
}

func TestInvalidTPFailsBeforeLaunch(t *testing.T) {
	config := loadTestConfig(t)
	modelPath := writeCheckpoint(t, "W4A16_NVFP4", 24)
	_, err := BuildLaunchSpec(
		config.profiles[qwenProfile], config.local, defaultOptions(modelPath, 5),
	)
	if err == nil || !strings.Contains(err.Error(), "attention heads") {
		t.Fatalf("invalid TP error: %v", err)
	}
}

func TestCommonEnvironmentAndQwenMultimodalContract(t *testing.T) {
	config := loadTestConfig(t)
	modelPath := writeCheckpoint(t, "W4A16_NVFP4", 24)
	spec, err := BuildLaunchSpec(
		config.profiles[qwenProfile], config.local, defaultOptions(modelPath, 1),
	)
	if err != nil {
		t.Fatal(err)
	}
	wantEnvironment := map[string]string{
		"CUDA_HOME":                   "/opt/cuda",
		"TRITON_PTXAS_PATH":           "/opt/cuda/bin/ptxas",
		"CUTE_DSL_ARCH":               "sm_120a",
		"PYTORCH_CUDA_ALLOC_CONF":     "expandable_segments:True",
		"B12X_POLICY_MODE":            "auto",
		"NCCL_IB_DISABLE":             "1",
		"VLLM_ENABLE_PCIE_ALLREDUCE":  "1",
		"VLLM_PCIE_ALLREDUCE_BACKEND": "b12x",
		"VLLM_PLE_CPU_OFFLOAD":        "1",
	}
	for name, want := range wantEnvironment {
		if got := spec.RuntimeEnvironment[name]; got != want {
			t.Errorf("%s: got %q, want %q", name, got, want)
		}
	}
	for _, name := range []string{"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"} {
		if !slices.Contains(spec.UnsetEnvironment, name) {
			t.Errorf("%s is not cleared for online Hugging Face resolution", name)
		}
	}
	if optionValue(t, spec.VLLMArgv, "--max-model-len") != "auto" ||
		optionValue(t, spec.VLLMArgv, "--max-num-seqs") != "8" ||
		optionValue(t, spec.VLLMArgv, "--kv-cache-dtype") != "fp8" {
		t.Fatalf("unexpected capacity argv: %v", spec.VLLMArgv)
	}
	for _, flag := range []string{
		"--mm-encoder-tp-mode", "--mm-processor-cache-gb", "--limit-mm-per-prompt",
	} {
		if !slices.Contains(spec.VLLMArgv, flag) {
			t.Errorf("Qwen multimodal argv is missing %s", flag)
		}
	}
	if got := speculativeConfigValue(t, spec)["moe_backend"]; got != "b12x" {
		t.Fatalf("Qwen MTP backend: got %v, want b12x", got)
	}
}

func TestMultimodalOptionsAreProfileScoped(t *testing.T) {
	config := loadTestConfig(t)
	modelPath := writeCheckpoint(t, "", 64)
	spec, err := BuildLaunchSpec(
		config.profiles[glmProfile], config.local,
		defaultOptions(modelPath, 8),
	)
	if err != nil {
		t.Fatal(err)
	}
	for _, flag := range []string{
		"--mm-encoder-tp-mode", "--mm-processor-cache-gb", "--limit-mm-per-prompt",
	} {
		if slices.Contains(spec.VLLMArgv, flag) {
			t.Errorf("GLM argv unexpectedly contains %s", flag)
		}
	}
}

func TestMTPBackendFollowsCheckpointQuantization(t *testing.T) {
	tests := []struct {
		algorithm    string
		quantization string
		backend      string
	}{
		{"W4A16_NVFP4", "nvfp4", "b12x"},
		{"MXFP8", "mxfp8", "triton"},
		{"BF16", "bf16", "triton"},
	}
	for _, test := range tests {
		t.Run(test.algorithm, func(t *testing.T) {
			modelPath := writeCheckpoint(t, test.algorithm, 64)
			decision, err := ResolveMTPMoEBackend(modelPath, "bfloat16")
			if err != nil {
				t.Fatal(err)
			}
			if decision.Quantization != test.quantization ||
				decision.Backend != test.backend {
				t.Fatalf("decision: got %+v", decision)
			}
		})
	}
}

func TestUnquantizedBF16MTPUsesTriton(t *testing.T) {
	modelPath := writeCheckpoint(t, "", 64)
	decision, err := ResolveMTPMoEBackend(modelPath, "bfloat16")
	if err != nil {
		t.Fatal(err)
	}
	if decision.Quantization != "bf16" || decision.Backend != "triton" {
		t.Fatalf("decision: got %+v", decision)
	}
}

func TestUnknownMTPQuantizationFailsClosed(t *testing.T) {
	modelPath := writeCheckpoint(t, "UNKNOWN_FP6", 64)
	_, err := ResolveMTPMoEBackend(modelPath, "bfloat16")
	if err == nil || !strings.Contains(err.Error(), "cannot determine MTP") {
		t.Fatalf("unknown quantization error: %v", err)
	}
}

func TestKVCacheAndCapacityAreCLIOverrides(t *testing.T) {
	config := loadTestConfig(t)
	modelPath := writeCheckpoint(t, "MXFP8", 64)
	options := defaultOptions(modelPath, 2)
	options.KVCacheDType = "bfloat16"
	options.GPUMemoryUtilization = floatPointer(0.87)
	options.MaxModelLen = stringPointerTest("32768")
	options.MaxNumSeqs = intPointer(17)
	spec, err := BuildLaunchSpec(config.profiles[glmFlashProfile], config.local, options)
	if err != nil {
		t.Fatal(err)
	}
	for flag, want := range map[string]string{
		"--kv-cache-dtype":         "bfloat16",
		"--gpu-memory-utilization": "0.87",
		"--max-model-len":          "32768",
		"--max-num-seqs":           "17",
	} {
		if got := optionValue(t, spec.VLLMArgv, flag); got != want {
			t.Errorf("%s: got %q, want %q", flag, got, want)
		}
	}
}

func TestCUDAGraphCaptureSizesCoverMTPVerificationAndMixedShapes(t *testing.T) {
	config := loadTestConfig(t)
	modelPath := writeCheckpoint(t, "", 64)
	options := defaultOptions(modelPath, 8)
	options.SpeculativeTokens = intPointer(3)
	spec, err := BuildLaunchSpec(config.profiles[glmProfile], config.local, options)
	if err != nil {
		t.Fatal(err)
	}
	var compilation map[string]any
	if err := json.Unmarshal(
		[]byte(optionValue(t, spec.VLLMArgv, "--compilation-config")),
		&compilation,
	); err != nil {
		t.Fatal(err)
	}
	want := []any{
		float64(1), float64(2), float64(4), float64(8),
		float64(12), float64(16), float64(20), float64(24),
		float64(28), float64(32), float64(40), float64(48),
		float64(56), float64(64),
	}
	if got := compilation["cudagraph_capture_sizes"]; !reflect.DeepEqual(got, want) {
		t.Fatalf("capture sizes: got %v, want %v", got, want)
	}
}

func TestCUDAGraphCaptureSizesFollowSequenceCountsWithoutMTP(t *testing.T) {
	config := loadTestConfig(t)
	modelPath := writeCheckpoint(t, "", 64)
	options := defaultOptions(modelPath, 8)
	options.Speculator = stringPointerTest("none")
	spec, err := BuildLaunchSpec(config.profiles[glmProfile], config.local, options)
	if err != nil {
		t.Fatal(err)
	}
	var compilation map[string]any
	if err := json.Unmarshal(
		[]byte(optionValue(t, spec.VLLMArgv, "--compilation-config")),
		&compilation,
	); err != nil {
		t.Fatal(err)
	}
	want := []any{
		float64(1), float64(2), float64(3), float64(4),
		float64(5), float64(6), float64(7), float64(8), float64(16),
	}
	if got := compilation["cudagraph_capture_sizes"]; !reflect.DeepEqual(got, want) {
		t.Fatalf("capture sizes: got %v, want %v", got, want)
	}
	if slices.Contains(spec.VLLMArgv, "--speculative-config") || spec.Metadata["speculative_tokens"] != 0 {
		t.Fatalf("MTP-off launch has speculative state: %+v", spec.Metadata)
	}
}

func TestCUDAGraphCaptureSizesKeepEveryMTPBatchShape(t *testing.T) {
	wantMTPShapes := []int{4, 8, 12, 16, 20, 24, 28, 32}
	got := cudagraphCaptureSizes(8, 4096, 4)
	for _, shape := range wantMTPShapes {
		if !slices.Contains(got, shape) {
			t.Errorf("capture sizes %v omit MTP verification shape %d", got, shape)
		}
	}
}

func TestMemoryEstimateUsesPostShardingWeightSize(t *testing.T) {
	config := loadTestConfig(t)
	modelPath := writeCheckpoint(t, "MXFP8", 64)
	spec, err := BuildLaunchSpec(
		config.profiles[glmFlashProfile], config.local,
		defaultOptions(modelPath, 2),
	)
	if err != nil {
		t.Fatal(err)
	}
	memory := spec.Metadata["memory"].(map[string]any)
	if got := memory["estimated_sharded_weight_bytes_per_rank"]; got != int64(24000000000) {
		t.Fatalf("estimated sharded bytes: got %v", got)
	}
	if memory["kv_cache_allocation"] != "vllm_runtime_profile" ||
		memory["estimate_is_advisory"] != true {
		t.Fatalf("unexpected memory policy: %+v", memory)
	}
}

func TestSparkSelectsFirstNNodesAndBuildsNativeDockerCommands(t *testing.T) {
	config := loadTestConfig(t)
	modelPath := writeCheckpoint(t, "W4A16_NVFP4", 24)
	spec, err := BuildLaunchSpec(
		config.profiles[qwenProfile], config.spark,
		defaultOptions(modelPath, 2),
	)
	if err != nil {
		t.Fatal(err)
	}
	if spec.DeviceIDs != nil || len(spec.SparkNodes) != 2 {
		t.Fatalf("unexpected Spark device mapping: %+v", spec)
	}
	for rank, node := range spec.SparkNodes {
		if node.Rank != rank || node.Node.SSHHost != []string{"tachyon", "luxon"}[rank] {
			t.Errorf("rank %d node: %+v", rank, node.Node)
		}
		if !reflect.DeepEqual(node.DockerArgv[:2], []string{"docker", "run"}) {
			t.Errorf("rank %d docker argv: %v", rank, node.DockerArgv)
		}
		if optionValue(t, node.VLLMArgv, "--nnodes") != "2" ||
			optionValue(t, node.VLLMArgv, "--node-rank") != strconv.Itoa(rank) {
			t.Errorf("rank %d distributed argv: %v", rank, node.VLLMArgv)
		}
		if slices.Contains(node.VLLMArgv, "--headless") != (rank > 0) {
			t.Errorf("rank %d headless mismatch", rank)
		}
	}
	if spec.RuntimeEnvironment["CUDA_VISIBLE_DEVICES"] != "0" ||
		spec.RuntimeEnvironment["VLLM_ENABLE_PCIE_ALLREDUCE"] != "0" ||
		spec.RuntimeEnvironment["NCCL_IB_GID_INDEX"] != "3" ||
		spec.RuntimeEnvironment["NCCL_IB_MERGE_NICS"] != "1" ||
		!slices.Contains(spec.VLLMArgv, "--disable-custom-all-reduce") {
		t.Fatalf("unexpected Spark runtime contract: %+v", spec)
	}
}

func TestGLMFlashSparkPolicyMatchesRDMAContract(t *testing.T) {
	config := loadTestConfig(t)
	modelPath := writeCheckpoint(t, "MXFP8", 64)
	spec, err := BuildLaunchSpec(
		config.profiles[glmFlashProfile], config.spark,
		defaultOptions(modelPath, 2),
	)
	if err != nil {
		t.Fatal(err)
	}
	for flag, want := range map[string]string{
		"--gpu-memory-utilization": "0.95",
		"--max-model-len":          "auto",
		"--max-num-seqs":           "8",
		"--max-num-batched-tokens": "4096",
	} {
		if got := optionValue(t, spec.VLLMArgv, flag); got != want {
			t.Errorf("%s: got %q, want %q", flag, got, want)
		}
	}
	if slices.Contains(spec.VLLMArgv, "--kv-cache-memory-bytes") {
		t.Fatalf("Spark policy must leave KV sizing to vLLM: %v", spec.VLLMArgv)
	}
	for name, want := range map[string]string{
		"INSTANTTENSOR_BUFFER_SIZE": "67108864",
		"INSTANTTENSOR_CHUNK_SIZE":  "8388608",
	} {
		if got := spec.RuntimeEnvironment[name]; got != want {
			t.Errorf("%s: got %q, want %q", name, got, want)
		}
	}
}

func TestSparkProfilerUsesRemotePathAndMount(t *testing.T) {
	config := loadTestConfig(t)
	modelPath := writeCheckpoint(t, "W4A16_NVFP4", 24)
	options := defaultOptions(modelPath, 2)
	options.Profiler = &ProfilerOptions{
		OutputDir: "traces", WithStack: true, UseGzip: true, MaxIterations: 4,
	}
	spec, err := BuildLaunchSpec(config.profiles[qwenProfile], config.spark, options)
	if err != nil {
		t.Fatal(err)
	}
	want := "/home/luke/projects/vllm/traces"
	if got, ok := ProfilerOutputDir(spec.VLLMArgv); !ok || got != want {
		t.Fatalf("profiler directory: got %q, %v; want %q", got, ok, want)
	}
	for _, node := range spec.SparkNodes {
		mount := "type=bind,src=/home/luke/projects/vllm,dst=/home/luke/projects/vllm"
		if !slices.Contains(node.DockerArgv, mount) {
			t.Errorf("rank %d missing runtime mount covering profiler path %q: %v", node.Rank, mount, node.DockerArgv)
		}
	}
}

func TestSparkTargetResolutionDoesNotInspectCheckpoint(t *testing.T) {
	config := loadTestConfig(t)
	target, err := ResolveSparkTarget(config.profiles[qwenProfile], config.spark, intPointer(1), nil)
	if err != nil {
		t.Fatal(err)
	}
	if target.TPSize != 1 || len(target.Nodes) != 1 || target.Nodes[0].SSHHost != "tachyon" ||
		target.ContainerName != "vllm-fleet-Qwen3.8-Flash-Next-NVFP4-tp1" {
		t.Fatalf("unexpected target: %+v", target)
	}
}

func TestSparkModelSyncRequiresExplicitModelPath(t *testing.T) {
	config := loadTestConfig(t)
	err := SyncSparkModel(LaunchSpec{
		Topology:    config.spark,
		ModelSource: "local-inference-lab/Qwen3.8-Flash-Next-NVFP4",
	})
	if err == nil || !strings.Contains(err.Error(), "--model-path") {
		t.Fatalf("sync error: %v", err)
	}
}

func TestManagedVLLMArgumentsCannotBeForwarded(t *testing.T) {
	config := loadTestConfig(t)
	modelPath := writeCheckpoint(t, "W4A16_NVFP4", 24)
	for _, arguments := range [][]string{
		{"--port", "9000"},
		{"--tensor_parallel_size=4"},
		{"-tp", "4"},
		{"--compilation-config.cudagraph-mode=NONE"},
		{"--headless"},
		{"--revision", "0123456789abcdef0123456789abcdef01234567"},
	} {
		options := defaultOptions(modelPath, 2)
		options.ExtraVLLMArgs = arguments
		_, err := BuildLaunchSpec(config.profiles[qwenProfile], config.local, options)
		if err == nil || !strings.Contains(err.Error(), "launcher-managed") {
			t.Errorf("arguments %v: %v", arguments, err)
		}
	}
}

func TestHubLaunchDownloadsUnpinnedTargetAndDraftRepositories(t *testing.T) {
	config := loadTestConfig(t)
	options := defaultOptions("", 2)
	options.ModelPath = nil
	options.Speculator = stringPointerTest("dflash2")
	spec, err := BuildLaunchSpec(
		config.profiles[glmFlashProfile], config.local, options,
	)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{
		"local-inference-lab/GLM-5.3-Flash-NVFP4",
		"local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8",
	}
	if !reflect.DeepEqual(spec.DownloadRepositories, want) {
		t.Fatalf("download repositories: got %v, want %v", spec.DownloadRepositories, want)
	}
	if slices.Contains(spec.VLLMArgv, "--revision") || spec.CheckpointPath != nil {
		t.Fatalf("Hub launch is unexpectedly revision-bound: %+v", spec)
	}
	if config := speculativeConfigValue(t, spec); config["revision"] != nil {
		t.Fatalf("DFlash configuration contains a revision: %+v", config)
	}
}

func TestExplicitModelPathSkipsTargetCacheDownload(t *testing.T) {
	config := loadTestConfig(t)
	modelPath := writeCheckpoint(t, "W4A16_NVFP4", 24)
	spec, err := BuildLaunchSpec(
		config.profiles[qwenProfile], config.local, defaultOptions(modelPath, 2),
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(spec.DownloadRepositories) != 0 {
		t.Fatalf("local checkpoint requested Hub downloads: %v", spec.DownloadRepositories)
	}
}

func TestLocalHuggingFaceDownloadUsesConfiguredEnvironmentOnline(t *testing.T) {
	directory := t.TempDir()
	bin := filepath.Join(directory, "bin")
	if err := os.Mkdir(bin, 0o755); err != nil {
		t.Fatal(err)
	}
	logPath := filepath.Join(directory, "hf.log")
	hf := filepath.Join(bin, "hf")
	script := `#!/bin/sh
if env | grep -q '^HF_HUB_OFFLINE=' || env | grep -q '^TRANSFORMERS_OFFLINE=' || env | grep -q '^HF_HUB_DISABLE_PROGRESS_BARS='; then
  exit 9
fi
printf '%s\n' "$*" > "$LIL_TEST_HF_LOG"
`
	if err := os.WriteFile(hf, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("HF_HUB_OFFLINE", "1")
	t.Setenv("TRANSFORMERS_OFFLINE", "1")
	t.Setenv("HF_HUB_DISABLE_PROGRESS_BARS", "1")
	t.Setenv("LIL_TEST_HF_LOG", logPath)
	spec := LaunchSpec{
		Topology:             Topology{Local: &LocalTopology{Python: filepath.Join(bin, "python")}},
		DownloadRepositories: []string{"local-inference-lab/Test-Model"},
		UnsetEnvironment:     []string{"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"},
	}
	if err := SyncHuggingFaceCache(spec); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.TrimSpace(string(data)); got != "download local-inference-lab/Test-Model" {
		t.Fatalf("hf arguments: %q", got)
	}
}

func TestMissingHuggingFaceCLIDoesNotBlockLaunch(t *testing.T) {
	directory := t.TempDir()
	t.Setenv("PATH", directory)
	spec := LaunchSpec{
		Topology: Topology{Local: &LocalTopology{
			Python: filepath.Join(directory, "missing", "python"),
		}},
		DownloadRepositories: []string{"local-inference-lab/Test-Model"},
	}
	if err := SyncHuggingFaceCache(spec); err != nil {
		t.Fatalf("missing hf CLI blocked launch: %v", err)
	}
}

func TestHuggingFaceDownloadFailureBlocksLaunch(t *testing.T) {
	directory := t.TempDir()
	bin := filepath.Join(directory, "bin")
	if err := os.Mkdir(bin, 0o755); err != nil {
		t.Fatal(err)
	}
	hf := filepath.Join(bin, "hf")
	if err := os.WriteFile(hf, []byte("#!/bin/sh\nexit 23\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	spec := LaunchSpec{
		Topology: Topology{Local: &LocalTopology{
			Python: filepath.Join(bin, "python"),
		}},
		DownloadRepositories: []string{"local-inference-lab/Test-Model"},
	}
	err := SyncHuggingFaceCache(spec)
	if err == nil || !strings.Contains(err.Error(), "download failed") {
		t.Fatalf("download failure was not propagated: %v", err)
	}
}

func TestSparkHuggingFaceCommandsUseRuntimeCLIAndPersistentCache(t *testing.T) {
	topology := &SparkRDMATopology{
		RuntimePython: "/opt/vllm/.venv/bin/python",
		Image:         "vllm:test",
		CacheMounts: []CacheMount{{
			Source: "/srv/cache/huggingface", Target: "/root/.cache/huggingface",
		}},
	}
	host := sparkHostHFArgv(
		topology, "/opt/vllm/.venv/bin/hf", "download", "org/model",
	)
	for _, want := range []string{
		"HF_HOME=/srv/cache/huggingface", "/opt/vllm/.venv/bin/hf",
		"download", "org/model",
	} {
		if !slices.Contains(host, want) {
			t.Errorf("host hf command is missing %q: %v", want, host)
		}
	}
	container := sparkContainerHFArgv(topology, "download", "org/model")
	for _, want := range []string{
		"type=bind,src=/srv/cache/huggingface,dst=/root/.cache/huggingface",
		"--entrypoint", "hf", "vllm:test", "download", "org/model",
	} {
		if !slices.Contains(container, want) {
			t.Errorf("container hf command is missing %q: %v", want, container)
		}
	}
}

func TestShellAndJSONRenderExposeResolvedCommand(t *testing.T) {
	config := loadTestConfig(t)
	modelPath := writeCheckpoint(t, "", 64)
	spec, err := BuildLaunchSpec(
		config.profiles[glmProfile], config.local,
		defaultOptions(modelPath, 8),
	)
	if err != nil {
		t.Fatal(err)
	}
	shell := ShellRender(spec)
	if !strings.Contains(shell, " \\\n  --served-model-name GLM-5.3 \\\n") ||
		!strings.HasPrefix(shell, "unset CUDA_VISIBLE_DEVICES\n") {
		t.Fatalf("unexpected shell render:\n%s", shell)
	}
	encoded, err := JSONRender(spec)
	if err != nil {
		t.Fatal(err)
	}
	var rendered map[string]any
	if err := json.Unmarshal(encoded, &rendered); err != nil {
		t.Fatal(err)
	}
	if rendered["model"] != glmProfile || rendered["topology_kind"] != "local" ||
		rendered["tensor_parallel_size"] != float64(8) {
		t.Fatalf("unexpected JSON render: %s", encoded)
	}
}
