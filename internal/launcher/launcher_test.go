// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package launcher

import (
	"context"
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
	qwenProfile           = "Qwen3.8-Flash-Next-NVFP4"
	glmFlashProfile       = "GLM-5.3-Flash-NVFP4"
	glmFlashSparkProfile  = "GLM-5.3-Flash-NVFP4-Spark"
	glmProfile            = "GLM-5.3-NVFP4"
	glmSparkProfile       = "GLM-5.3-NVFP4-Spark"
	deepseekProfile       = "DeepSeek-V4-Flash-0731"
	deepseekVisionProfile = "DeepSeek-V4-Flash-Vision-Exp"
	deepseekRevision      = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"
	testCommit            = "0123456789abcdef0123456789abcdef01234567"
)

type testConfig struct {
	profiles map[string]ModelProfile
	local    Topology
	spark    Topology
	families []byte
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
	families, err := os.ReadFile(filepath.Join("..", "..", "configs", "models", "_bases.yaml"))
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
		directory := filepath.Join(manifestRoot, entry.Name())
		data, err := os.ReadFile(filepath.Join(directory, "lil.yaml"))
		if err != nil {
			t.Fatal(err)
		}
		external, err := ManifestNamesModel(data, directory+"/lil.yaml")
		if err != nil {
			t.Fatal(err)
		}
		var profile ModelProfile
		if external {
			profile, err = LoadCatalogModelProfile(families, data, entry.Name(), testCommit, directory+"/lil.yaml")
		} else {
			profile, err = LoadRepositoryModelProfile(
				families, data, "local-inference-lab/"+entry.Name(), testCommit, directory+"/lil.yaml",
			)
		}
		if err != nil {
			t.Fatal(err)
		}
		profile.Facts, err = LoadCheckpointFacts(directory)
		if err != nil {
			t.Fatal(err)
		}
		profiles[profile.Name] = profile
	}
	local, err := LoadTopology(read("local.yaml"), "testdata/local.yaml", ".")
	if err != nil {
		t.Fatal(err)
	}
	spark, err := LoadTopology(read("spark.yaml"), "testdata/spark.yaml", ".")
	if err != nil {
		t.Fatal(err)
	}
	return testConfig{profiles: profiles, local: local, spark: spark, families: families}
}

// writeCheckpoint creates a metadata-only checkpoint whose config.json names
// the given architecture and head count and, optionally, an MTP expert
// quantization.
func writeCheckpoint(t *testing.T, algorithm, architecture string, attentionHeads int) string {
	t.Helper()
	modelPath := t.TempDir()
	config := map[string]any{
		"architectures": []string{architecture},
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

func defaultOptions(tp int) LaunchOptions {
	options := LaunchOptions{
		KVCacheDType:   "fp8",
		DCPSize:        1,
		DCPCommBackend: "a2a",
		AdaptiveWindow: 32,
		B12XPolicyMode: "auto",
	}
	if tp > 0 {
		options.TPSize = intPointer(tp)
	}
	return options
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

func loadManifest(t *testing.T, families []byte, name, manifest string) (ModelProfile, error) {
	t.Helper()
	return LoadRepositoryModelProfile(families, []byte(manifest), "local-inference-lab/"+name, testCommit, name+"/lil.yaml")
}

func TestProfileNamesMatchHuggingFaceRepositories(t *testing.T) {
	config := loadTestConfig(t)
	want := []string{
		deepseekProfile, deepseekVisionProfile,
		glmFlashProfile, glmFlashSparkProfile, glmProfile, glmSparkProfile,
		qwenProfile,
	}
	if got := SortedProfileNames(config.profiles); !reflect.DeepEqual(got, want) {
		t.Fatalf("profile names: got %v, want %v", got, want)
	}
	for name, profile := range config.profiles {
		owner, repository, ok := strings.Cut(profile.Model, "/")
		if !ok || repository != name || profile.ManifestCommit != testCommit {
			t.Errorf("profile %q has Hugging Face model %q commit %q", name, profile.Model, profile.ManifestCommit)
		}
		if (owner != "local-inference-lab") != (profile.Revision != "") {
			t.Errorf("profile %q owner %q revision %q: only catalog entries pin revisions", name, owner, profile.Revision)
		}
	}
}

func TestCheckpointFactsComeFromTheCheckpointNotTheManifest(t *testing.T) {
	config := loadTestConfig(t)
	for name, want := range map[string]struct {
		architecture string
		heads        int
		weightBytes  int64
	}{
		glmProfile:      {"GlmMoeDsaForCausalLM", 64, 464823066832},
		glmFlashProfile: {"Glm5NextForConditionalGeneration", 64, 198042331512},
		qwenProfile:     {"Qwen3_8FlashNextForConditionalGeneration", 24, 105839538520},
	} {
		facts := config.profiles[name].Facts
		if facts == nil || facts.Architectures[0] != want.architecture ||
			facts.AttentionHeads != want.heads || facts.WeightBytes != want.weightBytes {
			t.Errorf("%s facts: %+v", name, facts)
		}
	}
}

func TestRepositoryManifestsInjectIdentityAndSeparateDrafts(t *testing.T) {
	config := loadTestConfig(t)
	profile, err := loadManifest(t, config.families, "Repository-Model", `schema_version: 1
kind: model
family: glm
description: Repository model
serving:
  served_model_name: served
`)
	if err != nil {
		t.Fatal(err)
	}
	if profile.Name != "Repository-Model" || profile.Model != "local-inference-lab/Repository-Model" ||
		profile.ManifestCommit != testCommit || profile.Family != "glm" {
		t.Fatalf("repository identity was not injected: %+v", profile)
	}
	if profile.Speculators.Default != "none" || profile.Kernels.Attention == nil || *profile.Kernels.Attention != "B12X" {
		t.Fatalf("family defaults were not applied: %+v", profile)
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
		draftManifest, "local-inference-lab/Repository-Draft", testCommit, "lil.yaml",
	)
	if err != nil {
		t.Fatal(err)
	}
	if draft.Method != "dflash" || draft.CompatibleModels[0] != profile.Model {
		t.Fatalf("unexpected draft profile: %+v", draft)
	}
	if _, err := LoadRepositoryModelProfile(
		config.families, draftManifest, draft.RepositoryID, testCommit, "lil.yaml",
	); err == nil || !strings.Contains(err.Error(), "kind must be model") {
		t.Fatalf("draft accepted as serving model: %v", err)
	}
}

func TestFamilyValuesMergeAndExplicitNullClears(t *testing.T) {
	config := loadTestConfig(t)
	glm := config.profiles[glmProfile]
	if glm.Serving.ReasoningParser != "glm45" || glm.Serving.ToolCallParser != "glm47" ||
		!glm.Serving.AutoToolChoice || glm.Base.Environment["VLLM_SSM_CONV_STATE_LAYOUT"] != "DS" ||
		glm.Base.Environment["CUDA_DEVICE_MAX_CONNECTIONS"] != "32" {
		t.Fatalf("GLM family values were not merged: %+v", glm)
	}
	profile, err := loadManifest(t, config.families, "Cleared", `schema_version: 1
kind: model
family: glm
description: Family with cleared attention backend
serving:
  served_model_name: cleared
  tool_call_parser: null
kernels:
  attention: null
`)
	if err != nil {
		t.Fatal(err)
	}
	if profile.Kernels.Attention != nil || profile.Serving.ToolCallParser != "" || profile.Serving.AutoToolChoice {
		t.Fatalf("explicit null did not clear family values: %+v", profile)
	}
}

func TestManifestSchemaFailsClosed(t *testing.T) {
	config := loadTestConfig(t)
	cases := map[string]struct {
		manifest string
		want     string
	}{
		"unknown key":                     {"schema_version: 1\nkind: model\ndescription: x\nserving: {served_model_name: x}\nweight_bytes: 1\n", "unknown keys"},
		"model restated":                  {"schema_version: 1\nkind: model\nmodel: a/b\ndescription: x\nserving: {served_model_name: x}\n", "must not set model"},
		"unknown family":                  {"schema_version: 1\nkind: model\nfamily: nope\ndescription: x\nserving: {served_model_name: x}\n", "unknown model family"},
		"mtp default without section":     {"schema_version: 1\nkind: model\ndescription: x\nserving: {served_model_name: x}\nspeculators: {default: mtp}\n", "no mtp section"},
		"auto tool choice without parser": {"schema_version: 1\nkind: model\ndescription: x\nserving: {served_model_name: x, auto_tool_choice: true}\n", "requires tool_call_parser"},
		"derived environment":             {"schema_version: 1\nkind: model\ndescription: x\nserving: {served_model_name: x}\nenvironment: {CUTE_DSL_ARCH: sm_90a}\n", "derived by the launcher"},
		"unquoted environment value":      {"schema_version: 1\nkind: model\ndescription: x\nserving: {served_model_name: x}\nenvironment: {VLLM_PLE_CPU_OFFLOAD: 1}\n", "quote"},
		"override without condition":      {"schema_version: 1\nkind: model\ndescription: x\nserving: {served_model_name: x}\noverrides: [{when: {}, capacity: {max_num_seqs: 4}}]\n", "at least one condition"},
		"bad capacity key":                {"schema_version: 1\nkind: model\ndescription: x\nserving: {served_model_name: x}\ncapacity: {max_seqs: 4}\n", "unknown keys"},
		"bad moe quantization":            {"schema_version: 1\nkind: model\ndescription: x\nserving: {served_model_name: x}\nspeculators: {mtp: {moe_quantization: fp6}}\n", "moe_quantization"},
	}
	for name, test := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := loadManifest(t, config.families, "Broken", test.manifest)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error %v does not mention %q", err, test.want)
			}
		})
	}
	cycle := []byte("schema_version: 1\nfamilies:\n  first: {family: second}\n  second: {family: first}\n")
	if _, err := loadManifest(t, cycle, "Cycle", "schema_version: 1\nkind: model\nfamily: first\ndescription: x\nserving: {served_model_name: x}\n"); err == nil ||
		!strings.Contains(err.Error(), "inheritance cycle") {
		t.Fatalf("cycle error: %v", err)
	}
}

func TestDefaultTPFollowsFitOrWholeCluster(t *testing.T) {
	config := loadTestConfig(t)
	for name, want := range map[string]struct{ local, spark int }{
		glmProfile:            {8, 2},
		glmSparkProfile:       {8, 2},
		glmFlashProfile:       {4, 2},
		glmFlashSparkProfile:  {4, 2},
		qwenProfile:           {2, 2},
		deepseekProfile:       {2, 2},
		deepseekVisionProfile: {2, 2},
	} {
		profile := config.profiles[name]
		for _, item := range []struct {
			topology Topology
			want     int
		}{{config.local, want.local}, {config.spark, want.spark}} {
			spec, err := BuildLaunchSpec(profile, item.topology, defaultOptions(0))
			if err != nil {
				t.Fatalf("%s on %s: %v", name, item.topology.Name(), err)
			}
			if spec.TPSize != item.want {
				t.Errorf("%s on %s: default TP %d, want %d", name, item.topology.Name(), spec.TPSize, item.want)
			}
		}
	}
	huge := config.profiles[glmProfile]
	facts := *huge.Facts
	facts.WeightBytes = 4 << 40
	huge.Facts = &facts
	_, err := BuildLaunchSpec(huge, config.local, defaultOptions(0))
	if err == nil || !strings.Contains(err.Error(), "do not fit") {
		t.Fatalf("oversized checkpoint error: %v", err)
	}
	spec, err := BuildLaunchSpec(huge, config.local, defaultOptions(8))
	if err != nil {
		t.Fatalf("explicit TP must bypass the fit rule: %v", err)
	}
	if memory := spec.Metadata["memory"].(map[string]any); memory["fits"] != false {
		t.Fatalf("memory estimate should report no headroom: %+v", memory)
	}
}

func TestEveryModelSupportsEveryValidLocalTP(t *testing.T) {
	config := loadTestConfig(t)
	tests := []struct {
		profile string
		tp      []int
	}{
		{qwenProfile, []int{1, 2, 3, 4, 6, 8, 12}},
		{glmFlashProfile, []int{1, 2, 4, 8}},
		{glmFlashSparkProfile, []int{1, 2, 4, 8}},
		{glmProfile, []int{1, 2, 4, 8}},
		{glmSparkProfile, []int{1, 2, 4, 8}},
	}
	for _, test := range tests {
		t.Run(test.profile, func(t *testing.T) {
			profile := config.profiles[test.profile]
			for _, tp := range test.tp {
				spec, err := BuildLaunchSpec(profile, config.local, defaultOptions(tp))
				if err != nil {
					t.Fatalf("TP=%d: %v", tp, err)
				}
				if spec.TPSize != tp || len(spec.DeviceIDs) != tp ||
					optionValue(t, spec.VLLMArgv, "--tensor-parallel-size") != strconv.Itoa(tp) {
					t.Fatalf("bad TP=%d launch: %+v", tp, spec)
				}
			}
			supported := spec(t, profile, config.local).Metadata["tensor_parallel"].(map[string]any)["supported"].([]int)
			if !reflect.DeepEqual(supported, test.tp) {
				t.Fatalf("supported TP sizes: got %v, want %v", supported, test.tp)
			}
		})
	}
}

func spec(t *testing.T, profile ModelProfile, topology Topology) LaunchSpec {
	t.Helper()
	result, err := BuildLaunchSpec(profile, topology, defaultOptions(0))
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func TestInvalidTPFailsBeforeLaunch(t *testing.T) {
	config := loadTestConfig(t)
	_, err := BuildLaunchSpec(config.profiles[qwenProfile], config.local, defaultOptions(5))
	if err == nil || !strings.Contains(err.Error(), "attention heads") {
		t.Fatalf("invalid TP error: %v", err)
	}
	_, err = BuildLaunchSpec(config.profiles[qwenProfile], config.spark, defaultOptions(3))
	if err == nil || !strings.Contains(err.Error(), "one-GPU nodes") {
		t.Fatalf("oversized Spark TP error: %v", err)
	}
}

func TestOverridesApplyByTPAndKind(t *testing.T) {
	config := loadTestConfig(t)
	qwen := config.profiles[qwenProfile]
	cases := []struct {
		topology Topology
		tp       int
		tokens   string
		ple      string
		fused    bool
		applied  []int
	}{
		{config.local, 1, "2048", "1", false, []int{0}},
		{config.local, 2, "4096", "0", false, []int{}},
		{config.spark, 1, "2048", "1", true, []int{0, 1}},
		{config.spark, 2, "2048", "0", true, []int{1}},
	}
	for _, test := range cases {
		spec, err := BuildLaunchSpec(qwen, test.topology, defaultOptions(test.tp))
		if err != nil {
			t.Fatal(err)
		}
		if got := optionValue(t, spec.VLLMArgv, "--max-num-batched-tokens"); got != test.tokens {
			t.Errorf("%s TP=%d batched tokens %s, want %s", test.topology.Name(), test.tp, got, test.tokens)
		}
		if got := spec.RuntimeEnvironment["VLLM_PLE_CPU_OFFLOAD"]; got != test.ple {
			t.Errorf("%s TP=%d PLE %q, want %q", test.topology.Name(), test.tp, got, test.ple)
		}
		var compilation map[string]any
		if err := json.Unmarshal([]byte(optionValue(t, spec.VLLMArgv, "--compilation-config")), &compilation); err != nil {
			t.Fatal(err)
		}
		_, fused := compilation["pass_config"]
		if fused != test.fused {
			t.Errorf("%s TP=%d fused pass config %v, want %v", test.topology.Name(), test.tp, fused, test.fused)
		}
		if got := spec.Metadata["applied_overrides"].([]int); !reflect.DeepEqual(got, test.applied) {
			t.Errorf("%s TP=%d applied overrides %v, want %v", test.topology.Name(), test.tp, got, test.applied)
		}
	}
	local1, _ := BuildLaunchSpec(qwen, config.local, defaultOptions(1))
	if local1.RuntimeEnvironment["INSTANTTENSOR_BUFFER_SIZE"] != "1342177280" {
		t.Fatalf("TP=1 loader tuning missing: %+v", local1.RuntimeEnvironment)
	}
}

func TestEnvironmentLayersTopologyDerivedManifestAndCLI(t *testing.T) {
	config := loadTestConfig(t)
	options := defaultOptions(2)
	options.EnvironmentOverrides = []EnvironmentOverride{{"OMP_NUM_THREADS", "8"}, {"MY_EXPERIMENT", "1"}}
	spec, err := BuildLaunchSpec(config.profiles[glmFlashProfile], config.local, options)
	if err != nil {
		t.Fatal(err)
	}
	for name, want := range map[string]string{
		"CUDA_DEVICE_ORDER":           "PCI_BUS_ID",
		"NCCL_IB_DISABLE":             "1",
		"VLLM_ENABLE_PCIE_ALLREDUCE":  "1",
		"VLLM_PCIE_ALLREDUCE_BACKEND": "b12x",
		"CUDA_HOME":                   "/opt/cuda",
		"TRITON_PTXAS_PATH":           "/opt/cuda/bin/ptxas",
		"CUTE_DSL_ARCH":               "sm_120a",
		"B12X_POLICY_MODE":            "auto",
		"VLLM_SSM_CONV_STATE_LAYOUT":  "DS",
		"INSTANTTENSOR_BUFFER_SIZE":   "67108864",
		"INSTANTTENSOR_CHUNK_SIZE":    "8388608",
		"OMP_NUM_THREADS":             "8",
		"MY_EXPERIMENT":               "1",
	} {
		if got := spec.RuntimeEnvironment[name]; got != want {
			t.Errorf("%s: got %q, want %q", name, got, want)
		}
	}
	if !strings.HasPrefix(spec.RuntimeEnvironment["PYTHONPATH"], "/home/luke/projects/vllm-hh-rebase:/home/luke/projects/b12x") {
		t.Errorf("PYTHONPATH: %q", spec.RuntimeEnvironment["PYTHONPATH"])
	}
	for _, name := range []string{"CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"} {
		if !slices.Contains(spec.UnsetEnvironment, name) {
			t.Errorf("%s is not cleared", name)
		}
	}
	for _, name := range []string{"CUTE_DSL_ARCH", "CUDA_VISIBLE_DEVICES", "VLLM_HOST_IP"} {
		options := defaultOptions(2)
		options.EnvironmentOverrides = []EnvironmentOverride{{name, "x"}}
		if _, err := BuildLaunchSpec(config.profiles[glmFlashProfile], config.local, options); err == nil ||
			!strings.Contains(err.Error(), "derived") {
			t.Errorf("--env %s was accepted: %v", name, err)
		}
	}
	options = defaultOptions(2)
	options.PLECPUOffload = new(bool)
	if _, err := BuildLaunchSpec(config.profiles[glmFlashProfile], config.local, options); err == nil ||
		!strings.Contains(err.Error(), "VLLM_PLE_CPU_OFFLOAD") {
		t.Fatalf("PLE switch on a model without PLE policy: %v", err)
	}
}

func TestSparkEnvironmentUsesRecordedTuningAndDerivedRankValues(t *testing.T) {
	config := loadTestConfig(t)
	spec, err := BuildLaunchSpec(config.profiles[qwenProfile], config.spark, defaultOptions(2))
	if err != nil {
		t.Fatal(err)
	}
	for name, want := range map[string]string{
		"CUDA_VISIBLE_DEVICES":       "0",
		"VLLM_ENABLE_PCIE_ALLREDUCE": "0",
		"NCCL_IB_GID_INDEX":          "3",
		"NCCL_IB_MERGE_NICS":         "1",
		"NCCL_DEBUG":                 "INFO",
		"NCCL_NET_PLUGIN":            "none",
		"INSTANTTENSOR_BUFFER_SIZE":  "1342177280",
		"PYTHONPATH":                 "/home/luke/projects/vllm:/home/luke/projects/b12x",
	} {
		if got := spec.RuntimeEnvironment[name]; got != want {
			t.Errorf("%s: got %q, want %q", name, got, want)
		}
	}
	rank1 := spec.SparkNodes[1].RuntimeEnvironment
	if rank1["VLLM_HOST_IP"] != "10.200.0.2" || rank1["NCCL_IB_HCA"] != "rocep1s0f0,roceP2p1s0f0" ||
		rank1["NCCL_SOCKET_IFNAME"] != "enp1s0f0np0" || rank1["NCCL_IB_DISABLE"] != "0" {
		t.Fatalf("rank 1 environment: %+v", rank1)
	}
	if !slices.Contains(spec.VLLMArgv, "--disable-custom-all-reduce") {
		t.Fatalf("Spark launch must disable custom all-reduce: %v", spec.VLLMArgv)
	}
}

func TestTopologyRequiresEnvironmentBlock(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("testdata", "local.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	stripped := string(data[:strings.Index(string(data), "environment:")])
	if _, err := LoadTopology([]byte(stripped), "stripped.yaml", "."); err == nil ||
		!strings.Contains(err.Error(), "environment is required") {
		t.Fatalf("topology without environment: %v", err)
	}
	if _, err := LoadTopology([]byte(stripped+"environment:\n  CUTE_DSL_ARCH: sm_90a\n"), "derived.yaml", "."); err == nil ||
		!strings.Contains(err.Error(), "derived") {
		t.Fatalf("topology with derived variable: %v", err)
	}
	if _, err := LoadTopology([]byte(strings.Replace(string(data), "default_tp: fit", "default_tp: most", 1)), "policy.yaml", "."); err == nil ||
		!strings.Contains(err.Error(), "default_tp") {
		t.Fatalf("topology with bad default_tp: %v", err)
	}
}

func TestQwenMultimodalAndKernelContract(t *testing.T) {
	config := loadTestConfig(t)
	spec, err := BuildLaunchSpec(config.profiles[qwenProfile], config.local, defaultOptions(1))
	if err != nil {
		t.Fatal(err)
	}
	for flag, want := range map[string]string{
		"--mm-encoder-tp-mode":        "data",
		"--mm-processor-cache-gb":     "0",
		"--limit-mm-per-prompt":       `{"image":1}`,
		"--gdn-decode-kernel":         "b12x",
		"--block-size":                "16",
		"--mamba-cache-mode":          "align",
		"--model-loader-extra-config": `{"instanttensor_copy":false}`,
		"--reasoning-parser":          "qwen3",
		"--tool-call-parser":          "qwen3_xml",
		"--max-model-len":             "auto",
		"--max-num-seqs":              "8",
		"--kv-cache-dtype":            "fp8",
		"--quantization":              "modelopt_mixed",
	} {
		if got := optionValue(t, spec.VLLMArgv, flag); got != want {
			t.Errorf("%s: got %q, want %q", flag, got, want)
		}
	}
	for _, flag := range []string{"--no-enable-flashinfer-autotune", "--enable-auto-tool-choice", "--async-scheduling", "--enable-prefix-caching", "--enable-chunked-prefill"} {
		if !slices.Contains(spec.VLLMArgv, flag) {
			t.Errorf("Qwen argv is missing %s", flag)
		}
	}
	if slices.Contains(spec.VLLMArgv, "--attention-backend") {
		t.Errorf("Qwen argv must not select an attention backend: %v", spec.VLLMArgv)
	}
	if got := speculativeConfigValue(t, spec)["moe_backend"]; got != "b12x" {
		t.Fatalf("Qwen MTP backend: got %v, want b12x", got)
	}
}

func TestMultimodalOptionsAreProfileScoped(t *testing.T) {
	config := loadTestConfig(t)
	spec, err := BuildLaunchSpec(config.profiles[glmProfile], config.local, defaultOptions(8))
	if err != nil {
		t.Fatal(err)
	}
	for _, flag := range []string{
		"--mm-encoder-tp-mode", "--mm-processor-cache-gb", "--limit-mm-per-prompt", "--hf-overrides",
	} {
		if slices.Contains(spec.VLLMArgv, flag) {
			t.Errorf("GLM argv unexpectedly contains %s", flag)
		}
	}
	if got := optionValue(t, spec.VLLMArgv, "--attention-backend"); got != "B12X" {
		t.Errorf("GLM attention backend: %q", got)
	}
}

func TestOptionalServingFlagsFollowTheManifest(t *testing.T) {
	config := loadTestConfig(t)
	profile, err := loadManifest(t, config.families, "Plain", `schema_version: 1
kind: model
description: Base model without tool calling or speculation
serving:
  served_model_name: plain
  prefix_caching: false
`)
	if err != nil {
		t.Fatal(err)
	}
	profile.Facts = config.profiles[glmProfile].Facts
	spec, err := BuildLaunchSpec(profile, config.local, defaultOptions(8))
	if err != nil {
		t.Fatal(err)
	}
	for _, flag := range []string{
		"--reasoning-parser", "--tool-call-parser", "--enable-auto-tool-choice",
		"--speculative-config", "--enable-prefix-caching", "--compilation-config",
	} {
		if slices.Contains(spec.VLLMArgv, flag) {
			t.Errorf("plain argv unexpectedly contains %s", flag)
		}
	}
	if !slices.Contains(spec.VLLMArgv, "--enable-chunked-prefill") || spec.Metadata["speculator"] != "none" {
		t.Fatalf("plain launch defaults: %v %+v", spec.VLLMArgv, spec.Metadata)
	}
	options := defaultOptions(8)
	options.Speculator = stringPointerTest("mtp")
	if _, err := BuildLaunchSpec(profile, config.local, options); err == nil || !strings.Contains(err.Error(), "does not define an MTP speculator") {
		t.Fatalf("MTP on a model without an MTP section: %v", err)
	}
}

func TestRequiredArchitectureFailsClosed(t *testing.T) {
	config := loadTestConfig(t)
	profile := config.profiles[glmFlashSparkProfile]
	profile.Requires = Requirements{Arch: []string{"sm_121a"}}
	if _, err := BuildLaunchSpec(profile, config.local, defaultOptions(4)); err == nil || !strings.Contains(err.Error(), "requires sm_121a") {
		t.Fatalf("requirement on local topology: %v", err)
	}
	if _, err := BuildLaunchSpec(profile, config.spark, defaultOptions(2)); err != nil {
		t.Fatalf("requirement on Spark topology: %v", err)
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
			modelPath := writeCheckpoint(t, test.algorithm, "TestArchitecture", 64)
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
	modelPath := writeCheckpoint(t, "", "TestArchitecture", 64)
	decision, err := ResolveMTPMoEBackend(modelPath, "bfloat16")
	if err != nil || decision.Quantization != "bf16" || decision.Backend != "triton" {
		t.Fatalf("unquantized decision: %+v %v", decision, err)
	}
	if _, err := ResolveMTPMoEBackend(writeCheckpoint(t, "UNKNOWN_FP6", "TestArchitecture", 64), "bfloat16"); err == nil ||
		!strings.Contains(err.Error(), "cannot determine MTP") {
		t.Fatalf("unknown quantization error: %v", err)
	}
}

func TestMTPBackendDerivesFromRealConfigsAndMatchesManifestAssertions(t *testing.T) {
	config := loadTestConfig(t)
	for name, want := range map[string][2]string{
		glmProfile:            {"bf16", "triton"},
		glmSparkProfile:       {"nvfp4", "b12x"},
		glmFlashProfile:       {"mxfp8", "triton"},
		glmFlashSparkProfile:  {"nvfp4", "b12x"},
		qwenProfile:           {"nvfp4", "b12x"},
		deepseekProfile:       {"mxfp4", "b12x"},
		deepseekVisionProfile: {"mxfp4", "b12x"},
	} {
		options := defaultOptions(0)
		options.Speculator = stringPointerTest("mtp")
		spec, err := BuildLaunchSpec(config.profiles[name], config.local, options)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		mtp := spec.Metadata["mtp_moe"].(map[string]any)
		if mtp["quantization"] != want[0] || mtp["backend"] != want[1] {
			t.Errorf("%s MTP decision: %+v", name, mtp)
		}
		if got := speculativeConfigValue(t, spec)["moe_backend"]; got != want[1] {
			t.Errorf("%s speculative config backend %v", name, got)
		}
	}
}

func TestExplicitCheckpointMustAgreeWithRepositoryFacts(t *testing.T) {
	config := loadTestConfig(t)
	glmFlash := config.profiles[glmFlashProfile]
	options := defaultOptions(2)
	options.ModelPath = stringPointerTest(writeCheckpoint(t, "W4A16_NVFP4", "Glm5NextForConditionalGeneration", 64))
	_, err := BuildLaunchSpec(glmFlash, config.local, options)
	if err == nil || !strings.Contains(err.Error(), "MTP MoE quantization mismatch") {
		t.Fatalf("manifest assertion mismatch: %v", err)
	}
	options.ModelPath = stringPointerTest(writeCheckpoint(t, "MXFP8", "OtherArchitecture", 64))
	_, err = BuildLaunchSpec(glmFlash, config.local, options)
	if err == nil || !strings.Contains(err.Error(), "declares") {
		t.Fatalf("architecture mismatch: %v", err)
	}
	options.ModelPath = stringPointerTest(writeCheckpoint(t, "MXFP8", "Glm5NextForConditionalGeneration", 64))
	spec, err := BuildLaunchSpec(glmFlash, config.local, options)
	if err != nil {
		t.Fatal(err)
	}
	if len(spec.Downloads) != 0 || spec.CheckpointPath == nil {
		t.Fatalf("local checkpoint launch requested Hub downloads: %+v", spec.Downloads)
	}
	memory := spec.Metadata["memory"].(map[string]any)
	if got := memory["estimated_sharded_weight_bytes_per_rank"]; got != int64(24000000000) {
		t.Fatalf("estimate must use the local checkpoint size: %v", got)
	}
}

func TestKVCacheAndCapacityAreCLIOverrides(t *testing.T) {
	config := loadTestConfig(t)
	options := defaultOptions(2)
	options.KVCacheDType = "bfloat16"
	options.GPUMemoryUtilization = floatPointer(0.87)
	options.MaxModelLen = stringPointerTest("32768")
	options.MaxNumSeqs = intPointer(17)
	options.KVCacheMemoryBytes = stringPointerTest("40000000000")
	spec, err := BuildLaunchSpec(config.profiles[glmFlashProfile], config.local, options)
	if err != nil {
		t.Fatal(err)
	}
	for flag, want := range map[string]string{
		"--kv-cache-dtype":         "bfloat16",
		"--gpu-memory-utilization": "0.87",
		"--max-model-len":          "32768",
		"--max-num-seqs":           "17",
		"--kv-cache-memory-bytes":  "40000000000",
	} {
		if got := optionValue(t, spec.VLLMArgv, flag); got != want {
			t.Errorf("%s: got %q, want %q", flag, got, want)
		}
	}
	if spec.Metadata["memory"].(map[string]any)["kv_cache_allocation"] != "explicit" {
		t.Fatalf("explicit KV allocation not recorded: %+v", spec.Metadata["memory"])
	}
}

func TestCUDAGraphCaptureSizesCoverMTPVerificationAndMixedShapes(t *testing.T) {
	config := loadTestConfig(t)
	spec, err := BuildLaunchSpec(config.profiles[glmProfile], config.local, defaultOptions(8))
	if err != nil {
		t.Fatal(err)
	}
	var compilation map[string]any
	if err := json.Unmarshal([]byte(optionValue(t, spec.VLLMArgv, "--compilation-config")), &compilation); err != nil {
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
	if compilation["cudagraph_mode"] != "FULL_AND_PIECEWISE" {
		t.Fatalf("manifest compilation settings lost: %v", compilation)
	}
}

func TestCUDAGraphCaptureSizesFollowSequenceCountsWithoutMTP(t *testing.T) {
	config := loadTestConfig(t)
	options := defaultOptions(8)
	options.Speculator = stringPointerTest("none")
	spec, err := BuildLaunchSpec(config.profiles[glmProfile], config.local, options)
	if err != nil {
		t.Fatal(err)
	}
	var compilation map[string]any
	if err := json.Unmarshal([]byte(optionValue(t, spec.VLLMArgv, "--compilation-config")), &compilation); err != nil {
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
	got := cudagraphCaptureSizes(8, 4096, 4, 2)
	for _, shape := range wantMTPShapes {
		if !slices.Contains(got, shape) {
			t.Errorf("capture sizes %v omit MTP verification shape %d", got, shape)
		}
	}
}

func TestMemoryEstimateUsesPostShardingWeightSize(t *testing.T) {
	config := loadTestConfig(t)
	spec, err := BuildLaunchSpec(config.profiles[glmFlashProfile], config.local, defaultOptions(2))
	if err != nil {
		t.Fatal(err)
	}
	memory := spec.Metadata["memory"].(map[string]any)
	if got := memory["estimated_sharded_weight_bytes_per_rank"]; got != int64(99021165756) {
		t.Fatalf("estimated sharded bytes: got %v", got)
	}
	if memory["kv_cache_allocation"] != "vllm_runtime_profile" || memory["estimate_is_advisory"] != true ||
		memory["estimate_source"] != "safetensors file sizes" && !strings.HasSuffix(memory["estimate_source"].(string), "model.safetensors.index.json") {
		t.Fatalf("unexpected memory policy: %+v", memory)
	}
}

func TestSparkSelectsFirstNNodesAndBuildsPersistentNamedContainers(t *testing.T) {
	config := loadTestConfig(t)
	spec, err := BuildLaunchSpec(config.profiles[qwenProfile], config.spark, defaultOptions(2))
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
		if !reflect.DeepEqual(node.DockerArgv[:2], []string{"docker", "run"}) || slices.Contains(node.DockerArgv, "--rm") {
			t.Errorf("rank %d docker argv must create a persistent container: %v", rank, node.DockerArgv)
		}
		if optionValue(t, node.DockerArgv, "--name") != "vllm-fleet-Qwen3.8-Flash-Next-NVFP4-tp2" || !slices.Contains(node.DockerArgv, "--detach") {
			t.Errorf("rank %d container naming: %v", rank, node.DockerArgv)
		}
		for _, label := range []string{"lil.model=local-inference-lab/Qwen3.8-Flash-Next-NVFP4", "lil.tp=2", "lil.rank=" + strconv.Itoa(rank), "lil.manifest_commit=" + testCommit} {
			if !slices.Contains(node.DockerArgv, label) {
				t.Errorf("rank %d docker argv is missing label %s", rank, label)
			}
		}
		if optionValue(t, node.VLLMArgv, "--nnodes") != "2" ||
			optionValue(t, node.VLLMArgv, "--node-rank") != strconv.Itoa(rank) {
			t.Errorf("rank %d distributed argv: %v", rank, node.VLLMArgv)
		}
		if slices.Contains(node.VLLMArgv, "--headless") != (rank > 0) {
			t.Errorf("rank %d headless mismatch", rank)
		}
	}
}

func TestGLMFlashSparkPolicyMatchesRDMAContract(t *testing.T) {
	config := loadTestConfig(t)
	spec, err := BuildLaunchSpec(config.profiles[glmFlashSparkProfile], config.spark, defaultOptions(0))
	if err != nil {
		t.Fatal(err)
	}
	for flag, want := range map[string]string{
		"--gpu-memory-utilization": "0.95",
		"--max-model-len":          "auto",
		"--max-num-seqs":           "8",
		"--max-num-batched-tokens": "4096",
		"--tensor-parallel-size":   "2",
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
	options := defaultOptions(2)
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

func TestSparkTargetResolution(t *testing.T) {
	config := loadTestConfig(t)
	target, err := ResolveSparkTarget(config.profiles[qwenProfile], config.spark, intPointer(1), nil)
	if err != nil {
		t.Fatal(err)
	}
	if target.TPSize != 1 || len(target.Nodes) != 1 || target.Nodes[0].SSHHost != "tachyon" ||
		target.ContainerName != "vllm-fleet-Qwen3.8-Flash-Next-NVFP4-tp1" {
		t.Fatalf("unexpected target: %+v", target)
	}
	target, err = ResolveSparkTarget(config.profiles[qwenProfile], config.spark, nil, nil)
	if err != nil || target.TPSize != 2 {
		t.Fatalf("default Spark target must span the cluster: %+v %v", target, err)
	}
}

func TestSparkModelSyncRequiresExplicitModelPath(t *testing.T) {
	config := loadTestConfig(t)
	err := SyncSparkModel(context.Background(), LaunchSpec{
		Topology:    config.spark,
		ModelSource: "local-inference-lab/Qwen3.8-Flash-Next-NVFP4",
	})
	if err == nil || !strings.Contains(err.Error(), "--model-path") {
		t.Fatalf("sync error: %v", err)
	}
}

func TestManagedVLLMArgumentsCannotBeForwarded(t *testing.T) {
	config := loadTestConfig(t)
	for _, arguments := range [][]string{
		{"--port", "9000"},
		{"--tensor_parallel_size=4"},
		{"-tp", "4"},
		{"--compilation-config.cudagraph-mode=NONE"},
		{"--headless"},
		{"--revision", testCommit},
	} {
		options := defaultOptions(2)
		options.ExtraVLLMArgs = arguments
		_, err := BuildLaunchSpec(config.profiles[qwenProfile], config.local, options)
		if err == nil || !strings.Contains(err.Error(), "launcher-managed") {
			t.Errorf("arguments %v: %v", arguments, err)
		}
	}
}

func TestHubLaunchDownloadsTargetAndDraftRepositories(t *testing.T) {
	config := loadTestConfig(t)
	options := defaultOptions(2)
	options.Speculator = stringPointerTest("dflash")
	spec, err := BuildLaunchSpec(config.profiles[glmFlashProfile], config.local, options)
	if err != nil {
		t.Fatal(err)
	}
	want := []RepositoryDownload{
		{Repository: "local-inference-lab/GLM-5.3-Flash-NVFP4"},
		{Repository: "local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8"},
	}
	if !reflect.DeepEqual(spec.Downloads, want) {
		t.Fatalf("downloads: got %v, want %v", spec.Downloads, want)
	}
	if slices.Contains(spec.VLLMArgv, "--revision") || spec.CheckpointPath != nil {
		t.Fatalf("Hub launch is unexpectedly revision-bound: %+v", spec)
	}
	speculative := speculativeConfigValue(t, spec)
	if speculative["method"] != "dflash" || speculative["num_speculative_tokens"] != float64(7) || speculative["revision"] != nil {
		t.Fatalf("DFlash configuration: %+v", speculative)
	}
	if spec.Metadata["mtp_moe"] != nil {
		t.Fatalf("DFlash launch carries an MTP decision: %+v", spec.Metadata)
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
		Topology:         Topology{Local: &LocalTopology{Python: filepath.Join(bin, "python")}},
		Downloads:        []RepositoryDownload{{Repository: "local-inference-lab/Test-Model", Revision: testCommit}},
		UnsetEnvironment: []string{"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"},
	}
	if err := SyncHuggingFaceCache(context.Background(), spec); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.TrimSpace(string(data)); got != "download local-inference-lab/Test-Model --revision "+testCommit {
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
		Downloads: []RepositoryDownload{{Repository: "local-inference-lab/Test-Model"}},
	}
	if err := SyncHuggingFaceCache(context.Background(), spec); err != nil {
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
		Downloads: []RepositoryDownload{{Repository: "local-inference-lab/Test-Model"}},
	}
	err := SyncHuggingFaceCache(context.Background(), spec)
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
	host := sparkHostHFArgv(topology, "/opt/vllm/.venv/bin/hf", "download", "org/model")
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
	spec, err := BuildLaunchSpec(config.profiles[glmProfile], config.local, defaultOptions(8))
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
	if rendered["model"] != glmProfile || rendered["manifest_commit"] != testCommit || rendered["family"] != "glm" ||
		rendered["tensor_parallel_size"] != float64(8) || rendered["topology_kind"] != "local" {
		t.Fatalf("unexpected JSON render: %v", rendered)
	}
	metadata := rendered["metadata"].(map[string]any)
	facts := metadata["checkpoint_facts"].(map[string]any)
	if facts["attention_heads"] != float64(64) || facts["weight_bytes"] != float64(464823066832) {
		t.Fatalf("checkpoint facts missing from JSON: %v", facts)
	}
	sparkSpec, err := BuildLaunchSpec(config.profiles[glmFlashSparkProfile], config.spark, defaultOptions(2))
	if err != nil {
		t.Fatal(err)
	}
	sparkShell := ShellRender(sparkSpec)
	if !strings.Contains(sparkShell, "ServerAliveInterval=15") || strings.Count(sparkShell, "ssh ") != 2 {
		t.Fatalf("unexpected Spark shell render:\n%s", sparkShell)
	}
}

func TestCatalogManifestNamesUpstreamAndPinsRemoteCode(t *testing.T) {
	config := loadTestConfig(t)
	deepseek := config.profiles[deepseekProfile]
	if deepseek.Model != "deepseek-ai/DeepSeek-V4-Flash-0731" || deepseek.Revision != deepseekRevision ||
		deepseek.Family != "deepseek-v4" || !deepseek.Serving.TrustRemoteCode {
		t.Fatalf("catalog identity: %+v", deepseek)
	}
	unpinned := "schema_version: 1\nkind: model\nmodel: deepseek-ai/DeepSeek-V4-Flash\ndescription: x\nserving: {served_model_name: x, trust_remote_code: true}\n"
	if _, err := LoadCatalogModelProfile(config.families, []byte(unpinned), "Unpinned", testCommit, "lil.yaml"); err == nil ||
		!strings.Contains(err.Error(), "pin a revision") {
		t.Fatalf("unpinned remote-code entry: %v", err)
	}
	safe := "schema_version: 1\nkind: model\nmodel: someone/Model\ndescription: x\nserving: {served_model_name: x}\n"
	profile, err := LoadCatalogModelProfile(config.families, []byte(safe), "Safe", testCommit, "lil.yaml")
	if err != nil || profile.Model != "someone/Model" || profile.Revision != "" {
		t.Fatalf("unpinned entry without remote code: %+v %v", profile, err)
	}
	missing := "schema_version: 1\nkind: model\ndescription: x\nserving: {served_model_name: x}\n"
	if _, err := LoadCatalogModelProfile(config.families, []byte(missing), "Missing", testCommit, "lil.yaml"); err == nil ||
		!strings.Contains(err.Error(), "model must name") {
		t.Fatalf("catalog entry without model: %v", err)
	}
	restated := "schema_version: 1\nkind: model\nmodel: a/b\nrevision: " + testCommit + "\ndescription: x\nserving: {served_model_name: x}\n"
	if _, err := loadManifest(t, config.families, "Owned", restated); err == nil || !strings.Contains(err.Error(), "must not set model") {
		t.Fatalf("repository manifest with model: %v", err)
	}
}

func TestDeepSeekDSparkLaunchMatchesTheShellLauncher(t *testing.T) {
	config := loadTestConfig(t)
	spec, err := BuildLaunchSpec(config.profiles[deepseekProfile], config.local, defaultOptions(2))
	if err != nil {
		t.Fatal(err)
	}
	argv := spec.VLLMArgv
	if argv[3] != "serve" || argv[4] != "deepseek-ai/DeepSeek-V4-Flash-0731" || argv[5] != "--revision" || argv[6] != deepseekRevision {
		t.Fatalf("model and revision must lead the serve argv: %v", argv[:8])
	}
	for flag, want := range map[string]string{
		"--served-model-name":               "DeepSeek-V4-Flash-0731",
		"--tokenizer-mode":                  "deepseek_v4",
		"--tool-call-parser":                "deepseek_v4",
		"--reasoning-parser":                "deepseek_v4",
		"--kv-cache-dtype":                  "fp8",
		"--block-size":                      "256",
		"--load-format":                     "instanttensor",
		"--attention-backend":               "B12X",
		"--moe-backend":                     "b12x",
		"--linear-backend":                  "b12x",
		"--max-num-seqs":                    "16",
		"--max-num-batched-tokens":          "8192",
		"--max-model-len":                   "auto",
		"--prefix-cache-retention-interval": "4096",
	} {
		if got := optionValue(t, argv, flag); got != want {
			t.Errorf("%s: got %q, want %q", flag, got, want)
		}
	}
	for _, flag := range []string{
		"--trust-remote-code", "--async-scheduling", "--no-scheduler-reserve-full-isl",
		"--enable-chunked-prefill", "--enable-prefix-caching", "--enable-auto-tool-choice",
		"--enable-prompt-tokens-details", "--enable-force-include-usage",
		"--enable-request-id-headers", "--enable-flashinfer-autotune",
		"--default-chat-template-kwargs.thinking=true",
		"--default-chat-template-kwargs.reasoning_effort=high",
	} {
		if !slices.Contains(argv, flag) {
			t.Errorf("argv is missing %s", flag)
		}
	}
	if slices.Contains(argv, "--quantization") || slices.Contains(argv, "--max-cudagraph-capture-size") {
		t.Errorf("argv carries a flag the checkpoint or capture list already implies: %v", argv)
	}
	speculative := speculativeConfigValue(t, spec)
	if speculative["method"] != "dspark" || speculative["model"] != "deepseek-ai/DeepSeek-V4-Flash-0731" ||
		speculative["revision"] != deepseekRevision || speculative["num_speculative_tokens"] != float64(7) ||
		speculative["draft_sample_method"] != "probabilistic" || speculative["rejection_sample_method"] != "standard" ||
		speculative["enable_adaptive_verification"] != nil {
		t.Fatalf("DSpark speculative config: %+v", speculative)
	}
	var compilation map[string]any
	if err := json.Unmarshal([]byte(optionValue(t, argv, "--compilation-config")), &compilation); err != nil {
		t.Fatal(err)
	}
	sizes := compilation["cudagraph_capture_sizes"].([]any)
	if sizes[len(sizes)-1] != float64(128) {
		t.Fatalf("DSpark capture sizes must stop at max_num_seqs times K+1: %v", sizes)
	}
	for name, want := range map[string]string{
		"VLLM_B12X_MOE_FP4_FORCE_A16":              "0",
		"VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1",
		"VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD":   "1024",
	} {
		if got := spec.RuntimeEnvironment[name]; got != want {
			t.Errorf("%s: got %q, want %q", name, got, want)
		}
	}
	if !reflect.DeepEqual(spec.Downloads, []RepositoryDownload{{Repository: "deepseek-ai/DeepSeek-V4-Flash-0731", Revision: deepseekRevision}}) {
		t.Fatalf("pinned download: %+v", spec.Downloads)
	}
	if spec.Metadata["speculator"] != "dspark" || spec.Metadata["revision"] != deepseekRevision {
		t.Fatalf("metadata: %+v", spec.Metadata)
	}
}

func TestDeepSeekOverridesFollowTheSpeculator(t *testing.T) {
	config := loadTestConfig(t)
	for _, test := range []struct {
		speculator string
		seqs       string
		lastSize   float64
		hasSpec    bool
	}{
		{"dspark", "16", 128, true},
		{"mtp", "64", 512, true},
		{"none", "64", 128, false},
	} {
		options := defaultOptions(2)
		options.Speculator = stringPointerTest(test.speculator)
		spec, err := BuildLaunchSpec(config.profiles[deepseekProfile], config.local, options)
		if err != nil {
			t.Fatalf("%s: %v", test.speculator, err)
		}
		if got := optionValue(t, spec.VLLMArgv, "--max-num-seqs"); got != test.seqs {
			t.Errorf("%s: max-num-seqs %s, want %s", test.speculator, got, test.seqs)
		}
		var compilation map[string]any
		if err := json.Unmarshal([]byte(optionValue(t, spec.VLLMArgv, "--compilation-config")), &compilation); err != nil {
			t.Fatal(err)
		}
		sizes := compilation["cudagraph_capture_sizes"].([]any)
		if sizes[len(sizes)-1] != test.lastSize {
			t.Errorf("%s: largest capture size %v, want %v", test.speculator, sizes[len(sizes)-1], test.lastSize)
		}
		if slices.Contains(spec.VLLMArgv, "--speculative-config") != test.hasSpec {
			t.Errorf("%s: speculative config presence %v", test.speculator, !test.hasSpec)
		}
		if test.speculator == "mtp" {
			speculative := speculativeConfigValue(t, spec)
			if speculative["moe_backend"] != "b12x" || speculative["revision"] != nil || speculative["rejection_sample_method"] != "standard" {
				t.Errorf("MTP speculative config: %+v", speculative)
			}
		}
	}
	options := defaultOptions(2)
	options.AdaptiveVerification = true
	spec, err := BuildLaunchSpec(config.profiles[deepseekProfile], config.local, options)
	if err != nil {
		t.Fatal(err)
	}
	if speculativeConfigValue(t, spec)["enable_adaptive_verification"] != true {
		t.Fatalf("adaptive verification switch was not applied")
	}
	options.Speculator = stringPointerTest("mtp")
	if _, err := BuildLaunchSpec(config.profiles[deepseekProfile], config.local, options); err == nil || !strings.Contains(err.Error(), "only for DSpark") {
		t.Fatalf("adaptive verification on MTP: %v", err)
	}
}

func TestBlockFP8CheckpointDerivesExpertQuantization(t *testing.T) {
	config := map[string]any{
		"architectures":       []any{"DeepseekV4ForCausalLM"},
		"num_attention_heads": float64(64),
		"expert_dtype":        "fp8",
		"quantization_config": map[string]any{"quant_method": "fp8", "weight_block_size": []any{float64(128), float64(128)}},
	}
	_, err := MTPMoEBackendFromConfig(config, "bfloat16")
	if err == nil || !strings.Contains(err.Error(), "moe_backend") {
		t.Fatalf("FP8 experts must require an explicit backend: %v", err)
	}
	delete(config, "expert_dtype")
	decision, err := MTPMoEBackendFromConfig(config, "bfloat16")
	if err != nil || decision.Quantization != "mxfp4" || decision.Backend != "b12x" {
		t.Fatalf("default fp4 experts: %+v %v", decision, err)
	}
	backend := "triton"
	facts := &CheckpointFacts{Source: "test", Config: map[string]any{
		"quantization_config": map[string]any{"quant_method": "fp8"}, "expert_dtype": "fp8",
	}}
	decision, err = mtpBackendDecision(MTPPolicy{MoEBackend: &backend}, facts, "bfloat16")
	if err != nil || decision.Backend != "triton" || decision.Quantization != "fp8" {
		t.Fatalf("declared backend: %+v %v", decision, err)
	}
}
