// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package launcher

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"
)

func TestSharedDefaultsApplyWithoutFamilyAndRespectOverrides(t *testing.T) {
	config := loadTestConfig(t)
	for _, family := range []string{"", "family: glm\n", "family: deepseek-v4\n"} {
		manifest := "schema_version: 1\nkind: model\ndescription: Shared defaults\n" + family + "serving:\n  served_model_name: defaults\n"
		profile, err := loadManifest(t, config.families, "Defaults", manifest)
		if err != nil {
			t.Fatal(err)
		}
		profile.Facts = config.profiles[glmProfile].Facts
		spec, err := BuildLaunchSpec(profile, config.local, defaultOptions(8))
		if err != nil {
			t.Fatal(err)
		}
		for flag, want := range map[string]string{
			"--prefill-compute-share": "auto", "--prefill-compute-half-life": "smooth",
			"--max-parallel-prefills": "auto", "--prefill-policy": "decode-aware",
			"--decode-refill-target": "auto", "--generation-config": "vllm",
		} {
			if got := optionValue(t, spec.VLLMArgv, flag); got != want {
				t.Errorf("%s: %s = %q; want %q", family, flag, got, want)
			}
		}
		for _, flag := range []string{
			"--async-scheduling", "--no-scheduler-reserve-full-isl", "--enable-prefix-caching",
			"--enable-chunked-prefill", "--enable-prompt-tokens-details", "--enable-force-include-usage",
			"--enable-request-id-headers", "--compilation-config",
		} {
			if !slices.Contains(spec.VLLMArgv, flag) {
				t.Errorf("%s: missing shared flag %s", family, flag)
			}
		}
	}
	profile, err := loadManifest(t, config.families, "Overrides", `schema_version: 1
kind: model
description: Explicit serving overrides
family: glm
serving:
  served_model_name: overrides
  async_scheduling: false
  scheduler_reserve_full_isl: true
  prefix_caching: false
  chunked_prefill: false
  max_parallel_prefills: 1
  prefill_policy: round-robin
  prefill_compute_share: null
  prefill_compute_half_life: null
  generation_config: auto
  prompt_tokens_details: false
  force_include_usage: false
  request_id_headers: false
compilation: null
environment:
  VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD: "512"
`)
	if err != nil {
		t.Fatal(err)
	}
	profile.Facts = config.profiles[glmProfile].Facts
	options := defaultOptions(8)
	options.EnvironmentOverrides = []EnvironmentOverride{{"VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD", "256"}}
	spec, err := BuildLaunchSpec(profile, config.local, options)
	if err != nil {
		t.Fatal(err)
	}
	for _, flag := range []string{"--no-async-scheduling", "--scheduler-reserve-full-isl", "--no-enable-prefix-caching", "--no-enable-chunked-prefill"} {
		if !slices.Contains(spec.VLLMArgv, flag) {
			t.Errorf("missing explicit override %s", flag)
		}
	}
	for _, flag := range []string{"--prefill-compute-share", "--prefill-compute-half-life", "--compilation-config", "--enable-prompt-tokens-details", "--enable-force-include-usage", "--enable-request-id-headers"} {
		if slices.Contains(spec.VLLMArgv, flag) {
			t.Errorf("cleared setting still emitted: %s", flag)
		}
	}
	if optionValue(t, spec.VLLMArgv, "--generation-config") != "auto" || spec.RuntimeEnvironment["VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD"] != "256" {
		t.Fatal("explicit manifest or CLI value lost to shared defaults")
	}
}

func TestDeepSeek41FlashProductionProfile(t *testing.T) {
	config := loadTestConfig(t)
	profile := config.profiles[deepseek41Profile]
	spec, err := BuildLaunchSpec(profile, config.local, defaultOptions(4))
	if err != nil {
		t.Fatal(err)
	}
	for flag, want := range map[string]string{
		"--tokenizer-mode": "deepseek_v41", "--reasoning-parser": "deepseek_v41", "--tool-call-parser": "deepseek_v41",
		"--kv-cache-dtype": "auto", "--block-size": "256", "--attention-backend": "B12X",
		"--load-format": "instanttensor", "--safetensors-load-strategy": "lazy", "--jit-monitor-mode": "error",
		"--gpu-memory-utilization": "0.98", "--max-num-seqs": "8", "--max-num-batched-tokens": "4096",
		"--revision": "fb2764a5cf321eaa5070ca8f9e892818f477c16d", "--tensor-parallel-size": "4",
	} {
		if got := optionValue(t, spec.VLLMArgv, flag); got != want {
			t.Errorf("%s = %q; want %q", flag, got, want)
		}
	}
	draft := speculativeConfigValue(t, spec)
	for key, want := range map[string]any{
		"method": "dspark", "num_speculative_tokens": float64(7), "draft_tensor_parallel_size": float64(4),
		"draft_sample_method": "greedy", "rejection_sample_method": "standard", "attention_backend": "B12X",
		"enable_adaptive_verification": true, "adaptive_verification_cost_scale": float64(1),
	} {
		if draft[key] != want {
			t.Errorf("DSpark %s = %v; want %v", key, draft[key], want)
		}
	}
	var engram, compilation map[string]any
	if err := json.Unmarshal([]byte(optionValue(t, spec.VLLMArgv, "--engram-config")), &engram); err != nil {
		t.Fatal(err)
	}
	if len(engram) != 3 || engram["cpu_offload"] != false || engram["table_memory"] != "disk" || engram["disk_resident_scales"] != false {
		t.Fatalf("Engram config: %v", engram)
	}
	if err := json.Unmarshal([]byte(optionValue(t, spec.VLLMArgv, "--compilation-config")), &compilation); err != nil {
		t.Fatal(err)
	}
	sizes := compilation["cudagraph_capture_sizes"].([]any)
	if compilation["cudagraph_mode"] != "FULL_AND_PIECEWISE" || sizes[len(sizes)-1] != float64(64) {
		t.Fatalf("graph configuration: %v", compilation)
	}
	options := defaultOptions(4)
	tokens, adaptive := 3, false
	options.SpeculativeTokens = &tokens
	options.AdaptiveVerification = &adaptive
	spec, err = BuildLaunchSpec(profile, config.local, options)
	if err != nil {
		t.Fatal(err)
	}
	draft = speculativeConfigValue(t, spec)
	if draft["num_speculative_tokens"] != float64(3) || draft["enable_adaptive_verification"] == true {
		t.Fatalf("DSpark overrides: %v", draft)
	}
}

func TestServingPolicyRejectsConflictingSchedulerSettings(t *testing.T) {
	config := loadTestConfig(t)
	for _, test := range []struct{ serving, want string }{
		{"prefill_compute_share: 0", "prefill_compute_share"},
		{"prefill_compute_half_life: .inf", "prefill_compute_half_life"},
		{"max_parallel_prefills: 1.5", "max_parallel_prefills"},
		{"prefill_compute_share: 0.5", "prefill_compute_half_life requires"},
		{"prefill_schedule_interval: 2", "cannot be combined"},
		{"max_parallel_prefills: 1", "requires max_parallel_prefills"},
		{"max_parallel_prefills: 9", "cannot exceed resolved max_num_seqs"},
		{"decode_refill_target: 9", "cannot exceed resolved max_num_seqs"},
		{"chunked_prefill: false", "requires chunked_prefill"},
		{"prefill_policy: round-robin\n  decode_refill_target: 2", "requires prefill_policy"},
	} {
		t.Run(test.serving, func(t *testing.T) {
			profile, err := loadManifest(t, config.families, "Scheduler", "schema_version: 1\nkind: model\ndescription: Scheduler constraints\nserving:\n  served_model_name: scheduler\n  "+test.serving+"\n")
			if err == nil {
				profile.Facts = config.profiles[glmProfile].Facts
				_, err = BuildLaunchSpec(profile, config.local, defaultOptions(8))
			}
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("got %v; want %q", err, test.want)
			}
		})
	}
	for _, field := range []string{"projection_tp: false", "table_memory: invalid", "table_memory: ram\n  disk_resident_scales: true"} {
		_, err := loadManifest(t, config.families, "Engram", "schema_version: 1\nkind: model\ndescription: Engram constraints\nserving: {served_model_name: engram}\nengram:\n  "+field+"\n")
		if err == nil {
			t.Fatalf("accepted invalid Engram policy %q", field)
		}
	}
}
