// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package launcher

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestPolicySectionsInheritanceAndRendering(t *testing.T) {
	config := loadTestConfig(t)
	families := []byte(`schema_version: 1
families:
  policy:
    precision:
      dtype: float16
      quantization: modelopt_mixed
    cache:
      kv_cache_dtype: bfloat16
      indexer_kv_dtype: fp8
      block_size: 16
      mamba_cache_mode: align
      recurrent_checkpoint_policy: aligned
      prefix_cache_retention_interval: 4096
    loading:
      load_format: safetensors
      loader_extra_config:
        inherited: true
        replaced: 1
`)
	profile, err := loadManifest(t, families, "Policy", `schema_version: 1
kind: model
family: policy
description: Explicit precision, cache, and loading policy
serving:
  served_model_name: policy
precision:
  quantization: null
cache:
  indexer_kv_dtype: null
  prefix_cache_retention_interval: 0
loading:
  loader_extra_config:
    replaced: 2
`)
	if err != nil {
		t.Fatal(err)
	}
	if profile.Precision.Quantization != nil || profile.Cache.IndexerKVDType != nil {
		t.Fatal("null did not clear inherited selectors")
	}
	profile.Facts = config.profiles[glmProfile].Facts
	for _, topology := range []Topology{config.local, config.spark} {
		options := defaultOptions(2)
		dtype := "fp8_e4m3"
		options.KVCacheDType = &dtype
		spec, err := BuildLaunchSpec(profile, topology, options)
		if err != nil {
			t.Fatal(err)
		}
		for flag, want := range map[string]string{
			"--dtype": "float16", "--kv-cache-dtype": dtype, "--block-size": "16",
			"--mamba-cache-mode": "align", "--recurrent-checkpoint-policy": "aligned",
			"--prefix-cache-retention-interval": "0", "--load-format": "safetensors",
		} {
			if got := optionValue(t, spec.VLLMArgv, flag); got != want {
				t.Errorf("%s = %q; want %q", flag, got, want)
			}
		}
		var extra map[string]any
		if err := json.Unmarshal([]byte(optionValue(t, spec.VLLMArgv, "--model-loader-extra-config")), &extra); err != nil {
			t.Fatal(err)
		}
		if len(extra) != 2 || extra["inherited"] != true || extra["replaced"] != float64(2) {
			t.Fatalf("loader inheritance: %v", extra)
		}
	}
}

func TestPolicySectionDefaultsAndNull(t *testing.T) {
	families := []byte(`schema_version: 1
families:
  policy:
    precision:
      dtype: float16
    cache:
      kv_cache_dtype: bfloat16
      prefix_cache_retention_interval: 4096
    loading:
      load_format: safetensors
      loader_extra_config:
        enabled: true
`)
	for _, body := range []string{"", "family: policy\nprecision: null\ncache: null\nloading: null\n"} {
		profile, err := loadManifest(t, families, "Defaults", "schema_version: 1\nkind: model\ndescription: Default policy\nserving:\n  served_model_name: defaults\n"+body)
		if err != nil {
			t.Fatal(err)
		}
		if profile.Precision.DType != "bfloat16" || profile.Cache.KVCacheDType != "fp8" || profile.Cache.PrefixCacheRetentionInterval != nil || profile.Loading.LoadFormat != "" || profile.Loading.LoaderExtraConfig != nil {
			t.Fatalf("defaults: %+v", profile)
		}
	}
}

func TestPolicySectionsRejectMisplacedKeys(t *testing.T) {
	config := loadTestConfig(t)
	for section, keys := range map[string][]string{
		"kernels":   {"dtype", "quantization", "kv_cache_dtype", "indexer_kv_dtype", "block_size", "mamba_cache_mode", "recurrent_checkpoint_policy", "load_format", "loader_extra_config"},
		"serving":   {"prefix_cache_retention_interval"},
		"precision": {"attention"}, "cache": {"dtype"}, "loading": {"moe"},
	} {
		for _, key := range keys {
			manifest := "schema_version: 1\nkind: model\ndescription: Invalid placement\nserving:\n  served_model_name: invalid\n"
			if section != "serving" {
				manifest += section + ":\n"
			}
			manifest += "  " + key + ": null\n"
			_, err := loadManifest(t, config.families, "Invalid", manifest)
			if err == nil || !strings.Contains(err.Error(), section) || !strings.Contains(err.Error(), key) {
				t.Errorf("%s.%s: %v", section, key, err)
			}
		}
	}
}
