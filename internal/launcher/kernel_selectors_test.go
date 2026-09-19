// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package launcher

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"
)

func TestKernelSelectorsRender(t *testing.T) {
	config := loadTestConfig(t)
	profile, err := loadManifest(t, config.families, "Selectors", `schema_version: 1
kind: model
description: Kernel selector test
serving:
  served_model_name: selectors
family: glm
cache:
  indexer_kv_dtype: mxfp4
  recurrent_checkpoint_policy: request_boundaries
kernels:
  gdn_prefill: b12x
  kda_prefill: triton
  mamba: flashinfer
`)
	if err != nil {
		t.Fatal(err)
	}
	profile.Facts = config.profiles[glmProfile].Facts
	for _, topology := range []Topology{config.local, config.spark} {
		spec, err := BuildLaunchSpec(profile, topology, defaultOptions(2))
		if err != nil {
			t.Fatal(err)
		}
		for flag, want := range map[string]string{
			"--gdn-prefill-backend": "b12x", "--kda-prefill-backend": "triton",
			"--mamba-backend": "flashinfer", "--attention-backend": "B12X",
			"--recurrent-checkpoint-policy": "request_boundaries",
		} {
			if got := optionValue(t, spec.VLLMArgv, flag); got != want {
				t.Errorf("%s = %q, want %q", flag, got, want)
			}
		}
		var attention map[string]string
		if err := json.Unmarshal([]byte(optionValue(t, spec.VLLMArgv, "--attention-config")), &attention); err != nil {
			t.Fatal(err)
		}
		if len(attention) != 1 || attention["indexer_kv_dtype"] != "mxfp4" {
			t.Fatalf("attention config: %v", attention)
		}
	}
}

func TestKernelSelectorsInheritanceAndValidation(t *testing.T) {
	families := []byte(`schema_version: 1
families:
  selectors:
    cache:
      indexer_kv_dtype: fp8
    kernels:
      gdn_prefill: b12x
      kda_prefill: b12x
      mamba: triton
`)
	profile, err := loadManifest(t, families, "Selectors", `schema_version: 1
kind: model
description: Kernel selector test
serving:
  served_model_name: selectors
family: selectors
cache:
  indexer_kv_dtype: bf16
kernels:
  gdn_prefill: null
  kda_prefill: triton
`)
	if err != nil {
		t.Fatal(err)
	}
	k := profile.Kernels
	if k.GDNPrefill != nil || k.KDAPrefill == nil || *k.KDAPrefill != "triton" || k.Mamba == nil || *k.Mamba != "triton" || profile.Cache.IndexerKVDType == nil || *profile.Cache.IndexerKVDType != "bf16" {
		t.Fatalf("resolved selectors: %+v", k)
	}
	for _, key := range []string{"gdn_prefill", "kda_prefill", "mamba", "indexer_kv_dtype", "recurrent_checkpoint_policy"} {
		section := "kernels"
		if key == "indexer_kv_dtype" || key == "recurrent_checkpoint_policy" {
			section = "cache"
		}
		for _, value := range []string{"false", "123", "[]", `""`} {
			_, err := loadManifest(t, families, "Invalid", "schema_version: 1\nkind: model\ndescription: Kernel selector test\nserving:\n  served_model_name: selectors\n"+section+":\n  "+key+": "+value+"\n")
			if err == nil || !strings.Contains(err.Error(), key) {
				t.Errorf("%s: %s accepted: %v", key, value, err)
			}
		}
	}
}

func TestKernelSelectorsOmittedAndManaged(t *testing.T) {
	config := loadTestConfig(t)
	spec, err := BuildLaunchSpec(config.profiles[glmProfile], config.local, defaultOptions(2))
	if err != nil {
		t.Fatal(err)
	}
	for _, flag := range []string{"--gdn-prefill-backend", "--kda-prefill-backend", "--mamba-backend", "--attention-config", "--recurrent-checkpoint-policy"} {
		if slices.Contains(spec.VLLMArgv, flag) {
			t.Errorf("unset selector emitted %s", flag)
		}
		if err := validateExtraArgs([]string{flag + "=auto"}); err == nil {
			t.Errorf("managed option accepted: %s", flag)
		}
	}
	if err := validateExtraArgs([]string{"--attention-config.indexer_kv_dtype=fp8"}); err == nil {
		t.Fatal("managed attention config accepted")
	}
}
