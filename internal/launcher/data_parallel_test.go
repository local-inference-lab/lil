// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package launcher

import (
	"encoding/json"
	"reflect"
	"slices"
	"strings"
	"testing"
)

func TestDataParallelDefaultsToOne(t *testing.T) {
	config := loadTestConfig(t)
	for _, topology := range []Topology{config.local, config.spark} {
		for _, explicit := range []*int{nil, intPointer(1)} {
			options := defaultOptions(0)
			options.DPSize = explicit
			launch, err := BuildLaunchSpec(config.profiles[qwenProfile], topology, options)
			if err != nil {
				t.Fatal(err)
			}
			if launch.DPSize != 1 || optionValue(t, launch.VLLMArgv, "--data-parallel-size") != "1" {
				t.Fatalf("%s: omitted or explicit DP=1 changed: %+v", topology.Name(), launch)
			}
			if topology.Local != nil && len(launch.DeviceIDs) != launch.TPSize {
				t.Fatalf("default DP allocated extra devices: %v", launch.DeviceIDs)
			}
		}
	}
}

func TestDataParallelAllocatesReplicasAndPreservesPerReplicaSettings(t *testing.T) {
	config := loadTestConfig(t)
	profile := config.profiles[qwenProfile]
	baseline, err := BuildLaunchSpec(profile, config.local, defaultOptions(2))
	if err != nil {
		t.Fatal(err)
	}
	for _, tp := range []int{0, 2} {
		options := defaultOptions(tp)
		options.DPSize = intPointer(4)
		launch, err := BuildLaunchSpec(profile, config.local, options)
		if err != nil {
			t.Fatal(err)
		}
		if launch.TPSize != 2 || launch.DPSize != 4 || !slices.Equal(launch.DeviceIDs, []int{0, 1, 2, 3, 4, 5, 6, 7}) {
			t.Fatalf("unexpected replica allocation: TP=%d DP=%d devices=%v", launch.TPSize, launch.DPSize, launch.DeviceIDs)
		}
		if optionValue(t, launch.VLLMArgv, "--data-parallel-size") != "4" ||
			optionValue(t, launch.VLLMArgv, "--device-ids") != "0,1,2,3,4,5,6,7" {
			t.Fatalf("vLLM arguments do not allocate four TP=2 replicas: %v", launch.VLLMArgv)
		}
		for _, key := range []string{"memory", "capacity", "applied_overrides"} {
			if !reflect.DeepEqual(launch.Metadata[key], baseline.Metadata[key]) {
				t.Errorf("DP changed per-replica %s: %v vs %v", key, launch.Metadata[key], baseline.Metadata[key])
			}
		}
		for _, flag := range []string{"--tensor-parallel-size", "--max-num-seqs", "--max-num-batched-tokens", "--speculative-config", "--compilation-config"} {
			if optionValue(t, launch.VLLMArgv, flag) != optionValue(t, baseline.VLLMArgv, flag) {
				t.Errorf("DP changed per-replica flag %s", flag)
			}
		}
		encoded, err := JSONRender(launch)
		if err != nil {
			t.Fatal(err)
		}
		var rendered struct {
			TP      int   `json:"tensor_parallel_size"`
			DP      int   `json:"data_parallel_size"`
			Devices []int `json:"device_ids"`
		}
		if err := json.Unmarshal(encoded, &rendered); err != nil {
			t.Fatal(err)
		}
		if rendered.TP != 2 || rendered.DP != 4 || !slices.Equal(rendered.Devices, launch.DeviceIDs) {
			t.Fatalf("rendered replica allocation: %+v", rendered)
		}
	}
}

func TestDataParallelPreservesExplicitDeviceOrder(t *testing.T) {
	config := loadTestConfig(t)
	options := defaultOptions(2)
	options.DPSize = intPointer(4)
	options.DeviceIDs = []int{8, 9, 10, 11, 4, 5, 6, 7}
	launch, err := BuildLaunchSpec(config.profiles[qwenProfile], config.local, options)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(launch.DeviceIDs, options.DeviceIDs) || optionValue(t, launch.VLLMArgv, "--device-ids") != "8,9,10,11,4,5,6,7" {
		t.Fatalf("explicit replica device groups changed: %v", launch.DeviceIDs)
	}
}

func TestDataParallelOmitsOnlyInheritedComputeSharing(t *testing.T) {
	config := loadTestConfig(t)
	profile := config.profiles[qwenProfile]
	options := defaultOptions(2)
	options.DPSize = intPointer(4)
	launch, err := BuildLaunchSpec(profile, config.local, options)
	if err != nil {
		t.Fatal(err)
	}
	for _, flag := range []string{"--prefill-compute-share", "--prefill-compute-half-life"} {
		if slices.Contains(launch.VLLMArgv, flag) {
			t.Errorf("DP launch contains unsupported default %s", flag)
		}
	}
	if launch.Model.Serving.PrefillComputeShare != nil || launch.Model.Serving.PrefillComputeHalfLife != nil {
		t.Fatal("resolved profile retains omitted compute-sharing defaults")
	}
	if profile.Serving.PrefillComputeShare == nil || profile.Serving.PrefillComputeHalfLife == nil {
		t.Fatal("building a DP launch mutated the input profile")
	}
	for flag, want := range map[string]string{
		"--max-parallel-prefills": "auto", "--prefill-policy": "decode-aware", "--decode-refill-target": "auto",
	} {
		if got := optionValue(t, launch.VLLMArgv, flag); got != want {
			t.Errorf("%s = %q, want %q", flag, got, want)
		}
	}
	single, err := BuildLaunchSpec(profile, config.local, defaultOptions(2))
	if err != nil {
		t.Fatal(err)
	}
	if optionValue(t, single.VLLMArgv, "--prefill-compute-share") != "auto" ||
		optionValue(t, single.VLLMArgv, "--prefill-compute-half-life") != "smooth" {
		t.Fatal("DP=1 lost the compute-sharing defaults")
	}
}

func TestDataParallelRejectsExplicitComputeSharing(t *testing.T) {
	config := loadTestConfig(t)
	for _, test := range []struct {
		name    string
		serving string
	}{
		{"automatic", "prefill_compute_share: auto"},
		{"numeric", "prefill_compute_share: 0.5\n  prefill_compute_half_life: null"},
		{"half-life", "prefill_compute_half_life: responsive"},
	} {
		for _, layer := range []string{"manifest", "family"} {
			t.Run(test.name+"/"+layer, func(t *testing.T) {
				bases := config.families
				manifest := "schema_version: 1\nkind: model\ndescription: Explicit compute sharing\nserving:\n  served_model_name: sharing\n  " + test.serving + "\n"
				if layer == "family" {
					bases = append(slices.Clone(bases), []byte("\n  sharing:\n    serving:\n      "+strings.ReplaceAll(test.serving, "\n  ", "\n      ")+"\n")...)
					manifest = "schema_version: 1\nkind: model\nfamily: sharing\ndescription: Explicit compute sharing\nserving:\n  served_model_name: sharing\n"
				}
				profile, err := loadManifest(t, bases, "Sharing", manifest)
				if err != nil {
					t.Fatal(err)
				}
				profile.Facts = config.profiles[qwenProfile].Facts
				if _, err := BuildLaunchSpec(profile, config.local, defaultOptions(2)); err != nil {
					t.Fatalf("explicit DP=1 policy should remain valid: %v", err)
				}
				options := defaultOptions(2)
				options.DPSize = intPointer(4)
				if _, err := BuildLaunchSpec(profile, config.local, options); err == nil || !strings.Contains(err.Error(), "explicitly configures prefill compute sharing") {
					t.Fatalf("explicit policy was silently changed: %v", err)
				}
			})
		}
	}
}

func TestDataParallelAcceptsExplicitlyDisabledComputeSharing(t *testing.T) {
	config := loadTestConfig(t)
	profile, err := loadManifest(t, config.families, "SharingDisabled", `schema_version: 1
kind: model
description: Compute sharing disabled
serving:
  served_model_name: sharing-disabled
  prefill_compute_share: null
  prefill_compute_half_life: null
`)
	if err != nil {
		t.Fatal(err)
	}
	profile.Facts = config.profiles[qwenProfile].Facts
	options := defaultOptions(2)
	options.DPSize = intPointer(4)
	launch, err := BuildLaunchSpec(profile, config.local, options)
	if err != nil {
		t.Fatal(err)
	}
	if slices.Contains(launch.VLLMArgv, "--prefill-compute-share") || slices.Contains(launch.VLLMArgv, "--prefill-compute-half-life") {
		t.Fatal("explicitly disabled compute sharing was enabled")
	}
}

func TestInvalidDataParallelAllocationFailsBeforeLaunch(t *testing.T) {
	config := loadTestConfig(t)
	for _, test := range []struct {
		name     string
		dp       int
		devices  []int
		topology Topology
		want     string
	}{
		{"zero", 0, nil, config.local, "DP size must be positive"},
		{"negative", -1, nil, config.local, "DP size must be positive"},
		{"too many replicas", 7, nil, config.local, "exceeds"},
		{"overflow", int(^uint(0) >> 1), nil, config.local, "exceeds"},
		{"only one replica's devices", 4, []int{0, 1}, config.local, "requires 8 device IDs"},
		{"duplicate devices", 2, []int{0, 1, 0, 1}, config.local, "unique"},
		{"negative device", 2, []int{0, 1, 2, -1}, config.local, "non-negative"},
		{"Spark", 2, nil, config.spark, "Spark/RDMA requires DP=1"},
	} {
		t.Run(test.name, func(t *testing.T) {
			options := defaultOptions(2)
			options.DPSize = intPointer(test.dp)
			options.DeviceIDs = test.devices
			_, err := BuildLaunchSpec(config.profiles[qwenProfile], test.topology, options)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error %v does not mention %q", err, test.want)
			}
		})
	}
}

func TestDataParallelArgumentsCannotOverrideLauncherAllocation(t *testing.T) {
	for _, flag := range []string{
		"--data-parallel-size=4", "--data_parallel_size=4", "-dp=4",
		"--data-parallel-size-local=1", "-dpl=1", "--data-parallel-rank=0", "-dpn=0",
		"--data-parallel-start-rank=1", "-dpr=1", "--data-parallel-backend=ray", "-dpb=ray",
		"--data-parallel-external-lb", "-dpe", "--no-data-parallel-external-lb",
		"--data-parallel-hybrid-lb", "-dph", "--data-parallel-multi-port-external-lb", "-dpm",
	} {
		if err := validateExtraArgs([]string{flag}); err == nil || !strings.Contains(err.Error(), "launcher-managed") {
			t.Errorf("%s: %v", flag, err)
		}
	}
}
