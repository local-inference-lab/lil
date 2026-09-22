// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package launcher

import (
	"slices"
	"strconv"
	"testing"
)

func TestB12XPagedAttentionBlockSize(t *testing.T) {
	config := loadTestConfig(t)
	for _, topology := range []Topology{config.local, config.spark} {
		for _, test := range []struct {
			name      string
			attention string
			blockSize int
			want      int
		}{
			{"omitted", "B12X", 0, 128},
			{"incompatible16", "B12X", 16, 128},
			{"incompatible32", "B12X", 32, 128},
			{"incompatible256", "B12X", 256, 128},
			{"supported64", "B12X", 64, 64},
			{"supported128", "B12X", 128, 128},
			{"otherBackend", "FLASH_ATTN", 16, 16},
			{"otherBackendOmitted", "FLASH_ATTN", 0, 0},
			{"automaticBackend", "", 16, 16},
			{"automaticBackendOmitted", "", 0, 0},
		} {
			t.Run(topology.Kind+"/"+test.name, func(t *testing.T) {
				profile := config.profiles[qwenProfile]
				profile.Kernels.Attention = nil
				if test.attention != "" {
					profile.Kernels.Attention = stringPointer(test.attention)
				}
				profile.Cache.BlockSize = nil
				if test.blockSize != 0 {
					profile.Cache.BlockSize = intPointer(test.blockSize)
				}
				options := defaultOptions(2)
				options.Speculator = stringPointer("none")
				spec, err := BuildLaunchSpec(profile, topology, options)
				if err != nil {
					t.Fatal(err)
				}
				if test.want == 0 {
					if slices.Contains(spec.VLLMArgv, "--block-size") || spec.Model.Cache.BlockSize != nil {
						t.Fatalf("unspecified block size emitted: %v", spec.VLLMArgv)
					}
				} else {
					if got := optionValue(t, spec.VLLMArgv, "--block-size"); got != strconv.Itoa(test.want) {
						t.Fatalf("block size = %s, want %d", got, test.want)
					}
					if spec.Model.Cache.BlockSize == nil || *spec.Model.Cache.BlockSize != test.want {
						t.Fatalf("resolved profile block size: %v", spec.Model.Cache.BlockSize)
					}
					for _, node := range spec.SparkNodes {
						if got := optionValue(t, node.VLLMArgv, "--block-size"); got != strconv.Itoa(test.want) {
							t.Errorf("rank %d block size = %s, want %d", node.Rank, got, test.want)
						}
					}
				}
				if profile.Cache.BlockSize != nil && *profile.Cache.BlockSize != test.blockSize {
					t.Fatal("launch resolution mutated the input profile's block size")
				}
			})
		}
	}
}

func TestB12XPreservesMLABlockSize(t *testing.T) {
	config := loadTestConfig(t)
	for _, name := range []string{glmProfile, glmFlashProfile, deepseekProfile, deepseek41Profile} {
		for _, blockSize := range []int{0, 256} {
			t.Run(name+"/"+strconv.Itoa(blockSize), func(t *testing.T) {
				profile := config.profiles[name]
				profile.Cache.BlockSize = nil
				if blockSize != 0 {
					profile.Cache.BlockSize = intPointer(blockSize)
				}
				options := defaultOptions(8)
				options.Speculator = stringPointer("none")
				spec, err := BuildLaunchSpec(profile, config.local, options)
				if err != nil {
					t.Fatal(err)
				}
				if blockSize == 0 {
					if slices.Contains(spec.VLLMArgv, "--block-size") {
						t.Fatalf("paged attention default applied to MLA: %v", spec.VLLMArgv)
					}
				} else if got := optionValue(t, spec.VLLMArgv, "--block-size"); got != strconv.Itoa(blockSize) {
					t.Fatalf("MLA block size = %s, want %d", got, blockSize)
				}
			})
		}
	}
}

func TestB12XDraftAttentionBlockSize(t *testing.T) {
	config := loadTestConfig(t)
	for _, test := range []struct {
		method string
		tokens int
		want   string
	}{
		{"mtp", 3, "128"},
		{"mtp", 0, "16"},
		{"dspark", 3, "128"},
		{"dspark", 0, "16"},
		{"none", 3, "16"},
	} {
		t.Run(test.method+"/"+strconv.Itoa(test.tokens), func(t *testing.T) {
			profile := config.profiles[qwenProfile]
			mtp := *profile.Speculators.MTP
			mtp.Attention = stringPointer("B12X")
			profile.Speculators.MTP = &mtp
			profile.Speculators.DSpark = &DSparkPolicy{Tokens: 3, Attention: stringPointer("B12X")}
			options := defaultOptions(2)
			options.Speculator = &test.method
			options.SpeculativeTokens = &test.tokens
			spec, err := BuildLaunchSpec(profile, config.local, options)
			if err != nil {
				t.Fatal(err)
			}
			if got := optionValue(t, spec.VLLMArgv, "--block-size"); got != test.want {
				t.Fatalf("block size = %s, want %s", got, test.want)
			}
		})
	}
}
