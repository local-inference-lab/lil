// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package main

import (
	"context"
	"slices"
	"strings"
	"testing"
)

func TestDataParallelLaunchFlags(t *testing.T) {
	configureTestTopologies(t)
	for _, command := range []string{"render", "check", "run"} {
		for _, test := range []struct {
			name    string
			args    []string
			wantDP  int
			devices []int
			wantErr string
		}{
			{"default", nil, 1, []int{8, 9}, ""},
			{"explicit one", []string{"--dp=1"}, 1, []int{8, 9}, ""},
			{"four replicas", []string{"--dp=4"}, 4, []int{0, 1, 2, 3, 4, 5, 6, 7}, ""},
			{"explicit devices", []string{"--dp=2", "--devices=4,5,6,7"}, 2, []int{4, 5, 6, 7}, ""},
			{"zero", []string{"--dp=0"}, 0, nil, "DP size must be positive"},
			{"negative", []string{"--dp=-1"}, 0, nil, "DP size must be positive"},
		} {
			t.Run(command+"/"+test.name, func(t *testing.T) {
				fs := newFlagSet(command)
				var values launchFlags
				configureLaunchFlags(fs, &values, command == "run")
				if fs.Lookup("dp").DefValue != "1" {
					t.Fatalf("--dp default is %q, want 1", fs.Lookup("dp").DefValue)
				}
				values.modelsConfig = testModelsConfig()
				args := append([]string{"Qwen3.8-Flash-Next-NVFP4", "--config=local-test", "--tp=2"}, test.args...)
				if err := fs.Parse(args); err != nil {
					t.Fatal(err)
				}
				launch, err := buildSpec(context.Background(), fs, values)
				if test.wantErr != "" {
					if err == nil || !strings.Contains(err.Error(), test.wantErr) {
						t.Fatalf("error %v does not mention %q", err, test.wantErr)
					}
					return
				}
				if err != nil {
					t.Fatal(err)
				}
				if launch.DPSize != test.wantDP || launch.TPSize != 2 || !slices.Equal(launch.DeviceIDs, test.devices) {
					t.Fatalf("TP=%d DP=%d devices=%v", launch.TPSize, launch.DPSize, launch.DeviceIDs)
				}
			})
		}
	}
}
