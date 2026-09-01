// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package launcher

import (
	"reflect"
	"testing"
)

func TestParseNVIDIATopologyBuildsConnectedAndCumulativePools(t *testing.T) {
	data := []byte("\n\x1b[4mGPU0 GPU1 GPU2 GPU3 CPU Affinity\x1b[0m\n" + `
GPU0     X   PIX  SYS  SYS  0-31
GPU1    PIX   X   SYS  SYS  0-31
GPU2    SYS  SYS   X   PXB  32-63
GPU3    SYS  SYS  PXB   X   32-63
`)
	want := [][]int{{0, 1}, {2, 3}, {0, 1, 2, 3}}
	if got := parseNVIDIATopology(data, []int{0, 1, 2, 3}); !reflect.DeepEqual(got, want) {
		t.Fatalf("device pools: got %v, want %v", got, want)
	}
}

func TestSelectSparkNetworksUsesFirstCommonSubnetAndAllCommonHCAs(t *testing.T) {
	hosts := []string{"tachyon", "luxon"}
	observations := map[string][]remoteNetworkObservation{
		"tachyon": {
			{SSHHost: "tachyon", Address: "10.200.0.1", Network: "10.200.0.0/30", Ethernet: "eth-a", RDMADevice: "roce-a"},
			{SSHHost: "tachyon", Address: "10.200.0.5", Network: "10.200.0.4/30", Ethernet: "eth-b", RDMADevice: "roce-b"},
		},
		"luxon": {
			{SSHHost: "luxon", Address: "10.200.0.2", Network: "10.200.0.0/30", Ethernet: "eth-c", RDMADevice: "roce-c"},
			{SSHHost: "luxon", Address: "10.200.0.6", Network: "10.200.0.4/30", Ethernet: "eth-d", RDMADevice: "roce-d"},
		},
	}
	want := []SparkNode{
		{SSHHost: "tachyon", Address: "10.200.0.1", EthernetInterface: "eth-a", RDMAInterfaces: []string{"roce-a", "roce-b"}},
		{SSHHost: "luxon", Address: "10.200.0.2", EthernetInterface: "eth-c", RDMAInterfaces: []string{"roce-c", "roce-d"}},
	}
	got, err := selectSparkNetworks(hosts, observations)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Spark nodes: got %+v, want %+v", got, want)
	}
}

func TestMarshalTopologyRoundTrips(t *testing.T) {
	topology := Topology{
		Kind: "local", GPUMemoryUtilization: 0.91,
		Local: &LocalTopology{
			Name: "pulsar", Host: "0.0.0.0", Port: 8000,
			DeviceMemoryBytes: 1024,
			RepoRoot:          "/srv/vllm", Python: "/srv/vllm/.venv/bin/python",
			B12XRoot: "/srv/b12x", CUDAHome: "/opt/cuda",
			CuteDSLArch: "sm_120a", DevicePools: [][]int{{0, 1}},
		},
	}
	data, err := MarshalTopology(topology)
	if err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadTopology(data, "generated.yaml", ".")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(loaded, topology) {
		t.Fatalf("round trip: got %+v, want %+v", loaded, topology)
	}
}
