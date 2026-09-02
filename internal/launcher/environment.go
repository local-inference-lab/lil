// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package launcher

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// derivedEnvironment names variables the launcher computes from topology facts
// or command-line options. Neither topology YAML, manifests, nor --env may set
// them.
var derivedEnvironment = stringSet(
	"PYTHONPATH", "LD_LIBRARY_PATH", "CUDA_HOME", "TRITON_PTXAS_PATH", "CUTE_DSL_ARCH",
	"CUDA_VISIBLE_DEVICES", "B12X_POLICY_MODE", "NCCL_DEBUG",
	"NCCL_IB_GID_INDEX", "NCCL_IB_MERGE_NICS", "HF_HUB_OFFLINE",
	"TRANSFORMERS_OFFLINE", "GLOO_SOCKET_IFNAME", "MN_IF_NAME", "NCCL_IB_HCA",
	"NCCL_SOCKET_IFNAME", "OMPI_MCA_btl_tcp_if_include", "TP_SOCKET_IFNAME",
	"UCX_NET_DEVICES", "VLLM_HOST_IP",
)

var commonHostEnvironment = map[string]string{
	"INSTANTTENSOR_BACKEND":        "BUFFERED",
	"OMP_NUM_THREADS":              "16",
	"PYTORCH_CUDA_ALLOC_CONF":      "expandable_segments:True",
	"SAFETENSORS_FAST_GPU":         "1",
	"VLLM_PLUGINS":                 "",
	"VLLM_USE_AOT_COMPILE":         "1",
	"VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
	"VLLM_USE_FLASHINFER_SAMPLER":  "1",
	"VLLM_USE_MEGA_AOT_ARTIFACT":   "1",
	"VLLM_USE_STANDALONE_COMPILE":  "1",
	"VLLM_USE_V2_MODEL_RUNNER":     "1",
	"VLLM_WORKER_MULTIPROC_METHOD": "spawn",
}

// DefaultLocalEnvironment is the host tuning discovery records for a
// single-host PCIe topology.
func DefaultLocalEnvironment() map[string]string {
	environment := copyStringMap(commonHostEnvironment)
	for name, value := range map[string]string{
		"CUDA_DEVICE_ORDER":                             "PCI_BUS_ID",
		"NCCL_IB_DISABLE":                               "1",
		"NCCL_P2P_LEVEL":                                "SYS",
		"NCCL_PROTO":                                    "LL,LL128,Simple",
		"VLLM_ENABLE_PCIE_ALLREDUCE":                    "1",
		"VLLM_PCIE_ALLREDUCE_BACKEND":                   "b12x",
		"VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE":          "64KB",
		"VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE": "84KB",
	} {
		environment[name] = value
	}
	return environment
}

// DefaultSparkEnvironment is the host tuning discovery records for a
// one-GPU-per-node RDMA topology.
func DefaultSparkEnvironment() map[string]string {
	environment := copyStringMap(commonHostEnvironment)
	for name, value := range map[string]string{
		"INSTANTTENSOR_BUFFER_SIZE":    "1342177280",
		"INSTANTTENSOR_CONCURRENCY":    "1",
		"INSTANTTENSOR_IO_DEPTH":       "3",
		"NCCL_IB_DISABLE":              "0",
		"NCCL_IB_SUBNET_AWARE_ROUTING": "1",
		"NCCL_IGNORE_CPU_AFFINITY":     "1",
		"NCCL_NET_PLUGIN":              "none",
		"VLLM_ENABLE_PCIE_ALLREDUCE":   "0",
	} {
		environment[name] = value
	}
	return environment
}

func pythonPath(paths []string, inherit bool) string {
	return searchPath("PYTHONPATH", paths, inherit)
}

func searchPath(variable string, paths []string, inherit bool) string {
	entries := append([]string{}, paths...)
	if inherit {
		entries = append(entries, filepath.SplitList(os.Getenv(variable))...)
	}
	seen := map[string]bool{}
	unique := entries[:0]
	for _, entry := range entries {
		if entry != "" && !seen[entry] {
			seen[entry] = true
			unique = append(unique, entry)
		}
	}
	return strings.Join(unique, string(os.PathListSeparator))
}

// runtimeEnvironment layers, from lowest to highest precedence: topology host
// tuning, launcher-derived values, resolved manifest environment, the PLE
// command-line switch, and explicit --env overrides. needsNVRTC adds the
// discovered NVRTC library directory for launches that run Humming kernels.
func runtimeEnvironment(topology Topology, settings LaunchSettings, options LaunchOptions, needsNVRTC bool) (map[string]string, []string, error) {
	environment := copyStringMap(topology.Environment())
	runtimeRoots := []string{topology.RepoRoot(), topology.B12XRoot()}
	if topology.Spark != nil {
		runtimeRoots = []string{topology.Spark.RuntimeRepoRoot, topology.Spark.RuntimeB12XRoot}
	}
	environment["PYTHONPATH"] = pythonPath(runtimeRoots, topology.Local != nil)
	if needsNVRTC {
		if topology.NVRTCLibraryDir() == "" {
			return nil, nil, fmt.Errorf(
				"the Humming MoE backend needs the CUDA 13 NVRTC builtins, but topology %q recorded no nvrtc_library_dir; rediscover it",
				topology.Name(),
			)
		}
		environment["LD_LIBRARY_PATH"] = searchPath("LD_LIBRARY_PATH", []string{topology.NVRTCLibraryDir()}, topology.Local != nil)
	}
	environment["CUDA_HOME"] = topology.CUDAHome()
	environment["TRITON_PTXAS_PATH"] = filepath.Join(topology.CUDAHome(), "bin", "ptxas")
	environment["CUTE_DSL_ARCH"] = topology.CuteDSLArch()
	environment["B12X_POLICY_MODE"] = options.B12XPolicyMode
	var unset []string
	if topology.Local != nil {
		unset = []string{"CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"}
	} else {
		spark := topology.Spark
		environment["CUDA_VISIBLE_DEVICES"] = strconv.Itoa(spark.DeviceID)
		environment["NCCL_DEBUG"] = spark.NCCLDebug
		environment["NCCL_IB_GID_INDEX"] = strconv.Itoa(spark.NCCLIBGIDIndex)
		environment["NCCL_IB_MERGE_NICS"] = strconv.Itoa(boolInt(spark.NCCLIBMergeNICs))
		unset = []string{"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"}
	}
	for name, value := range settings.Environment {
		environment[name] = value
	}
	if options.PLECPUOffload != nil {
		if _, ok := environment["VLLM_PLE_CPU_OFFLOAD"]; !ok {
			return nil, nil, fmt.Errorf("--ple-cpu-offload is valid only when the resolved profile manages VLLM_PLE_CPU_OFFLOAD")
		}
		if *options.PLECPUOffload {
			environment["VLLM_PLE_CPU_OFFLOAD"] = "1"
		} else {
			environment["VLLM_PLE_CPU_OFFLOAD"] = "0"
		}
	}
	seenOverrides := map[string]bool{}
	for _, override := range options.EnvironmentOverrides {
		if !environmentName.MatchString(override.Name) {
			return nil, nil, fmt.Errorf("invalid environment variable name: %q", override.Name)
		}
		if seenOverrides[override.Name] {
			return nil, nil, fmt.Errorf("environment override %q is duplicated", override.Name)
		}
		seenOverrides[override.Name] = true
		if derivedEnvironment[override.Name] || contains(unset, override.Name) {
			return nil, nil, fmt.Errorf("environment variable %q is derived by the launcher and cannot be overridden with --env", override.Name)
		}
		environment[override.Name] = override.Value
	}
	return environment, unset, nil
}

func sparkNodeEnvironment(base map[string]string, node SparkNode) map[string]string {
	environment := copyStringMap(base)
	for name, value := range map[string]string{
		"VLLM_HOST_IP": node.Address, "MN_IF_NAME": node.EthernetInterface,
		"UCX_NET_DEVICES": node.EthernetInterface, "NCCL_SOCKET_IFNAME": node.EthernetInterface,
		"NCCL_IB_HCA":                 strings.Join(node.RDMAInterfaces, ","),
		"OMPI_MCA_btl_tcp_if_include": node.EthernetInterface,
		"GLOO_SOCKET_IFNAME":          node.EthernetInterface, "TP_SOCKET_IFNAME": node.EthernetInterface,
	} {
		environment[name] = value
	}
	return environment
}

func stringSet(values ...string) map[string]bool {
	result := make(map[string]bool, len(values))
	for _, value := range values {
		result[value] = true
	}
	return result
}

func contains(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func copyStringMap(source map[string]string) map[string]string {
	result := make(map[string]string, len(source))
	for key, value := range source {
		result[key] = value
	}
	return result
}

func boolInt(value bool) int {
	if value {
		return 1
	}
	return 0
}
