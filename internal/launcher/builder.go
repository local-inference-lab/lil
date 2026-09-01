// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package launcher

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

var environmentName = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)
var containerNameSanitizer = regexp.MustCompile(`[^A-Za-z0-9_.-]`)

var managedVLLMFlags = stringSet(
	"--async-scheduling", "--attention-backend", "--block-size",
	"--compilation-config", "--dcp-comm-backend",
	"--decode-context-parallel-size", "--disable-custom-all-reduce", "--dtype",
	"--enable-auto-tool-choice", "--enable-chunked-prefill",
	"--enable-prefix-caching", "--gdn-decode-kernel", "--generation-config",
	"--gpu-memory-utilization", "--hf-overrides", "--host", "--kv-cache-dtype",
	"--kv-cache-memory-bytes", "--limit-mm-per-prompt", "--linear-backend",
	"--load-format", "--long-prefill-token-threshold", "--mamba-cache-mode",
	"--master-addr", "--master-port", "--max-model-len",
	"--max-num-batched-tokens", "--max-num-seqs", "--mm-encoder-tp-mode",
	"--mm-processor-cache-gb", "--model-loader-extra-config", "--moe-backend",
	"--nnodes", "--node-rank", "--no-enable-flashinfer-autotune",
	"--pipeline-parallel-size", "--port", "--profiler-config", "--quantization",
	"--reasoning-parser", "--revision", "--served-model-name", "--speculative-config",
	"--tensor-parallel-size", "--tool-call-parser", "--trust-remote-code",
	"--device-ids", "--distributed-executor-backend", "--headless",
)

var managedVLLMNegatedFlags = stringSet(
	"--enable-flashinfer-autotune", "--no-async-scheduling",
	"--no-disable-custom-all-reduce", "--no-enable-auto-tool-choice",
	"--no-enable-chunked-prefill", "--no-enable-prefix-caching",
	"--no-trust-remote-code", "--no-headless",
)

var vllmFlagAliases = map[string]string{
	"-cc": "--compilation-config", "-dcp": "--decode-context-parallel-size",
	"-n": "--nnodes", "-r": "--node-rank", "-pp": "--pipeline-parallel-size",
	"-tp": "--tensor-parallel-size",
}

var sparkNodeEnvironment = stringSet(
	"GLOO_SOCKET_IFNAME", "MN_IF_NAME", "NCCL_IB_DISABLE", "NCCL_IB_HCA",
	"NCCL_SOCKET_IFNAME", "OMPI_MCA_btl_tcp_if_include", "TP_SOCKET_IFNAME",
	"UCX_NET_DEVICES", "VLLM_HOST_IP",
)

func stringSet(values ...string) map[string]bool {
	result := make(map[string]bool, len(values))
	for _, value := range values {
		result[value] = true
	}
	return result
}

func compactJSON(value any) (string, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return "", fmt.Errorf("cannot encode generated JSON: %w", err)
	}
	return string(encoded), nil
}

func appendOption(argv *[]string, flag string, value *string) {
	if value != nil {
		*argv = append(*argv, flag, *value)
	}
}

func stringPointer(value string) *string { return &value }

func intPointerString(value *int) *string {
	if value == nil {
		return nil
	}
	result := strconv.Itoa(*value)
	return &result
}

func floatPointerString(value *float64) *string {
	if value == nil {
		return nil
	}
	result := strconv.FormatFloat(*value, 'g', -1, 64)
	return &result
}

func pythonPath(paths []string, inherit bool) string {
	entries := append([]string{}, paths...)
	if inherit {
		entries = append(entries, filepath.SplitList(os.Getenv("PYTHONPATH"))...)
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

func resolveTPSize(profile ModelProfile, topology Topology, requested *int) (int, error) {
	policy, err := profile.Launch.Topology(topology.Kind)
	if err != nil {
		return 0, err
	}
	configured := policy.DefaultTPSize
	if policy.DefaultTPAll {
		if topology.Spark != nil {
			configured = len(topology.Spark.Nodes)
		} else {
			for _, pool := range topology.Local.DevicePools {
				if len(pool) > configured {
					configured = len(pool)
				}
			}
		}
	}
	tpSize := configured
	if requested != nil {
		tpSize = *requested
	}
	if tpSize <= 0 {
		return 0, fmt.Errorf("tensor parallel size must be positive")
	}
	if topology.Spark != nil && tpSize > len(topology.Spark.Nodes) {
		return 0, fmt.Errorf(
			"Spark/RDMA topology %q has %d one-GPU nodes, but TP=%d was requested",
			topology.Name(), len(topology.Spark.Nodes), tpSize,
		)
	}
	if profile.AttentionHeads%tpSize != 0 {
		return 0, fmt.Errorf(
			"model %q has %d attention heads, which is not divisible by TP=%d",
			profile.Name, profile.AttentionHeads, tpSize,
		)
	}
	return tpSize, nil
}

func resolveDevices(topology Topology, options LaunchOptions, tpSize int) ([]int, error) {
	if topology.Spark != nil {
		if options.DeviceIDs != nil {
			return nil, fmt.Errorf("--devices is valid only for local topologies; Spark/RDMA device selection belongs to its YAML topology")
		}
		return nil, nil
	}
	devices := options.DeviceIDs
	var err error
	if devices == nil {
		devices, err = topology.Local.SelectDevices(tpSize)
	}
	if err != nil {
		return nil, err
	}
	if len(devices) != tpSize {
		return nil, fmt.Errorf("TP=%d requires %d device IDs; got %d", tpSize, tpSize, len(devices))
	}
	seen := map[int]bool{}
	for _, device := range devices {
		if device < 0 || seen[device] {
			return nil, fmt.Errorf("device IDs must be unique non-negative integers")
		}
		seen[device] = true
	}
	return append([]int(nil), devices...), nil
}

func resolveCapacity(settings LaunchSettings, topology Topology, options LaunchOptions) Capacity {
	capacity := settings.Capacity
	utilization := topology.MemoryUtilization()
	capacity.GPUMemoryUtilization = &utilization
	if options.GPUMemoryUtilization != nil {
		capacity.GPUMemoryUtilization = options.GPUMemoryUtilization
	}
	if options.KVCacheMemoryBytes != nil {
		if strings.EqualFold(*options.KVCacheMemoryBytes, "auto") {
			capacity.KVCacheMemoryBytes = nil
		} else {
			capacity.KVCacheMemoryBytes = options.KVCacheMemoryBytes
		}
	}
	if options.MaxModelLen != nil {
		capacity.MaxModelLen = *options.MaxModelLen
	}
	if options.MaxNumSeqs != nil {
		capacity.MaxNumSeqs = *options.MaxNumSeqs
	}
	if options.MaxNumBatchedTokens != nil {
		capacity.MaxNumBatchedTokens = *options.MaxNumBatchedTokens
	}
	return capacity
}

func validateOptions(options LaunchOptions) error {
	if options.Port != nil {
		if err := validatePort(*options.Port, "port"); err != nil {
			return err
		}
	}
	if options.KVCacheDType == "" {
		return fmt.Errorf("KV cache dtype must not be empty")
	}
	if options.GPUMemoryUtilization != nil && (*options.GPUMemoryUtilization <= 0 || *options.GPUMemoryUtilization > 1) {
		return fmt.Errorf("GPU memory utilization must be in (0, 1]")
	}
	positive := map[string]*int{
		"max-num-seqs":           options.MaxNumSeqs,
		"max-num-batched-tokens": options.MaxNumBatchedTokens,
		"DCP size":               &options.DCPSize, "adaptive window": &options.AdaptiveWindow,
	}
	for name, value := range positive {
		if value != nil && *value <= 0 {
			return fmt.Errorf("%s must be positive", name)
		}
	}
	if options.SpeculativeTokens != nil && *options.SpeculativeTokens < 0 {
		return fmt.Errorf("speculative token count must be non-negative")
	}
	if !stringSet("auto", "heuristic-only", "preplanned-only")[options.B12XPolicyMode] {
		return fmt.Errorf("B12X policy mode must be auto, heuristic-only, or preplanned-only")
	}
	return nil
}

func runtimeEnvironment(profile ModelProfile, topology Topology, settings LaunchSettings, options LaunchOptions) (map[string]string, []string, error) {
	runtimeRoots := []string{topology.RepoRoot(), topology.B12XRoot()}
	inherit := topology.Local != nil
	if topology.Spark != nil {
		runtimeRoots = []string{topology.Spark.RuntimeRepoRoot, topology.Spark.RuntimeB12XRoot}
	}
	environment := map[string]string{
		"PYTHONPATH":              pythonPath(runtimeRoots, inherit),
		"CUDA_HOME":               topology.CUDAHome(),
		"TRITON_PTXAS_PATH":       filepath.Join(topology.CUDAHome(), "bin", "ptxas"),
		"CUTE_DSL_ARCH":           topology.CuteDSLArch(),
		"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
		"SAFETENSORS_FAST_GPU":    "1", "OMP_NUM_THREADS": "16",
		"VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_PLUGINS": "",
		"VLLM_USE_AOT_COMPILE": "1", "VLLM_USE_STANDALONE_COMPILE": "1",
		"VLLM_USE_MEGA_AOT_ARTIFACT": "1", "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
		"VLLM_USE_FLASHINFER_SAMPLER": "1", "VLLM_USE_V2_MODEL_RUNNER": "1",
		"B12X_POLICY_MODE": options.B12XPolicyMode, "INSTANTTENSOR_BACKEND": "BUFFERED",
	}
	var unset []string
	if topology.Local != nil {
		for name, value := range map[string]string{
			"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "NCCL_IB_DISABLE": "1",
			"NCCL_P2P_LEVEL": "SYS", "NCCL_PROTO": "LL,LL128,Simple",
			"VLLM_ENABLE_PCIE_ALLREDUCE": "1", "VLLM_PCIE_ALLREDUCE_BACKEND": "b12x",
			"VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE":          "64KB",
			"VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE": "84KB",
		} {
			environment[name] = value
		}
		unset = []string{"CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"}
	} else {
		for name, value := range map[string]string{
			"CUDA_VISIBLE_DEVICES":       strconv.Itoa(topology.Spark.DeviceID),
			"VLLM_ENABLE_PCIE_ALLREDUCE": "0", "INSTANTTENSOR_BUFFER_SIZE": "1342177280",
			"INSTANTTENSOR_CONCURRENCY": "1", "INSTANTTENSOR_IO_DEPTH": "3",
			"NCCL_NET_PLUGIN": "none", "NCCL_DEBUG": topology.Spark.NCCLDebug,
			"NCCL_IGNORE_CPU_AFFINITY":     "1",
			"NCCL_IB_GID_INDEX":            strconv.Itoa(topology.Spark.NCCLIBGIDIndex),
			"NCCL_IB_MERGE_NICS":           strconv.Itoa(boolInt(topology.Spark.NCCLIBMergeNICs)),
			"NCCL_IB_SUBNET_AWARE_ROUTING": "1",
		} {
			environment[name] = value
		}
		unset = []string{"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"}
	}
	if profile.MambaCacheMode != nil {
		environment["VLLM_SSM_CONV_STATE_LAYOUT"] = "DS"
	}
	if profile.CUDADeviceMaxConnections != nil {
		environment["CUDA_DEVICE_MAX_CONNECTIONS"] = strconv.Itoa(*profile.CUDADeviceMaxConnections)
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
		_, managed := environment[override.Name]
		managed = managed || contains(unset, override.Name) || topology.Spark != nil && sparkNodeEnvironment[override.Name]
		if managed {
			return nil, nil, fmt.Errorf("environment variable %q is launcher-managed and cannot be overridden with --env", override.Name)
		}
		environment[override.Name] = override.Value
	}
	return environment, unset, nil
}

func contains(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func speculativeConfig(profile ModelProfile, modelSource string, checkpointPath *string, options LaunchOptions) (*string, string, int, *MTPMoEBackendDecision, error) {
	method := profile.DefaultSpeculator
	if options.Speculator != nil {
		method = *options.Speculator
	}
	tokens := profile.MTPTokens
	if options.SpeculativeTokens != nil {
		tokens = *options.SpeculativeTokens
	} else if method == "dflash2" {
		if profile.DFlash2Tokens == nil {
			return nil, method, 0, nil, fmt.Errorf("model %q does not define a DFlash2 speculator", profile.Name)
		}
		tokens = *profile.DFlash2Tokens
	}
	if method == "none" || tokens == 0 {
		if options.AdaptiveSpeculativeTokens {
			return nil, method, tokens, nil, fmt.Errorf("adaptive speculation requires positive MTP depth")
		}
		return nil, method, 0, nil, nil
	}
	if method == "dflash2" {
		if options.AdaptiveSpeculativeTokens {
			return nil, method, tokens, nil, fmt.Errorf("adaptive speculation is supported only for MTP")
		}
		if profile.DFlash2Model == nil {
			return nil, method, tokens, nil, fmt.Errorf("model %q does not define a DFlash2 speculator", profile.Name)
		}
		config := map[string]any{"method": "dflash", "model": *profile.DFlash2Model, "num_speculative_tokens": tokens, "kv_cache_dtype": "auto"}
		encoded, err := compactJSON(config)
		return &encoded, method, tokens, nil, err
	}
	if method != "mtp" {
		return nil, method, tokens, nil, fmt.Errorf("unsupported speculator: %q", method)
	}
	var decision MTPMoEBackendDecision
	var err error
	if checkpointPath != nil {
		decision, err = ResolveMTPMoEBackend(*checkpointPath, profile.DType)
		if err != nil {
			return nil, method, tokens, nil, err
		}
		if profile.MTPMoEQuantization != "" && decision.Quantization != profile.MTPMoEQuantization {
			return nil, method, tokens, nil, fmt.Errorf(
				"MTP MoE quantization mismatch: profile declares %s, checkpoint contains %s",
				profile.MTPMoEQuantization, decision.Quantization,
			)
		}
	} else {
		if profile.MTPMoEQuantization == "" {
			return nil, method, tokens, nil, fmt.Errorf("cannot select an MTP MoE backend for %q; its manifest does not declare mtp_moe_quantization and no local checkpoint is available", modelSource)
		}
		decision, err = MTPMoEBackendFromQuantization(profile.MTPMoEQuantization, "lil.yaml")
		if err != nil {
			return nil, method, tokens, nil, err
		}
	}
	config := map[string]any{"method": "mtp", "num_speculative_tokens": tokens, "moe_backend": decision.Backend}
	if profile.MTPModel != nil && *profile.MTPModel == "target" {
		config["model"] = modelSource
	}
	if profile.MTPAttentionBackend != nil {
		config["attention_backend"] = *profile.MTPAttentionBackend
	}
	if profile.MTPDraftSampleMethod != nil {
		config["draft_sample_method"] = *profile.MTPDraftSampleMethod
	}
	if options.AdaptiveSpeculativeTokens {
		initial := min(3, tokens)
		if options.AdaptiveInitial != nil {
			initial = *options.AdaptiveInitial
		}
		if initial <= 0 || initial > tokens {
			return nil, method, tokens, nil, fmt.Errorf("adaptive initial depth must be positive and no greater than the speculative token count")
		}
		config["adaptive_speculative_tokens_window"] = options.AdaptiveWindow
		config["adaptive_speculative_tokens_initial"] = initial
	}
	encoded, err := compactJSON(config)
	return &encoded, method, tokens, &decision, err
}

func cudagraphCaptureSizes(maxNumSeqs, maxNumBatchedTokens, queryLen int) []int {
	maxMixedTokens := min(maxNumSeqs*queryLen*2, maxNumBatchedTokens, 1024)
	sizes := make(map[int]bool, maxNumSeqs+maxMixedTokens/8)
	add := func(size int) {
		if size > 0 && size <= maxNumBatchedTokens {
			sizes[size] = true
		}
	}

	for sequenceCount := 1; sequenceCount <= maxNumSeqs; sequenceCount++ {
		add(sequenceCount * queryLen)
	}
	for _, size := range []int{1, 2, 4} {
		if size <= maxMixedTokens {
			add(size)
		}
	}
	for size := 8; size <= min(maxMixedTokens, 255); size += 8 {
		add(size)
	}
	for size := 256; size <= maxMixedTokens; size += 16 {
		add(size)
	}
	add(maxMixedTokens)

	result := make([]int, 0, len(sizes))
	for size := range sizes {
		result = append(result, size)
	}
	sort.Ints(result)
	return result
}

func compilationConfig(settings LaunchSettings, maxNumSeqs, maxNumBatchedTokens int, speculator string, speculativeTokens int) map[string]any {
	if settings.CompilationConfig == nil {
		return nil
	}
	config := cloneValue(settings.CompilationConfig).(map[string]any)
	queryLen := 1
	if speculator == "mtp" && speculativeTokens > 0 {
		queryLen = speculativeTokens + 1
	}
	config["cudagraph_capture_sizes"] = cudagraphCaptureSizes(
		maxNumSeqs, maxNumBatchedTokens, queryLen,
	)
	return config
}

func profilerConfig(options LaunchOptions, topology Topology) (*string, error) {
	if options.Profiler == nil {
		return nil, nil
	}
	profiler := options.Profiler
	outputDir, err := expandPath(profiler.OutputDir)
	if err != nil {
		return nil, err
	}
	if !filepath.IsAbs(outputDir) {
		base := topology.RepoRoot()
		if topology.Spark != nil {
			base = topology.Spark.RuntimeRepoRoot
		}
		outputDir = filepath.Join(base, outputDir)
	}
	outputDir, err = filepath.Abs(outputDir)
	if err != nil {
		return nil, err
	}
	if profiler.MaxIterations <= 0 {
		return nil, fmt.Errorf("profiler max iterations must be positive")
	}
	encoded, err := compactJSON(map[string]any{
		"profiler": "torch", "torch_profiler_dir": outputDir,
		"torch_profiler_record_shapes": profiler.RecordShapes,
		"torch_profiler_with_memory":   profiler.WithMemory,
		"torch_profiler_with_stack":    profiler.WithStack,
		"torch_profiler_with_flops":    profiler.WithFlops,
		"torch_profiler_use_gzip":      profiler.UseGzip,
		"ignore_frontend":              true, "delay_iterations": 0,
		"max_iterations": profiler.MaxIterations,
	})
	return &encoded, err
}

func validateExtraArgs(extraArgs []string) error {
	for _, argument := range extraArgs {
		flag := strings.SplitN(argument, "=", 2)[0]
		if alias, ok := vllmFlagAliases[flag]; ok {
			flag = alias
		}
		flag = strings.ReplaceAll(flag, "_", "-")
		managed := managedVLLMFlags[flag] || managedVLLMNegatedFlags[flag]
		if !managed {
			for managedFlag := range managedVLLMFlags {
				if strings.HasPrefix(flag, managedFlag+".") {
					managed = true
					break
				}
			}
		}
		if managed {
			return fmt.Errorf("vLLM option %q is launcher-managed; use its launcher option instead", flag)
		}
	}
	return nil
}

func buildVLLMArgv(profile ModelProfile, topology Topology, options LaunchOptions, tpSize int, deviceIDs []int, modelSource string, checkpointPath *string, servedModelName, host string, port int, capacity Capacity, settings LaunchSettings) ([]string, map[string]any, error) {
	var argv []string
	if topology.Spark != nil {
		argv = []string{topology.Spark.VLLMBin, "serve", modelSource}
	} else {
		argv = []string{topology.Local.Python, "-m", "vllm.entrypoints.cli.main", "serve", modelSource}
	}
	appendOption(&argv, "--served-model-name", &servedModelName)
	appendOption(&argv, "--host", &host)
	portValue := strconv.Itoa(port)
	appendOption(&argv, "--port", &portValue)
	if profile.TrustRemoteCode {
		argv = append(argv, "--trust-remote-code")
	}
	if deviceIDs != nil {
		parts := make([]string, len(deviceIDs))
		for i, id := range deviceIDs {
			parts[i] = strconv.Itoa(id)
		}
		value := strings.Join(parts, ",")
		appendOption(&argv, "--device-ids", &value)
	}
	tpValue := strconv.Itoa(tpSize)
	appendOption(&argv, "--tensor-parallel-size", &tpValue)
	appendOption(&argv, "--pipeline-parallel-size", stringPointer("1"))
	appendOption(&argv, "--decode-context-parallel-size", stringPointer(strconv.Itoa(options.DCPSize)))
	appendOption(&argv, "--dcp-comm-backend", &options.DCPCommBackend)
	if topology.Spark != nil {
		argv = append(argv, "--disable-custom-all-reduce")
	}
	appendOption(&argv, "--mamba-cache-mode", profile.MambaCacheMode)
	argv = append(argv, "--enable-prefix-caching", "--enable-chunked-prefill")
	if profile.AsyncScheduling {
		argv = append(argv, "--async-scheduling")
	}
	appendOption(&argv, "--dtype", &profile.DType)
	appendOption(&argv, "--kv-cache-dtype", &options.KVCacheDType)
	appendOption(&argv, "--quantization", profile.Quantization)
	appendOption(&argv, "--attention-backend", profile.AttentionBackend)
	appendOption(&argv, "--block-size", intPointerString(profile.BlockSize))
	appendOption(&argv, "--linear-backend", &profile.LinearBackend)
	appendOption(&argv, "--moe-backend", &profile.MoEBackend)
	appendOption(&argv, "--gdn-decode-kernel", profile.GDNDecodeKernel)
	if profile.DisableFlashinferAutotune {
		argv = append(argv, "--no-enable-flashinfer-autotune")
	}
	appendOption(&argv, "--load-format", &profile.LoadFormat)
	if profile.ModelLoaderExtraConfig != nil {
		value, err := compactJSON(profile.ModelLoaderExtraConfig)
		if err != nil {
			return nil, nil, err
		}
		appendOption(&argv, "--model-loader-extra-config", &value)
	}
	appendOption(&argv, "--gpu-memory-utilization", floatPointerString(capacity.GPUMemoryUtilization))
	appendOption(&argv, "--kv-cache-memory-bytes", capacity.KVCacheMemoryBytes)
	appendOption(&argv, "--max-model-len", &capacity.MaxModelLen)
	appendOption(&argv, "--max-num-seqs", stringPointer(strconv.Itoa(capacity.MaxNumSeqs)))
	appendOption(&argv, "--max-num-batched-tokens", stringPointer(strconv.Itoa(capacity.MaxNumBatchedTokens)))
	speculative, speculator, tokens, decision, err := speculativeConfig(profile, modelSource, checkpointPath, options)
	if err != nil {
		return nil, nil, err
	}
	appendOption(&argv, "--speculative-config", speculative)
	appendOption(&argv, "--mm-encoder-tp-mode", profile.MMEncoderTPMode)
	appendOption(&argv, "--mm-processor-cache-gb", floatPointerString(profile.MMProcessorCacheGB))
	if profile.LimitMMPerPrompt != nil {
		value, err := compactJSON(profile.LimitMMPerPrompt)
		if err != nil {
			return nil, nil, err
		}
		appendOption(&argv, "--limit-mm-per-prompt", &value)
	}
	appendOption(&argv, "--long-prefill-token-threshold", intPointerString(profile.LongPrefillTokenThreshold))
	appendOption(&argv, "--reasoning-parser", &profile.ReasoningParser)
	appendOption(&argv, "--tool-call-parser", &profile.ToolCallParser)
	argv = append(argv, "--enable-auto-tool-choice")
	appendOption(&argv, "--generation-config", profile.GenerationConfig)
	if profile.HFOverridesSet {
		value, err := compactJSON(profile.HFOverrides)
		if err != nil {
			return nil, nil, err
		}
		appendOption(&argv, "--hf-overrides", &value)
	}
	resolvedCompilationConfig := compilationConfig(
		settings, capacity.MaxNumSeqs, capacity.MaxNumBatchedTokens,
		speculator, tokens,
	)
	if resolvedCompilationConfig != nil {
		value, err := compactJSON(resolvedCompilationConfig)
		if err != nil {
			return nil, nil, err
		}
		appendOption(&argv, "--compilation-config", &value)
	}
	profiler, err := profilerConfig(options, topology)
	if err != nil {
		return nil, nil, err
	}
	appendOption(&argv, "--profiler-config", profiler)
	if err := validateExtraArgs(options.ExtraVLLMArgs); err != nil {
		return nil, nil, err
	}
	argv = append(argv, options.ExtraVLLMArgs...)
	metadata := map[string]any{
		"capacity": capacity, "speculator": speculator,
		"speculative_tokens": tokens, "kv_cache_dtype": options.KVCacheDType,
	}
	if decision != nil {
		metadata["mtp_moe"] = map[string]any{"quantization": decision.Quantization, "backend": decision.Backend, "evidence": decision.Evidence}
	}
	return argv, metadata, nil
}

func memoryMetadata(profile ModelProfile, topology Topology, checkpointPath *string, runtimeEnvironment map[string]string, capacity Capacity, tpSize int) (map[string]any, error) {
	policy := "vllm_runtime_profile"
	if capacity.KVCacheMemoryBytes != nil {
		policy = "explicit"
	}
	metadata := map[string]any{"device_bytes_per_rank": topology.DeviceMemoryBytes(), "kv_cache_allocation": policy, "checkpoint_path": checkpointPath}
	var memory *CheckpointMemory
	if checkpointPath != nil {
		var err error
		memory, err = CheckpointMemoryInfo(*checkpointPath, runtimeEnvironment["VLLM_PLE_CPU_OFFLOAD"] == "1")
		if err != nil {
			return nil, err
		}
	}
	if memory == nil && profile.WeightBytes > 0 {
		memory = &CheckpointMemory{TotalBytes: profile.WeightBytes, Source: "lil.yaml weight_bytes"}
	}
	if memory == nil {
		metadata["estimate"] = "unavailable: no safetensors size metadata or manifest weight_bytes"
		return metadata, nil
	}
	deviceWeightBytes := max(int64(0), memory.TotalBytes-memory.MappedHostBytes)
	shardedWeightBytes := (deviceWeightBytes + int64(tpSize) - 1) / int64(tpSize)
	utilization := 1.0
	if capacity.GPUMemoryUtilization != nil {
		utilization = *capacity.GPUMemoryUtilization
	}
	allocatableBytes := int64(math.Floor(float64(topology.DeviceMemoryBytes()) * utilization))
	runtimeReserveBytes := max(int64(8*1024*1024*1024), int64(math.Ceil(float64(topology.DeviceMemoryBytes())*0.08)))
	estimatedKVBytes := max(int64(0), allocatableBytes-shardedWeightBytes-runtimeReserveBytes)
	for key, value := range map[string]any{
		"checkpoint_weight_bytes":                 memory.TotalBytes,
		"mapped_host_weight_bytes":                memory.MappedHostBytes,
		"estimated_sharded_weight_bytes_per_rank": shardedWeightBytes,
		"utilization_limit_bytes_per_rank":        allocatableBytes,
		"runtime_reserve_bytes_per_rank":          runtimeReserveBytes,
		"estimated_safe_kv_bytes_per_rank":        estimatedKVBytes,
		"estimate_source":                         memory.Source, "estimate_is_advisory": true,
	} {
		metadata[key] = value
	}
	return metadata, nil
}

type mountSpec struct {
	source, target string
	readOnly       bool
}

func boolInt(value bool) int {
	if value {
		return 1
	}
	return 0
}

func sparkContainerName(topology *SparkRDMATopology, profileName string, tpSize int) string {
	return containerNameSanitizer.ReplaceAllString(
		fmt.Sprintf("%s-%s-tp%d", topology.ContainerNamePrefix, profileName, tpSize),
		"-",
	)
}

func pathWithin(path, root string) bool {
	relative, err := filepath.Rel(root, path)
	return err == nil && relative != ".." && !strings.HasPrefix(relative, ".."+string(filepath.Separator))
}

func ProfilerOutputDir(argv []string) (string, bool) {
	for index, argument := range argv {
		if index == 0 || argv[index-1] != "--profiler-config" {
			continue
		}
		var config map[string]any
		if json.Unmarshal([]byte(argument), &config) != nil {
			return "", false
		}
		value, ok := config["torch_profiler_dir"].(string)
		return value, ok && value != ""
	}
	return "", false
}

func sparkNodeLaunches(topology *SparkRDMATopology, profile ModelProfile, runtimeEnvironment map[string]string, vllmArgv []string, localModelPath *string, tpSize int) []SparkNodeLaunch {
	containerName := sparkContainerName(topology, profile.Name, tpSize)
	mounts := []mountSpec{{topology.RuntimeRepoRoot, topology.RuntimeRepoRoot, false}, {topology.RuntimeB12XRoot, topology.RuntimeB12XRoot, false}}
	for _, mount := range topology.CacheMounts {
		mounts = append(mounts, mountSpec{mount.Source, mount.Target, false})
	}
	if localModelPath != nil {
		mounts = append(mounts, mountSpec{*localModelPath, *localModelPath, true})
	}
	if profileDir, ok := ProfilerOutputDir(vllmArgv); ok {
		covered := false
		for _, mount := range mounts {
			covered = covered || pathWithin(profileDir, mount.source)
		}
		if !covered {
			mounts = append(mounts, mountSpec{profileDir, profileDir, false})
		}
	}
	common := []string{"docker", "run", "--rm", "--network", "host", "--name", containerName, "--gpus", "all", "--cap-add=IPC_LOCK", "--device=/dev/infiniband", "--memory", fmt.Sprintf("%dg", topology.ContainerMemoryGB), "--memory-swap", fmt.Sprintf("%dg", topology.ContainerMemorySwapGB), "--shm-size", fmt.Sprintf("%dg", topology.ContainerShmGB), "--pids-limit", strconv.Itoa(topology.ContainerPidsLimit), "--ulimit", fmt.Sprintf("nofile=%d:%d", topology.ContainerNofileLimit, topology.ContainerNofileLimit), "--entrypoint", ""}
	for _, mount := range mounts {
		value := fmt.Sprintf("type=bind,src=%s,dst=%s", mount.source, mount.target)
		if mount.readOnly {
			value += ",readonly"
		}
		common = append(common, "--mount", value)
	}
	selected := topology.Nodes[:tpSize]
	launches := make([]SparkNodeLaunch, 0, len(selected))
	for rank, node := range selected {
		environment := copyStringMap(runtimeEnvironment)
		for name, value := range map[string]string{
			"VLLM_HOST_IP": node.Address, "MN_IF_NAME": node.EthernetInterface,
			"UCX_NET_DEVICES": node.EthernetInterface, "NCCL_SOCKET_IFNAME": node.EthernetInterface,
			"NCCL_IB_HCA": strings.Join(node.RDMAInterfaces, ","), "NCCL_IB_DISABLE": "0",
			"OMPI_MCA_btl_tcp_if_include": node.EthernetInterface,
			"GLOO_SOCKET_IFNAME":          node.EthernetInterface, "TP_SOCKET_IFNAME": node.EthernetInterface,
		} {
			environment[name] = value
		}
		nodeVLLM := append([]string{}, vllmArgv...)
		nodeVLLM = append(nodeVLLM, "--nnodes", strconv.Itoa(len(selected)), "--node-rank", strconv.Itoa(rank), "--master-addr", selected[0].Address, "--master-port", strconv.Itoa(topology.MasterPort))
		if rank > 0 {
			nodeVLLM = append(nodeVLLM, "--headless")
		}
		dockerArgv := append([]string{}, common...)
		dockerArgv = append(dockerArgv, "--detach")
		for _, name := range sortedMapKeys(environment) {
			dockerArgv = append(dockerArgv, "--env", name+"="+environment[name])
		}
		dockerArgv = append(dockerArgv, topology.Image)
		dockerArgv = append(dockerArgv, nodeVLLM...)
		launches = append(launches, SparkNodeLaunch{node, rank, containerName, environment, nodeVLLM, dockerArgv})
	}
	return launches
}

func copyStringMap(source map[string]string) map[string]string {
	result := make(map[string]string, len(source))
	for key, value := range source {
		result[key] = value
	}
	return result
}

func sortedMapKeys[V any](values map[string]V) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func BuildLaunchSpec(profile ModelProfile, topology Topology, options LaunchOptions) (LaunchSpec, error) {
	if err := validateOptions(options); err != nil {
		return LaunchSpec{}, err
	}
	tpSize, err := resolveTPSize(profile, topology, options.TPSize)
	if err != nil {
		return LaunchSpec{}, err
	}
	deviceIDs, err := resolveDevices(topology, options, tpSize)
	if err != nil {
		return LaunchSpec{}, err
	}
	var localModelPath, checkpointPath *string
	modelSource := profile.Model
	if options.ModelPath != nil {
		resolved := resolveExistingPath(*options.ModelPath)
		localModelPath = &resolved
		checkpointPath = &resolved
		modelSource = resolved
	} else {
		checkpointPath = nil
	}
	servedModelName := profile.ServedModelName
	if options.ServedModelName != nil {
		servedModelName = *options.ServedModelName
	}
	host := topology.Host()
	if options.Host != nil {
		host = *options.Host
	}
	port := topology.Port()
	if options.Port != nil {
		port = *options.Port
	}
	policy, err := profile.Launch.Topology(topology.Kind)
	if err != nil {
		return LaunchSpec{}, err
	}
	settings := policy.Resolve(tpSize)
	capacity := resolveCapacity(settings, topology, options)
	runtimeEnvironment, unsetEnvironment, err := runtimeEnvironment(profile, topology, settings, options)
	if err != nil {
		return LaunchSpec{}, err
	}
	vllmArgv, metadata, err := buildVLLMArgv(profile, topology, options, tpSize, deviceIDs, modelSource, checkpointPath, servedModelName, host, port, capacity, settings)
	if err != nil {
		return LaunchSpec{}, err
	}
	memory, err := memoryMetadata(profile, topology, checkpointPath, runtimeEnvironment, capacity, tpSize)
	if err != nil {
		return LaunchSpec{}, err
	}
	metadata["memory"] = memory
	downloadRepositories := []string{}
	if checkpointPath == nil {
		downloadRepositories = append(downloadRepositories, profile.Model)
	}
	if metadata["speculator"] == "dflash2" && profile.DFlash2Model != nil &&
		!contains(downloadRepositories, *profile.DFlash2Model) {
		downloadRepositories = append(downloadRepositories, *profile.DFlash2Model)
	}
	spec := LaunchSpec{Model: profile, Topology: topology, TPSize: tpSize, ModelSource: modelSource, CheckpointPath: checkpointPath, ServedModelName: servedModelName, Host: host, Port: port, Detach: options.Detach, SyncCode: options.SyncCode, SyncModel: options.SyncModel, DownloadRepositories: downloadRepositories, RuntimeEnvironment: runtimeEnvironment, UnsetEnvironment: unsetEnvironment, VLLMArgv: vllmArgv, DeviceIDs: deviceIDs, Metadata: metadata}
	if topology.Spark != nil {
		spec.SparkNodes = sparkNodeLaunches(topology.Spark, profile, runtimeEnvironment, vllmArgv, localModelPath, tpSize)
		spec.HostEnvironment = map[string]string{}
		spec.CommandArgv = append([]string{}, spec.SparkNodes[0].DockerArgv...)
	} else {
		spec.HostEnvironment = copyStringMap(runtimeEnvironment)
		spec.CommandArgv = append([]string{}, vllmArgv...)
	}
	return spec, nil
}
