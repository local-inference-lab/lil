// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package launcher

import (
	"encoding/json"
	"fmt"
	"math"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

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
	"--tokenizer-mode", "--scheduler-reserve-full-isl", "--enable-prompt-tokens-details",
	"--prefill-schedule-interval",
	"--enable-force-include-usage", "--enable-request-id-headers",
	"--default-chat-template-kwargs", "--prefix-cache-retention-interval",
	"--max-cudagraph-capture-size", "--cudagraph-capture-sizes",
	"--enable-flashinfer-autotune",
)

var managedVLLMNegatedFlags = stringSet(
	"--no-async-scheduling", "--no-disable-custom-all-reduce",
	"--no-enable-auto-tool-choice", "--no-enable-chunked-prefill",
	"--no-enable-prefix-caching", "--no-trust-remote-code", "--no-headless",
	"--no-scheduler-reserve-full-isl", "--no-enable-prompt-tokens-details",
	"--no-enable-force-include-usage", "--no-enable-request-id-headers",
)

var vllmFlagAliases = map[string]string{
	"-cc": "--compilation-config", "-dcp": "--decode-context-parallel-size",
	"-n": "--nnodes", "-r": "--node-rank", "-pp": "--pipeline-parallel-size",
	"-tp": "--tensor-parallel-size",
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

// MemoryEstimate is the advisory per-rank memory arithmetic used both to pick
// a default tensor-parallel size and to report headroom.
type MemoryEstimate struct {
	CheckpointWeightBytes int64
	MappedHostWeightBytes int64
	ShardedWeightBytes    int64
	UtilizationLimitBytes int64
	RuntimeReserveBytes   int64
	EstimatedSafeKVBytes  int64
	Fits                  bool
}

func estimateMemory(deviceBytes int64, utilization float64, weightBytes, mappedHostBytes int64, tpSize int) MemoryEstimate {
	deviceWeightBytes := max(int64(0), weightBytes-mappedHostBytes)
	sharded := (deviceWeightBytes + int64(tpSize) - 1) / int64(tpSize)
	allocatable := int64(math.Floor(float64(deviceBytes) * utilization))
	reserve := max(int64(8*1024*1024*1024), int64(math.Ceil(float64(deviceBytes)*0.08)))
	kv := allocatable - sharded - reserve
	return MemoryEstimate{
		CheckpointWeightBytes: weightBytes, MappedHostWeightBytes: mappedHostBytes,
		ShardedWeightBytes: sharded, UtilizationLimitBytes: allocatable,
		RuntimeReserveBytes: reserve, EstimatedSafeKVBytes: max(int64(0), kv),
		Fits: kv > 0,
	}
}

// SupportedTPSizes lists the tensor-parallel sizes the topology can host that
// divide the checkpoint's attention heads.
func SupportedTPSizes(facts *CheckpointFacts, topology Topology) []int {
	sizes := []int{}
	for size := 1; size <= topology.MaxTPSize(); size++ {
		if facts.AttentionHeads%size == 0 {
			sizes = append(sizes, size)
		}
	}
	return sizes
}

// DefaultTPSize derives the default tensor-parallel size from the topology
// policy: the whole cluster, or the smallest size whose sharded weights leave
// KV headroom.
func DefaultTPSize(facts *CheckpointFacts, topology Topology, utilization float64) (int, error) {
	supported := SupportedTPSizes(facts, topology)
	if len(supported) == 0 {
		return 0, fmt.Errorf(
			"%d attention heads are not divisible by any TP size up to %d",
			facts.AttentionHeads, topology.MaxTPSize(),
		)
	}
	if topology.DefaultTP == DefaultTPAll {
		size := topology.MaxTPSize()
		if facts.AttentionHeads%size != 0 {
			return 0, fmt.Errorf(
				"topology %q defaults to all %d ranks, which does not divide %d attention heads; pass --tp",
				topology.Name(), size, facts.AttentionHeads,
			)
		}
		return size, nil
	}
	for _, size := range supported {
		if estimateMemory(topology.DeviceMemoryBytes(), utilization, facts.WeightBytes, 0, size).Fits {
			return size, nil
		}
	}
	largest := supported[len(supported)-1]
	estimate := estimateMemory(topology.DeviceMemoryBytes(), utilization, facts.WeightBytes, 0, largest)
	return 0, fmt.Errorf(
		"stored weights of %.1f GiB do not fit topology %q at any TP up to %d (%.1f GiB per rank after sharding, %.1f GiB usable); pass --tp to override",
		gib(facts.WeightBytes), topology.Name(), largest,
		gib(estimate.ShardedWeightBytes), gib(estimate.UtilizationLimitBytes-estimate.RuntimeReserveBytes),
	)
}

func gib(bytes int64) float64 { return float64(bytes) / float64(int64(1)<<30) }

func resolveTPSize(facts *CheckpointFacts, topology Topology, requested *int, utilization float64) (int, int, error) {
	defaultSize, defaultErr := DefaultTPSize(facts, topology, utilization)
	if requested == nil {
		return defaultSize, defaultSize, defaultErr
	}
	tpSize := *requested
	if tpSize <= 0 {
		return 0, 0, fmt.Errorf("tensor parallel size must be positive")
	}
	if tpSize > topology.MaxTPSize() {
		if topology.Spark != nil {
			return 0, 0, fmt.Errorf(
				"Spark/RDMA topology %q has %d one-GPU nodes, but TP=%d was requested",
				topology.Name(), topology.MaxTPSize(), tpSize,
			)
		}
		return 0, 0, fmt.Errorf(
			"topology %q has at most %d local GPUs, but TP=%d was requested",
			topology.Name(), topology.MaxTPSize(), tpSize,
		)
	}
	if facts.AttentionHeads%tpSize != 0 {
		return 0, 0, fmt.Errorf(
			"checkpoint has %d attention heads, which is not divisible by TP=%d",
			facts.AttentionHeads, tpSize,
		)
	}
	return tpSize, defaultSize, nil
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

func defaultCapacityLayer() map[string]any {
	return map[string]any{
		"gpu_memory_utilization": nil, "kv_cache_memory_bytes": nil, "max_model_len": "auto",
		"max_num_seqs": 8, "max_num_batched_tokens": 4096,
	}
}

// resolveLaunchSettings folds launcher defaults, the manifest base layer, and
// every override whose condition matches this launch, in manifest order.
func resolveLaunchSettings(profile ModelProfile, topology Topology, tpSize int, speculator string) (LaunchSettings, []int, error) {
	capacity := defaultCapacityLayer()
	var compilation map[string]any
	environment := map[string]string{}
	apply := func(layer LaunchLayer) {
		if layer.Capacity != nil {
			capacity = deepMerge(capacity, layer.Capacity)
		}
		if layer.Compilation != nil {
			if compilation == nil {
				compilation = map[string]any{}
			}
			compilation = deepMerge(compilation, layer.Compilation)
		}
		for name, value := range layer.Environment {
			environment[name] = value
		}
	}
	apply(profile.Base)
	applied := []int{}
	for index, override := range profile.Overrides {
		if override.When.Matches(topology.Kind, topology.CuteDSLArch(), tpSize, speculator) {
			apply(override.Layer)
			applied = append(applied, index)
		}
	}
	parsed, err := parseCapacity(capacity, "resolved capacity")
	if err != nil {
		return LaunchSettings{}, nil, err
	}
	return LaunchSettings{Capacity: parsed, CompilationConfig: compilation, Environment: environment}, applied, nil
}

// resolveUtilization picks the GPU memory utilization: the command line wins,
// then a manifest capacity value, then the topology default.
func resolveUtilization(settings LaunchSettings, topology Topology, options LaunchOptions) float64 {
	if options.GPUMemoryUtilization != nil {
		return *options.GPUMemoryUtilization
	}
	if settings.Capacity.GPUMemoryUtilization != nil {
		return *settings.Capacity.GPUMemoryUtilization
	}
	return topology.MemoryUtilization()
}

func resolveCapacity(settings LaunchSettings, topology Topology, options LaunchOptions) Capacity {
	capacity := settings.Capacity
	utilization := resolveUtilization(settings, topology, options)
	capacity.GPUMemoryUtilization = &utilization
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
	if options.KVCacheDType != nil && *options.KVCacheDType == "" {
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
	if options.Speculator != nil && !speculatorMethods[*options.Speculator] {
		return fmt.Errorf("speculator must be mtp, dflash, dspark, or none")
	}
	return nil
}

type speculation struct {
	Config   *string
	Method   string
	Tokens   int
	Decision *MTPMoEBackendDecision
}

// Effective is the method that actually runs: a zero-depth launch is "none"
// for overrides and graph sizing.
func (s speculation) Effective() string {
	if s.Tokens == 0 {
		return "none"
	}
	return s.Method
}

func mtpBackendDecision(policy MTPPolicy, facts *CheckpointFacts, dtype string) (MTPMoEBackendDecision, error) {
	decision, err := MTPMoEBackendFromConfig(facts.Config, dtype)
	if policy.MoEQuantization != "" && decision.Quantization != "" && decision.Quantization != policy.MoEQuantization {
		return MTPMoEBackendDecision{}, fmt.Errorf(
			"MTP MoE quantization mismatch: manifest declares %s, %s contains %s",
			policy.MoEQuantization, facts.Source, decision.Quantization,
		)
	}
	if policy.MoEBackend != nil {
		quantization := decision.Quantization
		if quantization == "" {
			quantization = policy.MoEQuantization
		}
		return MTPMoEBackendDecision{
			Quantization: quantization, Backend: *policy.MoEBackend,
			Evidence: []string{"lil.yaml moe_backend"},
		}, nil
	}
	if err != nil {
		if policy.MoEQuantization == "" {
			return MTPMoEBackendDecision{}, fmt.Errorf(
				"cannot select an MTP MoE backend from %s and the manifest declares no moe_quantization: %w",
				facts.Source, err,
			)
		}
		return MTPMoEBackendFromQuantization(
			policy.MoEQuantization, fmt.Sprintf("lil.yaml (checkpoint metadata inconclusive: %v)", err),
		)
	}
	return decision, nil
}

func speculativeConfig(profile ModelProfile, facts *CheckpointFacts, modelSource string, options LaunchOptions) (speculation, error) {
	method := profile.Speculators.Default
	if options.Speculator != nil {
		method = *options.Speculator
	}
	result := speculation{Method: method}
	if method == "none" {
		if options.AdaptiveSpeculativeTokens {
			return result, fmt.Errorf("adaptive speculation requires an MTP launch")
		}
		if options.AdaptiveVerification {
			return result, fmt.Errorf("adaptive verification requires a DSpark launch")
		}
		return result, nil
	}
	if options.AdaptiveVerification && method != "dspark" {
		return result, fmt.Errorf("adaptive verification is supported only for DSpark")
	}
	if options.AdaptiveSpeculativeTokens && method != "mtp" {
		return result, fmt.Errorf("adaptive speculation is supported only for MTP")
	}
	switch method {
	case "dflash":
		policy := profile.Speculators.DFlash
		if policy == nil {
			return result, fmt.Errorf("model %q does not define a DFlash speculator", profile.Name)
		}
		tokens := policy.Tokens
		if options.SpeculativeTokens != nil {
			tokens = *options.SpeculativeTokens
		}
		result.Tokens = tokens
		if tokens == 0 {
			return result, nil
		}
		config := map[string]any{
			"method": "dflash", "model": policy.Model,
			"num_speculative_tokens": tokens, "kv_cache_dtype": "auto",
		}
		encoded, err := compactJSON(config)
		result.Config = &encoded
		return result, err
	case "dspark":
		policy := profile.Speculators.DSpark
		if policy == nil {
			return result, fmt.Errorf("model %q does not define a DSpark speculator", profile.Name)
		}
		tokens := policy.Tokens
		if options.SpeculativeTokens != nil {
			tokens = *options.SpeculativeTokens
		}
		result.Tokens = tokens
		if tokens == 0 {
			return result, nil
		}
		config := map[string]any{
			"method": "dspark", "model": modelSource, "num_speculative_tokens": tokens,
		}
		if profile.Revision != "" && modelSource == profile.Model {
			config["revision"] = profile.Revision
		}
		if policy.DraftSampleMethod != nil {
			config["draft_sample_method"] = *policy.DraftSampleMethod
		}
		if policy.RejectionSampleMethod != nil {
			config["rejection_sample_method"] = *policy.RejectionSampleMethod
		}
		if policy.Attention != nil {
			config["attention_backend"] = *policy.Attention
		}
		if policy.AdaptiveVerification || options.AdaptiveVerification {
			config["enable_adaptive_verification"] = true
		}
		encoded, err := compactJSON(config)
		result.Config = &encoded
		return result, err
	case "mtp":
		policy := profile.Speculators.MTP
		if policy == nil {
			return result, fmt.Errorf("model %q does not define an MTP speculator", profile.Name)
		}
		tokens := policy.Tokens
		if options.SpeculativeTokens != nil {
			tokens = *options.SpeculativeTokens
		}
		result.Tokens = tokens
		if tokens == 0 {
			if options.AdaptiveSpeculativeTokens {
				return result, fmt.Errorf("adaptive speculation requires positive MTP depth")
			}
			return result, nil
		}
		decision, err := mtpBackendDecision(*policy, facts, profile.Kernels.DType)
		if err != nil {
			return result, err
		}
		result.Decision = &decision
		config := map[string]any{
			"method": "mtp", "num_speculative_tokens": tokens, "moe_backend": decision.Backend,
		}
		if policy.WeightsInTarget {
			config["model"] = modelSource
			if profile.Revision != "" && modelSource == profile.Model {
				config["revision"] = profile.Revision
			}
		}
		if policy.Attention != nil {
			config["attention_backend"] = *policy.Attention
		}
		if policy.DraftSampleMethod != nil {
			config["draft_sample_method"] = *policy.DraftSampleMethod
		}
		if policy.RejectionSampleMethod != nil {
			config["rejection_sample_method"] = *policy.RejectionSampleMethod
		}
		if options.AdaptiveSpeculativeTokens {
			initial := min(3, tokens)
			if options.AdaptiveInitial != nil {
				initial = *options.AdaptiveInitial
			}
			if initial <= 0 || initial > tokens {
				return result, fmt.Errorf("adaptive initial depth must be positive and no greater than the speculative token count")
			}
			config["adaptive_speculative_tokens_window"] = options.AdaptiveWindow
			config["adaptive_speculative_tokens_initial"] = initial
		}
		encoded, err := compactJSON(config)
		result.Config = &encoded
		return result, err
	default:
		return result, fmt.Errorf("unsupported speculator: %q", method)
	}
}

// cudagraphCaptureSizes covers every uniform verification batch through
// maxNumSeqs plus a mixed-batch ladder. The ladder extends to twice the
// uniform maximum except for DSpark, whose verifier never exceeds one
// sampled token plus its drafts per request.
func cudagraphCaptureSizes(maxNumSeqs, maxNumBatchedTokens, queryLen, mixedMultiplier int) []int {
	maxMixedTokens := min(maxNumSeqs*queryLen*mixedMultiplier, maxNumBatchedTokens, 1024)
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

func compilationConfig(settings LaunchSettings, maxNumSeqs, maxNumBatchedTokens int, spec speculation) map[string]any {
	if settings.CompilationConfig == nil {
		return nil
	}
	config := cloneValue(settings.CompilationConfig).(map[string]any)
	queryLen, mixedMultiplier := 1, 2
	switch spec.Effective() {
	case "mtp":
		queryLen = spec.Tokens + 1
	case "dspark":
		queryLen, mixedMultiplier = spec.Tokens+1, 1
	}
	config["cudagraph_capture_sizes"] = cudagraphCaptureSizes(
		maxNumSeqs, maxNumBatchedTokens, queryLen, mixedMultiplier,
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

type argvInputs struct {
	tpSize          int
	deviceIDs       []int
	modelSource     string
	servedModelName string
	host            string
	port            int
	capacity        Capacity
	settings        LaunchSettings
	speculation     speculation
	kvCacheDType    string
}

func buildVLLMArgv(profile ModelProfile, topology Topology, options LaunchOptions, in argvInputs) ([]string, error) {
	serving, kernels := profile.Serving, profile.Kernels
	var argv []string
	if topology.Spark != nil {
		argv = []string{topology.Spark.VLLMBin, "serve", in.modelSource}
	} else {
		argv = []string{topology.Local.Python, "-m", "vllm.entrypoints.cli.main", "serve", in.modelSource}
	}
	if profile.Revision != "" && in.modelSource == profile.Model {
		appendOption(&argv, "--revision", &profile.Revision)
	}
	appendOption(&argv, "--served-model-name", &in.servedModelName)
	appendOption(&argv, "--host", &in.host)
	appendOption(&argv, "--port", stringPointer(strconv.Itoa(in.port)))
	if serving.TrustRemoteCode {
		argv = append(argv, "--trust-remote-code")
	}
	if serving.TokenizerMode != "" {
		appendOption(&argv, "--tokenizer-mode", &serving.TokenizerMode)
	}
	if in.deviceIDs != nil {
		parts := make([]string, len(in.deviceIDs))
		for index, id := range in.deviceIDs {
			parts[index] = strconv.Itoa(id)
		}
		appendOption(&argv, "--device-ids", stringPointer(strings.Join(parts, ",")))
	}
	appendOption(&argv, "--tensor-parallel-size", stringPointer(strconv.Itoa(in.tpSize)))
	appendOption(&argv, "--pipeline-parallel-size", stringPointer("1"))
	appendOption(&argv, "--decode-context-parallel-size", stringPointer(strconv.Itoa(options.DCPSize)))
	appendOption(&argv, "--dcp-comm-backend", &options.DCPCommBackend)
	if topology.Spark != nil {
		argv = append(argv, "--disable-custom-all-reduce")
	}
	appendOption(&argv, "--mamba-cache-mode", kernels.MambaCacheMode)
	if serving.PrefixCaching {
		argv = append(argv, "--enable-prefix-caching")
		appendOption(&argv, "--prefix-cache-retention-interval", intPointerString(serving.PrefixCacheRetentionInterval))
	}
	if serving.ChunkedPrefill {
		argv = append(argv, "--enable-chunked-prefill")
	}
	if serving.AsyncScheduling {
		argv = append(argv, "--async-scheduling")
	}
	if !serving.SchedulerReserveFullISL {
		argv = append(argv, "--no-scheduler-reserve-full-isl")
	}
	appendOption(&argv, "--dtype", &kernels.DType)
	appendOption(&argv, "--kv-cache-dtype", &in.kvCacheDType)
	appendOption(&argv, "--quantization", kernels.Quantization)
	appendOption(&argv, "--attention-backend", kernels.Attention)
	appendOption(&argv, "--block-size", intPointerString(kernels.BlockSize))
	appendOption(&argv, "--linear-backend", &kernels.Linear)
	appendOption(&argv, "--moe-backend", &kernels.MoE)
	appendOption(&argv, "--gdn-decode-kernel", kernels.GDNDecode)
	if kernels.FlashinferAutotune != nil {
		if *kernels.FlashinferAutotune {
			argv = append(argv, "--enable-flashinfer-autotune")
		} else {
			argv = append(argv, "--no-enable-flashinfer-autotune")
		}
	}
	appendOption(&argv, "--load-format", &kernels.LoadFormat)
	if kernels.LoaderExtraConfig != nil {
		value, err := compactJSON(kernels.LoaderExtraConfig)
		if err != nil {
			return nil, err
		}
		appendOption(&argv, "--model-loader-extra-config", &value)
	}
	appendOption(&argv, "--gpu-memory-utilization", floatPointerString(in.capacity.GPUMemoryUtilization))
	appendOption(&argv, "--kv-cache-memory-bytes", in.capacity.KVCacheMemoryBytes)
	appendOption(&argv, "--max-model-len", &in.capacity.MaxModelLen)
	appendOption(&argv, "--max-num-seqs", stringPointer(strconv.Itoa(in.capacity.MaxNumSeqs)))
	appendOption(&argv, "--max-num-batched-tokens", stringPointer(strconv.Itoa(in.capacity.MaxNumBatchedTokens)))
	appendOption(&argv, "--speculative-config", in.speculation.Config)
	if multimodal := serving.Multimodal; multimodal != nil {
		appendOption(&argv, "--mm-encoder-tp-mode", multimodal.EncoderTPMode)
		appendOption(&argv, "--mm-processor-cache-gb", floatPointerString(multimodal.ProcessorCacheGB))
		if multimodal.LimitPerPrompt != nil {
			value, err := compactJSON(multimodal.LimitPerPrompt)
			if err != nil {
				return nil, err
			}
			appendOption(&argv, "--limit-mm-per-prompt", &value)
		}
	}
	appendOption(&argv, "--long-prefill-token-threshold", intPointerString(serving.LongPrefillTokenThreshold))
	appendOption(&argv, "--prefill-schedule-interval", intPointerString(serving.PrefillScheduleInterval))
	if serving.ReasoningParser != "" {
		appendOption(&argv, "--reasoning-parser", &serving.ReasoningParser)
	}
	if serving.ToolCallParser != "" {
		appendOption(&argv, "--tool-call-parser", &serving.ToolCallParser)
	}
	if serving.AutoToolChoice {
		argv = append(argv, "--enable-auto-tool-choice")
	}
	appendOption(&argv, "--generation-config", serving.GenerationConfig)
	if serving.PromptTokensDetails {
		argv = append(argv, "--enable-prompt-tokens-details")
	}
	if serving.ForceIncludeUsage {
		argv = append(argv, "--enable-force-include-usage")
	}
	if serving.RequestIDHeaders {
		argv = append(argv, "--enable-request-id-headers")
	}
	for _, key := range sortedMapKeys(serving.ChatTemplateKwargs) {
		argv = append(argv, fmt.Sprintf("--default-chat-template-kwargs.%s=%v", key, serving.ChatTemplateKwargs[key]))
	}
	if len(serving.HFOverrides) > 0 {
		value, err := compactJSON(serving.HFOverrides)
		if err != nil {
			return nil, err
		}
		appendOption(&argv, "--hf-overrides", &value)
	}
	if resolved := compilationConfig(in.settings, in.capacity.MaxNumSeqs, in.capacity.MaxNumBatchedTokens, in.speculation); resolved != nil {
		value, err := compactJSON(resolved)
		if err != nil {
			return nil, err
		}
		appendOption(&argv, "--compilation-config", &value)
	}
	profiler, err := profilerConfig(options, topology)
	if err != nil {
		return nil, err
	}
	appendOption(&argv, "--profiler-config", profiler)
	if err := validateExtraArgs(options.ExtraVLLMArgs); err != nil {
		return nil, err
	}
	return append(argv, options.ExtraVLLMArgs...), nil
}

func memoryMetadata(facts *CheckpointFacts, topology Topology, checkpointPath *string, runtimeEnvironment map[string]string, capacity Capacity, tpSize int) (map[string]any, error) {
	policy := "vllm_runtime_profile"
	if capacity.KVCacheMemoryBytes != nil {
		policy = "explicit"
	}
	metadata := map[string]any{
		"device_bytes_per_rank": topology.DeviceMemoryBytes(),
		"kv_cache_allocation":   policy, "checkpoint_path": checkpointPath,
	}
	weightBytes, mappedHost, source := facts.WeightBytes, int64(0), facts.WeightBytesSource
	if checkpointPath != nil {
		memory, err := CheckpointMemoryInfo(*checkpointPath, runtimeEnvironment["VLLM_PLE_CPU_OFFLOAD"] == "1")
		if err != nil {
			return nil, err
		}
		if memory != nil {
			weightBytes, mappedHost, source = memory.TotalBytes, memory.MappedHostBytes, memory.Source
		}
	}
	utilization := 1.0
	if capacity.GPUMemoryUtilization != nil {
		utilization = *capacity.GPUMemoryUtilization
	}
	estimate := estimateMemory(topology.DeviceMemoryBytes(), utilization, weightBytes, mappedHost, tpSize)
	for key, value := range map[string]any{
		"checkpoint_weight_bytes":                 estimate.CheckpointWeightBytes,
		"mapped_host_weight_bytes":                estimate.MappedHostWeightBytes,
		"estimated_sharded_weight_bytes_per_rank": estimate.ShardedWeightBytes,
		"utilization_limit_bytes_per_rank":        estimate.UtilizationLimitBytes,
		"runtime_reserve_bytes_per_rank":          estimate.RuntimeReserveBytes,
		"estimated_safe_kv_bytes_per_rank":        estimate.EstimatedSafeKVBytes,
		"fits":                                    estimate.Fits,
		"estimate_source":                         source, "estimate_is_advisory": true,
	} {
		metadata[key] = value
	}
	return metadata, nil
}

type mountSpec struct {
	source, target string
	readOnly       bool
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

func sparkMounts(topology *SparkRDMATopology, localModelPath *string, vllmArgv []string) []mountSpec {
	mounts := []mountSpec{
		{topology.RuntimeRepoRoot, topology.RuntimeRepoRoot, false},
		{topology.RuntimeB12XRoot, topology.RuntimeB12XRoot, false},
	}
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
	return mounts
}

// sparkDockerRun builds the docker run prefix shared by rank containers and
// the container-side preflight. A named container is created detached and
// kept after exit so its logs and exit code survive a crash; an unnamed one
// is ephemeral.
func sparkDockerRun(topology *SparkRDMATopology, mounts []mountSpec, environment map[string]string, name string, labels map[string]string) []string {
	argv := []string{"docker", "run", "--network", "host"}
	if name == "" {
		argv = append(argv, "--rm")
	} else {
		argv = append(argv, "--name", name, "--detach")
	}
	argv = append(argv,
		"--gpus", "all", "--cap-add=IPC_LOCK", "--device=/dev/infiniband",
		"--memory", fmt.Sprintf("%dg", topology.ContainerMemoryGB),
		"--memory-swap", fmt.Sprintf("%dg", topology.ContainerMemorySwapGB),
		"--shm-size", fmt.Sprintf("%dg", topology.ContainerShmGB),
		"--pids-limit", strconv.Itoa(topology.ContainerPidsLimit),
		"--ulimit", fmt.Sprintf("nofile=%d:%d", topology.ContainerNofileLimit, topology.ContainerNofileLimit),
		"--entrypoint", "",
	)
	for _, mount := range mounts {
		value := fmt.Sprintf("type=bind,src=%s,dst=%s", mount.source, mount.target)
		if mount.readOnly {
			value += ",readonly"
		}
		argv = append(argv, "--mount", value)
	}
	for _, key := range sortedMapKeys(labels) {
		argv = append(argv, "--label", key+"="+labels[key])
	}
	for _, key := range sortedMapKeys(environment) {
		argv = append(argv, "--env", key+"="+environment[key])
	}
	return append(argv, topology.Image)
}

func sparkNodeLaunches(topology *SparkRDMATopology, profile ModelProfile, runtimeEnvironment map[string]string, vllmArgv []string, localModelPath *string, tpSize int) []SparkNodeLaunch {
	containerName := sparkContainerName(topology, profile.Name, tpSize)
	mounts := sparkMounts(topology, localModelPath, vllmArgv)
	labels := map[string]string{
		"lil.model": profile.Model, "lil.tp": strconv.Itoa(tpSize),
	}
	if profile.ManifestCommit != "" {
		labels["lil.manifest_commit"] = profile.ManifestCommit
	}
	if profile.Revision != "" {
		labels["lil.revision"] = profile.Revision
	}
	selected := topology.Nodes[:tpSize]
	launches := make([]SparkNodeLaunch, 0, len(selected))
	for rank, node := range selected {
		environment := sparkNodeEnvironment(runtimeEnvironment, node)
		nodeVLLM := append([]string{}, vllmArgv...)
		nodeVLLM = append(nodeVLLM,
			"--nnodes", strconv.Itoa(len(selected)), "--node-rank", strconv.Itoa(rank),
			"--master-addr", selected[0].Address, "--master-port", strconv.Itoa(topology.MasterPort),
		)
		if rank > 0 {
			nodeVLLM = append(nodeVLLM, "--headless")
		}
		rankLabels := copyStringMap(labels)
		rankLabels["lil.rank"] = strconv.Itoa(rank)
		dockerArgv := sparkDockerRun(topology, mounts, environment, containerName, rankLabels)
		dockerArgv = append(dockerArgv, nodeVLLM...)
		launches = append(launches, SparkNodeLaunch{node, rank, containerName, environment, nodeVLLM, dockerArgv})
	}
	return launches
}

func sameArchitectures(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	seen := stringSet(left...)
	for _, value := range right {
		if !seen[value] {
			return false
		}
	}
	return true
}

func factsMetadata(facts *CheckpointFacts) map[string]any {
	return map[string]any{
		"source": facts.Source, "architectures": facts.Architectures,
		"attention_heads": facts.AttentionHeads, "weight_bytes": facts.WeightBytes,
		"weight_bytes_source": facts.WeightBytesSource,
	}
}

func BuildLaunchSpec(profile ModelProfile, topology Topology, options LaunchOptions) (LaunchSpec, error) {
	if err := validateOptions(options); err != nil {
		return LaunchSpec{}, err
	}
	if profile.Facts == nil {
		return LaunchSpec{}, fmt.Errorf("model %q has no checkpoint facts; config.json was not resolved", profile.Name)
	}
	if len(profile.Requires.Arch) > 0 && !contains(profile.Requires.Arch, topology.CuteDSLArch()) {
		return LaunchSpec{}, fmt.Errorf(
			"model %q requires %s; topology %q is %s",
			profile.Name, strings.Join(profile.Requires.Arch, " or "), topology.Name(), topology.CuteDSLArch(),
		)
	}
	facts := profile.Facts
	var localModelPath, checkpointPath *string
	modelSource := profile.Model
	if options.ModelPath != nil {
		resolved := resolveExistingPath(*options.ModelPath)
		localModelPath, checkpointPath, modelSource = &resolved, &resolved, resolved
		local, err := LoadCheckpointFacts(resolved)
		if err != nil {
			return LaunchSpec{}, err
		}
		if !sameArchitectures(local.Architectures, facts.Architectures) || local.AttentionHeads != facts.AttentionHeads {
			return LaunchSpec{}, fmt.Errorf(
				"checkpoint %s declares %v with %d heads, but %s declares %v with %d heads",
				resolved, local.Architectures, local.AttentionHeads,
				facts.Source, facts.Architectures, facts.AttentionHeads,
			)
		}
		facts = local
	}
	fitUtilization := topology.MemoryUtilization()
	if value, ok := profile.Base.Capacity["gpu_memory_utilization"].(float64); ok && value > 0 {
		fitUtilization = value
	}
	if options.GPUMemoryUtilization != nil {
		fitUtilization = *options.GPUMemoryUtilization
	}
	tpSize, defaultTP, err := resolveTPSize(facts, topology, options.TPSize, fitUtilization)
	if err != nil {
		return LaunchSpec{}, err
	}
	deviceIDs, err := resolveDevices(topology, options, tpSize)
	if err != nil {
		return LaunchSpec{}, err
	}
	servedModelName := profile.Serving.ServedModelName
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
	spec, err := speculativeConfig(profile, facts, modelSource, options)
	if err != nil {
		return LaunchSpec{}, err
	}
	settings, appliedOverrides, err := resolveLaunchSettings(profile, topology, tpSize, spec.Effective())
	if err != nil {
		return LaunchSpec{}, err
	}
	capacity := resolveCapacity(settings, topology, options)
	needsNVRTC := spec.Decision != nil && spec.Decision.Backend == "humming"
	environment, unsetEnvironment, err := runtimeEnvironment(topology, settings, options, needsNVRTC)
	if err != nil {
		return LaunchSpec{}, err
	}
	kvCacheDType := profile.Kernels.KVCacheDType
	if options.KVCacheDType != nil {
		kvCacheDType = *options.KVCacheDType
	}
	vllmArgv, err := buildVLLMArgv(profile, topology, options, argvInputs{
		tpSize: tpSize, deviceIDs: deviceIDs, modelSource: modelSource,
		servedModelName: servedModelName, host: host, port: port,
		capacity: capacity, settings: settings, speculation: spec, kvCacheDType: kvCacheDType,
	})
	if err != nil {
		return LaunchSpec{}, err
	}
	memory, err := memoryMetadata(facts, topology, checkpointPath, environment, capacity, tpSize)
	if err != nil {
		return LaunchSpec{}, err
	}
	metadata := map[string]any{
		"capacity": capacity, "speculator": spec.Effective(), "speculative_tokens": spec.Tokens,
		"kv_cache_dtype": kvCacheDType, "memory": memory,
		"manifest_commit": profile.ManifestCommit, "family": profile.Family,
		"repository": profile.Model, "revision": profile.Revision,
		"checkpoint_facts": factsMetadata(facts), "applied_overrides": appliedOverrides,
		"tensor_parallel": map[string]any{
			"size": tpSize, "default": defaultTP, "policy": topology.DefaultTP,
			"supported": SupportedTPSizes(facts, topology),
		},
	}
	if spec.Decision != nil {
		metadata["mtp_moe"] = map[string]any{
			"quantization": spec.Decision.Quantization, "backend": spec.Decision.Backend,
			"evidence": spec.Decision.Evidence,
		}
	}
	downloads := []RepositoryDownload{}
	if checkpointPath == nil {
		downloads = append(downloads, RepositoryDownload{Repository: profile.Model, Revision: profile.Revision})
	}
	if spec.Effective() == "dflash" && profile.Speculators.DFlash != nil &&
		profile.Speculators.DFlash.Model != profile.Model {
		downloads = append(downloads, RepositoryDownload{Repository: profile.Speculators.DFlash.Model})
	}
	launch := LaunchSpec{
		Model: profile, Topology: topology, TPSize: tpSize, ModelSource: modelSource,
		CheckpointPath: checkpointPath, ServedModelName: servedModelName, Host: host, Port: port,
		Detach: options.Detach, SyncCode: options.SyncCode, SyncModel: options.SyncModel,
		Downloads: downloads, RuntimeEnvironment: environment,
		UnsetEnvironment: unsetEnvironment, VLLMArgv: vllmArgv, DeviceIDs: deviceIDs,
		Metadata: metadata,
	}
	if topology.Spark != nil {
		launch.SparkNodes = sparkNodeLaunches(topology.Spark, profile, environment, vllmArgv, localModelPath, tpSize)
		launch.HostEnvironment = map[string]string{}
		launch.CommandArgv = append([]string{}, launch.SparkNodes[0].DockerArgv...)
	} else {
		launch.HostEnvironment = copyStringMap(environment)
		launch.CommandArgv = append([]string{}, vllmArgv...)
	}
	return launch, nil
}
