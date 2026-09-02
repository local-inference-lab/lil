// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package launcher

import (
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

var hfModelID = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$`)
var hfCommit = regexp.MustCompile(`^[0-9a-f]{40}$`)
var hfRepositoryName = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.-]*$`)
var environmentName = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

// Launcher-wide defaults that any manifest section may override.
const (
	defaultDType      = "bfloat16"
	defaultLinear     = "b12x"
	defaultMoE        = "b12x"
	defaultLoadFormat = "instanttensor"
	defaultMTPTokens  = 3
)

var capacityKeys = []string{
	"kv_cache_memory_bytes", "max_model_len", "max_num_seqs", "max_num_batched_tokens",
}

var speculatorMethods = stringSet("mtp", "dflash", "dspark", "none")

var manifestSections = []string{
	"description", "serving", "kernels", "speculators", "capacity",
	"compilation", "environment", "requires", "overrides",
}

func normalizeYAML(value any) (any, error) {
	switch item := value.(type) {
	case map[string]any:
		result := make(map[string]any, len(item))
		for key, child := range item {
			normalized, err := normalizeYAML(child)
			if err != nil {
				return nil, err
			}
			result[key] = normalized
		}
		return result, nil
	case map[any]any:
		result := make(map[string]any, len(item))
		for key, child := range item {
			var name string
			switch typed := key.(type) {
			case string:
				name = typed
			case int:
				name = strconv.Itoa(typed)
			default:
				return nil, fmt.Errorf("unsupported YAML mapping key %v", key)
			}
			normalized, err := normalizeYAML(child)
			if err != nil {
				return nil, err
			}
			result[name] = normalized
		}
		return result, nil
	case []any:
		result := make([]any, len(item))
		for index, child := range item {
			normalized, err := normalizeYAML(child)
			if err != nil {
				return nil, err
			}
			result[index] = normalized
		}
		return result, nil
	default:
		return value, nil
	}
}

func parseYAMLDocument(data []byte, label string) (map[string]any, error) {
	var loaded any
	if err := yaml.Unmarshal(data, &loaded); err != nil {
		return nil, fmt.Errorf("invalid YAML in %s: %w", label, err)
	}
	normalized, err := normalizeYAML(loaded)
	if err != nil {
		return nil, err
	}
	return mapping(normalized, label)
}

func mapping(value any, context string) (map[string]any, error) {
	result, ok := value.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("%s must be a string-keyed mapping", context)
	}
	return result, nil
}

func checkKeys(data map[string]any, context string, required, optional []string) error {
	allowed := map[string]bool{}
	for _, key := range required {
		allowed[key] = true
		if _, ok := data[key]; !ok {
			return fmt.Errorf("%s is missing required key: %s", context, key)
		}
	}
	for _, key := range optional {
		allowed[key] = true
	}
	unknown := []string{}
	for key := range data {
		if !allowed[key] {
			unknown = append(unknown, key)
		}
	}
	if len(unknown) > 0 {
		sort.Strings(unknown)
		return fmt.Errorf("%s contains unknown keys: %s", context, strings.Join(unknown, ", "))
	}
	return nil
}

func cloneValue(value any) any {
	switch item := value.(type) {
	case map[string]any:
		result := make(map[string]any, len(item))
		for key, child := range item {
			result[key] = cloneValue(child)
		}
		return result
	case []any:
		result := make([]any, len(item))
		for index, child := range item {
			result[index] = cloneValue(child)
		}
		return result
	default:
		return value
	}
}

// deepMerge overlays a mapping onto a base. Nested mappings merge; scalars,
// lists, and explicit nulls replace the inherited value.
func deepMerge(base, overlay map[string]any) map[string]any {
	merged := cloneValue(base).(map[string]any)
	for key, value := range overlay {
		inherited, inheritedOK := merged[key].(map[string]any)
		child, childOK := value.(map[string]any)
		if inheritedOK && childOK {
			merged[key] = deepMerge(inherited, child)
		} else {
			merged[key] = cloneValue(value)
		}
	}
	return merged
}

func schemaVersionOf(document map[string]any, label string) error {
	version, ok := document["schema_version"].(int)
	if !ok || version != schemaVersion {
		return fmt.Errorf(
			"%s.schema_version must be %d; got %v",
			label, schemaVersion, document["schema_version"],
		)
	}
	return nil
}

// LoadFamilies parses the embedded family bases document.
func LoadFamilies(data []byte, label string) (map[string]map[string]any, error) {
	document, err := parseYAMLDocument(data, label)
	if err != nil {
		return nil, err
	}
	if err := checkKeys(document, label, []string{"schema_version"}, []string{"families"}); err != nil {
		return nil, err
	}
	if err := schemaVersionOf(document, label); err != nil {
		return nil, err
	}
	families := map[string]map[string]any{}
	if raw, ok := document["families"]; ok && raw != nil {
		table, err := mapping(raw, label+".families")
		if err != nil {
			return nil, err
		}
		for name, value := range table {
			family, err := mapping(value, label+".families."+name)
			if err != nil {
				return nil, err
			}
			if err := checkKeys(family, label+".families."+name, nil, append([]string{"family"}, manifestSections...)); err != nil {
				return nil, err
			}
			families[name] = family
		}
	}
	return families, nil
}

func resolveFamily(families map[string]map[string]any, name string, visiting []string) (map[string]any, error) {
	for index, active := range visiting {
		if active == name {
			cycle := append(append([]string{}, visiting[index:]...), name)
			return nil, fmt.Errorf("family inheritance cycle: %s", strings.Join(cycle, " -> "))
		}
	}
	raw, ok := families[name]
	if !ok {
		return nil, fmt.Errorf("unknown model family %q", name)
	}
	resolved := map[string]any{}
	if parent, exists := raw["family"]; exists && parent != nil {
		parentName, ok := parent.(string)
		if !ok || parentName == "" {
			return nil, fmt.Errorf("family %q has an invalid parent", name)
		}
		var err error
		resolved, err = resolveFamily(families, parentName, append(visiting, name))
		if err != nil {
			return nil, err
		}
	}
	overlay := cloneValue(raw).(map[string]any)
	delete(overlay, "family")
	return deepMerge(resolved, overlay), nil
}

type section struct {
	data    map[string]any
	context string
}

func subsection(parent map[string]any, key, context string) (section, bool, error) {
	raw, ok := parent[key]
	if !ok || raw == nil {
		return section{}, false, nil
	}
	data, err := mapping(raw, context+"."+key)
	if err != nil {
		return section{}, false, err
	}
	return section{data, context + "." + key}, true, nil
}

func (s section) present(key string) bool {
	value, ok := s.data[key]
	return ok && value != nil
}

func (s section) optionalString(key string) (*string, error) {
	if !s.present(key) {
		return nil, nil
	}
	value, ok := s.data[key].(string)
	if !ok || value == "" {
		return nil, fmt.Errorf("%s.%s must be a non-empty string or null", s.context, key)
	}
	return &value, nil
}

func (s section) stringOr(key, fallback string) (string, error) {
	value, err := s.optionalString(key)
	if err != nil || value == nil {
		return fallback, err
	}
	return *value, nil
}

func (s section) boolOr(key string, fallback bool) (bool, error) {
	if !s.present(key) {
		return fallback, nil
	}
	value, ok := s.data[key].(bool)
	if !ok {
		return false, fmt.Errorf("%s.%s must be a boolean", s.context, key)
	}
	return value, nil
}

func (s section) optionalBool(key string) (*bool, error) {
	if !s.present(key) {
		return nil, nil
	}
	value, ok := s.data[key].(bool)
	if !ok {
		return nil, fmt.Errorf("%s.%s must be a boolean or null", s.context, key)
	}
	return &value, nil
}

func (s section) optionalPositiveInt(key string) (*int, error) {
	if !s.present(key) {
		return nil, nil
	}
	value, ok := s.data[key].(int)
	if !ok || value <= 0 {
		return nil, fmt.Errorf("%s.%s must be a positive integer or null", s.context, key)
	}
	return &value, nil
}

func (s section) optionalNonNegativeInt(key string) (*int, error) {
	if !s.present(key) {
		return nil, nil
	}
	value, ok := s.data[key].(int)
	if !ok || value < 0 {
		return nil, fmt.Errorf("%s.%s must be a non-negative integer or null", s.context, key)
	}
	return &value, nil
}

func (s section) optionalNonNegativeFloat(key string) (*float64, error) {
	if !s.present(key) {
		return nil, nil
	}
	var value float64
	switch typed := s.data[key].(type) {
	case int:
		value = float64(typed)
	case float64:
		value = typed
	default:
		return nil, fmt.Errorf("%s.%s must be a number or null", s.context, key)
	}
	if value < 0 {
		return nil, fmt.Errorf("%s.%s must be non-negative", s.context, key)
	}
	return &value, nil
}

func (s section) optionalMapping(key string) (map[string]any, error) {
	if !s.present(key) {
		return nil, nil
	}
	value, err := mapping(s.data[key], s.context+"."+key)
	if err != nil {
		return nil, err
	}
	return cloneValue(value).(map[string]any), nil
}

func (s section) optionalStringList(key string) ([]string, error) {
	if !s.present(key) {
		return nil, nil
	}
	items, ok := s.data[key].([]any)
	if !ok || len(items) == 0 {
		return nil, fmt.Errorf("%s.%s must be a non-empty list of strings", s.context, key)
	}
	result := make([]string, 0, len(items))
	for _, item := range items {
		value, ok := item.(string)
		if !ok || value == "" {
			return nil, fmt.Errorf("%s.%s must be a non-empty list of strings", s.context, key)
		}
		result = append(result, value)
	}
	return result, nil
}

func parseEnvironmentMap(raw any, context string) (map[string]string, error) {
	if raw == nil {
		return nil, nil
	}
	data, err := mapping(raw, context)
	if err != nil {
		return nil, err
	}
	environment := make(map[string]string, len(data))
	for key, value := range data {
		text, ok := value.(string)
		if !ok {
			return nil, fmt.Errorf("%s values must be strings; quote %s", context, key)
		}
		if !environmentName.MatchString(key) {
			return nil, fmt.Errorf("%s contains an invalid variable name: %q", context, key)
		}
		if derivedEnvironment[key] {
			return nil, fmt.Errorf("%s.%s is derived by the launcher and cannot be configured", context, key)
		}
		environment[key] = text
	}
	return environment, nil
}

func parseCapacityLayer(raw any, context string) (map[string]any, error) {
	if raw == nil {
		return nil, nil
	}
	data, err := mapping(raw, context)
	if err != nil {
		return nil, err
	}
	if err := checkKeys(data, context, nil, capacityKeys); err != nil {
		return nil, err
	}
	return cloneValue(data).(map[string]any), nil
}

func parseLaunchLayer(data map[string]any, context string) (LaunchLayer, error) {
	capacity, err := parseCapacityLayer(data["capacity"], context+".capacity")
	if err != nil {
		return LaunchLayer{}, err
	}
	var compilation map[string]any
	if raw, ok := data["compilation"]; ok && raw != nil {
		compilation, err = mapping(raw, context+".compilation")
		if err != nil {
			return LaunchLayer{}, err
		}
		compilation = cloneValue(compilation).(map[string]any)
	}
	environment, err := parseEnvironmentMap(data["environment"], context+".environment")
	if err != nil {
		return LaunchLayer{}, err
	}
	return LaunchLayer{Capacity: capacity, Compilation: compilation, Environment: environment}, nil
}

func parseCapacity(value any, context string) (Capacity, error) {
	data, err := mapping(value, context)
	if err != nil {
		return Capacity{}, err
	}
	if err := checkKeys(data, context, capacityKeys, nil); err != nil {
		return Capacity{}, err
	}
	var kvCache *string
	if data["kv_cache_memory_bytes"] != nil {
		value, ok := data["kv_cache_memory_bytes"].(string)
		if !ok {
			return Capacity{}, fmt.Errorf("%s.kv_cache_memory_bytes must be a string or null", context)
		}
		kvCache = &value
	}
	maxModelLen, ok := data["max_model_len"].(string)
	if !ok || maxModelLen == "" {
		return Capacity{}, fmt.Errorf("%s.max_model_len must be a string", context)
	}
	maxSeqs, ok := data["max_num_seqs"].(int)
	if !ok || maxSeqs <= 0 {
		return Capacity{}, fmt.Errorf("%s.max_num_seqs must be a positive integer", context)
	}
	maxTokens, ok := data["max_num_batched_tokens"].(int)
	if !ok || maxTokens <= 0 {
		return Capacity{}, fmt.Errorf("%s.max_num_batched_tokens must be a positive integer", context)
	}
	return Capacity{nil, kvCache, maxModelLen, maxSeqs, maxTokens}, nil
}

func parseServing(data map[string]any, context string) (ServingPolicy, error) {
	s, ok, err := subsection(data, "serving", context)
	if err != nil {
		return ServingPolicy{}, err
	}
	if !ok {
		return ServingPolicy{}, fmt.Errorf("%s.serving is required", context)
	}
	if err := checkKeys(s.data, s.context, []string{"served_model_name"}, []string{
		"trust_remote_code", "tokenizer_mode", "reasoning_parser", "tool_call_parser",
		"auto_tool_choice", "generation_config", "hf_overrides", "chat_template_kwargs",
		"async_scheduling", "scheduler_reserve_full_isl", "prefix_caching",
		"prefix_cache_retention_interval", "chunked_prefill",
		"long_prefill_token_threshold", "prompt_tokens_details", "force_include_usage",
		"request_id_headers", "multimodal",
	}); err != nil {
		return ServingPolicy{}, err
	}
	var policy ServingPolicy
	if policy.ServedModelName, err = s.stringOr("served_model_name", ""); err != nil {
		return policy, err
	}
	if policy.ServedModelName == "" {
		return policy, fmt.Errorf("%s.served_model_name must be a non-empty string", s.context)
	}
	if policy.TrustRemoteCode, err = s.boolOr("trust_remote_code", false); err != nil {
		return policy, err
	}
	if policy.TokenizerMode, err = s.stringOr("tokenizer_mode", ""); err != nil {
		return policy, err
	}
	if policy.ReasoningParser, err = s.stringOr("reasoning_parser", ""); err != nil {
		return policy, err
	}
	if policy.ToolCallParser, err = s.stringOr("tool_call_parser", ""); err != nil {
		return policy, err
	}
	if policy.AutoToolChoice, err = s.boolOr("auto_tool_choice", policy.ToolCallParser != ""); err != nil {
		return policy, err
	}
	if policy.AutoToolChoice && policy.ToolCallParser == "" {
		return policy, fmt.Errorf("%s.auto_tool_choice requires tool_call_parser", s.context)
	}
	if policy.GenerationConfig, err = s.optionalString("generation_config"); err != nil {
		return policy, err
	}
	if policy.HFOverrides, err = s.optionalMapping("hf_overrides"); err != nil {
		return policy, err
	}
	if policy.ChatTemplateKwargs, err = s.optionalMapping("chat_template_kwargs"); err != nil {
		return policy, err
	}
	for key, value := range policy.ChatTemplateKwargs {
		switch value.(type) {
		case string, bool, int, float64:
		default:
			return policy, fmt.Errorf("%s.chat_template_kwargs.%s must be a string, boolean, or number", s.context, key)
		}
	}
	if policy.AsyncScheduling, err = s.boolOr("async_scheduling", false); err != nil {
		return policy, err
	}
	if policy.SchedulerReserveFullISL, err = s.boolOr("scheduler_reserve_full_isl", true); err != nil {
		return policy, err
	}
	if policy.PrefixCaching, err = s.boolOr("prefix_caching", true); err != nil {
		return policy, err
	}
	if policy.PrefixCacheRetentionInterval, err = s.optionalNonNegativeInt("prefix_cache_retention_interval"); err != nil {
		return policy, err
	}
	if policy.ChunkedPrefill, err = s.boolOr("chunked_prefill", true); err != nil {
		return policy, err
	}
	if policy.LongPrefillTokenThreshold, err = s.optionalPositiveInt("long_prefill_token_threshold"); err != nil {
		return policy, err
	}
	if policy.PromptTokensDetails, err = s.boolOr("prompt_tokens_details", false); err != nil {
		return policy, err
	}
	if policy.ForceIncludeUsage, err = s.boolOr("force_include_usage", false); err != nil {
		return policy, err
	}
	if policy.RequestIDHeaders, err = s.boolOr("request_id_headers", false); err != nil {
		return policy, err
	}
	multimodal, ok, err := subsection(s.data, "multimodal", s.context)
	if err != nil {
		return policy, err
	}
	if ok {
		if err := checkKeys(multimodal.data, multimodal.context, nil, []string{
			"encoder_tp_mode", "processor_cache_gb", "limit_per_prompt",
		}); err != nil {
			return policy, err
		}
		var mm MultimodalPolicy
		if mm.EncoderTPMode, err = multimodal.optionalString("encoder_tp_mode"); err != nil {
			return policy, err
		}
		if mm.ProcessorCacheGB, err = multimodal.optionalNonNegativeFloat("processor_cache_gb"); err != nil {
			return policy, err
		}
		if multimodal.present("limit_per_prompt") {
			limits, err := mapping(multimodal.data["limit_per_prompt"], multimodal.context+".limit_per_prompt")
			if err != nil {
				return policy, err
			}
			mm.LimitPerPrompt = map[string]int{}
			for modality, raw := range limits {
				count, ok := raw.(int)
				if !ok || count < 0 {
					return policy, fmt.Errorf("%s.limit_per_prompt.%s must be a non-negative integer", multimodal.context, modality)
				}
				mm.LimitPerPrompt[modality] = count
			}
		}
		policy.Multimodal = &mm
	}
	return policy, nil
}

func parseKernels(data map[string]any, context string) (KernelPolicy, error) {
	policy := KernelPolicy{
		DType: defaultDType, Linear: defaultLinear, MoE: defaultMoE,
		LoadFormat: defaultLoadFormat,
	}
	s, ok, err := subsection(data, "kernels", context)
	if err != nil || !ok {
		return policy, err
	}
	if err := checkKeys(s.data, s.context, nil, []string{
		"dtype", "quantization", "attention", "linear", "moe", "gdn_decode",
		"block_size", "mamba_cache_mode", "flashinfer_autotune", "load_format",
		"loader_extra_config",
	}); err != nil {
		return policy, err
	}
	if policy.DType, err = s.stringOr("dtype", defaultDType); err != nil {
		return policy, err
	}
	if policy.Quantization, err = s.optionalString("quantization"); err != nil {
		return policy, err
	}
	if policy.Attention, err = s.optionalString("attention"); err != nil {
		return policy, err
	}
	if policy.Linear, err = s.stringOr("linear", defaultLinear); err != nil {
		return policy, err
	}
	if policy.MoE, err = s.stringOr("moe", defaultMoE); err != nil {
		return policy, err
	}
	if policy.GDNDecode, err = s.optionalString("gdn_decode"); err != nil {
		return policy, err
	}
	if policy.BlockSize, err = s.optionalPositiveInt("block_size"); err != nil {
		return policy, err
	}
	if policy.MambaCacheMode, err = s.optionalString("mamba_cache_mode"); err != nil {
		return policy, err
	}
	if policy.FlashinferAutotune, err = s.optionalBool("flashinfer_autotune"); err != nil {
		return policy, err
	}
	if policy.LoadFormat, err = s.stringOr("load_format", defaultLoadFormat); err != nil {
		return policy, err
	}
	if policy.LoaderExtraConfig, err = s.optionalMapping("loader_extra_config"); err != nil {
		return policy, err
	}
	return policy, nil
}

func parseSpeculators(data map[string]any, context string) (SpeculatorPolicy, error) {
	policy := SpeculatorPolicy{Default: "none"}
	s, ok, err := subsection(data, "speculators", context)
	if err != nil || !ok {
		return policy, err
	}
	if err := checkKeys(s.data, s.context, nil, []string{"default", "mtp", "dflash", "dspark"}); err != nil {
		return policy, err
	}
	if policy.Default, err = s.stringOr("default", "none"); err != nil {
		return policy, err
	}
	if !speculatorMethods[policy.Default] {
		return policy, fmt.Errorf("%s.default must be mtp, dflash, dspark, or none", s.context)
	}
	mtp, ok, err := subsection(s.data, "mtp", s.context)
	if err != nil {
		return policy, err
	}
	if ok {
		if err := checkKeys(mtp.data, mtp.context, nil, []string{
			"tokens", "moe_quantization", "moe_backend", "attention",
			"draft_sample_method", "rejection_sample_method", "model",
		}); err != nil {
			return policy, err
		}
		settings := MTPPolicy{Tokens: defaultMTPTokens}
		if tokens, err := mtp.optionalNonNegativeInt("tokens"); err != nil {
			return policy, err
		} else if tokens != nil {
			settings.Tokens = *tokens
		}
		if settings.MoEQuantization, err = mtp.stringOr("moe_quantization", ""); err != nil {
			return policy, err
		}
		if settings.MoEQuantization != "" {
			if _, err := MTPMoEBackendFromQuantization(settings.MoEQuantization, mtp.context); err != nil {
				return policy, fmt.Errorf("%s.moe_quantization must be nvfp4, mxfp8, or bf16", mtp.context)
			}
		}
		if settings.MoEBackend, err = mtp.optionalString("moe_backend"); err != nil {
			return policy, err
		}
		if settings.Attention, err = mtp.optionalString("attention"); err != nil {
			return policy, err
		}
		if settings.DraftSampleMethod, err = mtp.optionalString("draft_sample_method"); err != nil {
			return policy, err
		}
		if settings.RejectionSampleMethod, err = mtp.optionalString("rejection_sample_method"); err != nil {
			return policy, err
		}
		model, err := mtp.optionalString("model")
		if err != nil {
			return policy, err
		}
		if model != nil {
			if *model != "target" {
				return policy, fmt.Errorf("%s.model must be target or null", mtp.context)
			}
			settings.WeightsInTarget = true
		}
		policy.MTP = &settings
	}
	dflash, ok, err := subsection(s.data, "dflash", s.context)
	if err != nil {
		return policy, err
	}
	if ok {
		if err := checkKeys(dflash.data, dflash.context, []string{"tokens", "model"}, nil); err != nil {
			return policy, err
		}
		tokens, err := dflash.optionalNonNegativeInt("tokens")
		if err != nil {
			return policy, err
		}
		if tokens == nil {
			return policy, fmt.Errorf("%s.tokens must be a non-negative integer", dflash.context)
		}
		model, err := dflash.stringOr("model", "")
		if err != nil {
			return policy, err
		}
		if !hfModelID.MatchString(model) {
			return policy, fmt.Errorf("%s.model must be a Hugging Face model ID", dflash.context)
		}
		policy.DFlash = &DFlashPolicy{Tokens: *tokens, Model: model}
	}
	dspark, ok, err := subsection(s.data, "dspark", s.context)
	if err != nil {
		return policy, err
	}
	if ok {
		if err := checkKeys(dspark.data, dspark.context, []string{"tokens", "model"}, []string{
			"attention", "draft_sample_method", "rejection_sample_method", "adaptive_verification",
		}); err != nil {
			return policy, err
		}
		tokens, err := dspark.optionalNonNegativeInt("tokens")
		if err != nil {
			return policy, err
		}
		if tokens == nil {
			return policy, fmt.Errorf("%s.tokens must be a non-negative integer", dspark.context)
		}
		model, err := dspark.stringOr("model", "")
		if err != nil {
			return policy, err
		}
		if model != "target" {
			return policy, fmt.Errorf("%s.model must be target; DSpark drafts ship inside the target checkpoint", dspark.context)
		}
		settings := DSparkPolicy{Tokens: *tokens}
		if settings.Attention, err = dspark.optionalString("attention"); err != nil {
			return policy, err
		}
		if settings.DraftSampleMethod, err = dspark.optionalString("draft_sample_method"); err != nil {
			return policy, err
		}
		if settings.RejectionSampleMethod, err = dspark.optionalString("rejection_sample_method"); err != nil {
			return policy, err
		}
		if settings.AdaptiveVerification, err = dspark.boolOr("adaptive_verification", false); err != nil {
			return policy, err
		}
		policy.DSpark = &settings
	}
	if policy.Default == "mtp" && policy.MTP == nil {
		return policy, fmt.Errorf("%s.default is mtp but no mtp section is defined", s.context)
	}
	if policy.Default == "dflash" && policy.DFlash == nil {
		return policy, fmt.Errorf("%s.default is dflash but no dflash section is defined", s.context)
	}
	if policy.Default == "dspark" && policy.DSpark == nil {
		return policy, fmt.Errorf("%s.default is dspark but no dspark section is defined", s.context)
	}
	return policy, nil
}

func parseRequirements(data map[string]any, context string) (Requirements, error) {
	s, ok, err := subsection(data, "requires", context)
	if err != nil || !ok {
		return Requirements{}, err
	}
	if err := checkKeys(s.data, s.context, nil, []string{"arch"}); err != nil {
		return Requirements{}, err
	}
	arch, err := s.optionalStringList("arch")
	if err != nil {
		return Requirements{}, err
	}
	return Requirements{Arch: arch}, nil
}

func parseOverrides(data map[string]any, context string) ([]LaunchOverride, error) {
	raw, ok := data["overrides"]
	if !ok || raw == nil {
		return nil, nil
	}
	items, ok := raw.([]any)
	if !ok {
		return nil, fmt.Errorf("%s.overrides must be a list", context)
	}
	overrides := make([]LaunchOverride, 0, len(items))
	for index, item := range items {
		itemContext := fmt.Sprintf("%s.overrides[%d]", context, index)
		entry, err := mapping(item, itemContext)
		if err != nil {
			return nil, err
		}
		if err := checkKeys(entry, itemContext, []string{"when"}, []string{"capacity", "compilation", "environment"}); err != nil {
			return nil, err
		}
		when, err := mapping(entry["when"], itemContext+".when")
		if err != nil {
			return nil, err
		}
		if err := checkKeys(when, itemContext+".when", nil, []string{"kind", "arch", "tp", "speculator"}); err != nil {
			return nil, err
		}
		if len(when) == 0 {
			return nil, fmt.Errorf("%s.when must name at least one condition", itemContext)
		}
		condition := section{when, itemContext + ".when"}
		var override LaunchOverride
		if override.When.Kind, err = condition.stringOr("kind", ""); err != nil {
			return nil, err
		}
		if override.When.Kind != "" && override.When.Kind != "local" && override.When.Kind != "spark_rdma" {
			return nil, fmt.Errorf("%s.when.kind must be local or spark_rdma", itemContext)
		}
		if override.When.Arch, err = condition.stringOr("arch", ""); err != nil {
			return nil, err
		}
		tp, err := condition.optionalPositiveInt("tp")
		if err != nil {
			return nil, err
		}
		if tp != nil {
			override.When.TP = *tp
		}
		if override.When.Speculator, err = condition.stringOr("speculator", ""); err != nil {
			return nil, err
		}
		if override.When.Speculator != "" && !speculatorMethods[override.When.Speculator] {
			return nil, fmt.Errorf("%s.when.speculator must be mtp, dflash, dspark, or none", itemContext)
		}
		if override.Layer, err = parseLaunchLayer(entry, itemContext); err != nil {
			return nil, err
		}
		overrides = append(overrides, override)
	}
	return overrides, nil
}

func decodeProfile(name string, document map[string]any, context string) (ModelProfile, error) {
	if err := checkKeys(document, context, []string{"description", "serving"}, manifestSections); err != nil {
		return ModelProfile{}, err
	}
	root := section{document, context}
	description, err := root.stringOr("description", "")
	if err != nil {
		return ModelProfile{}, err
	}
	if description == "" {
		return ModelProfile{}, fmt.Errorf("%s.description must be a non-empty string", context)
	}
	serving, err := parseServing(document, context)
	if err != nil {
		return ModelProfile{}, err
	}
	kernels, err := parseKernels(document, context)
	if err != nil {
		return ModelProfile{}, err
	}
	speculators, err := parseSpeculators(document, context)
	if err != nil {
		return ModelProfile{}, err
	}
	base, err := parseLaunchLayer(document, context)
	if err != nil {
		return ModelProfile{}, err
	}
	requires, err := parseRequirements(document, context)
	if err != nil {
		return ModelProfile{}, err
	}
	overrides, err := parseOverrides(document, context)
	if err != nil {
		return ModelProfile{}, err
	}
	return ModelProfile{
		Name: name, Description: description, Serving: serving, Kernels: kernels,
		Speculators: speculators, Base: base, Overrides: overrides, Requires: requires,
	}, nil
}

func loadFamilies(familiesData []byte) (map[string]map[string]any, error) {
	return LoadFamilies(familiesData, "embedded model families")
}

// resolveManifestDocument validates the manifest envelope and merges the
// named family under the manifest body. It returns the merged body and the
// family name.
func resolveManifestDocument(families map[string]map[string]any, manifestData []byte, label string, extraKeys []string) (map[string]any, map[string]any, string, error) {
	document, err := parseYAMLDocument(manifestData, label)
	if err != nil {
		return nil, nil, "", err
	}
	if err := schemaVersionOf(document, label); err != nil {
		return nil, nil, "", err
	}
	kind, ok := document["kind"].(string)
	if !ok || kind != "model" {
		return nil, nil, "", fmt.Errorf("%s.kind must be model", label)
	}
	optional := append([]string{"family"}, manifestSections...)
	optional = append(optional, extraKeys...)
	if err := checkKeys(document, label, []string{"schema_version", "kind"}, optional); err != nil {
		return nil, nil, "", err
	}
	resolved := map[string]any{}
	familyName := ""
	if raw, exists := document["family"]; exists && raw != nil {
		familyName, ok = raw.(string)
		if !ok || familyName == "" {
			return nil, nil, "", fmt.Errorf("%s.family must be a family name or null", label)
		}
		resolved, err = resolveFamily(families, familyName, nil)
		if err != nil {
			return nil, nil, "", fmt.Errorf("%s: %w", label, err)
		}
	}
	overlay := cloneValue(document).(map[string]any)
	for _, key := range append([]string{"schema_version", "kind", "family"}, extraKeys...) {
		delete(overlay, key)
	}
	return document, deepMerge(resolved, overlay), familyName, nil
}

// LoadRepositoryModelProfile parses a manifest stored at the root of the
// model's own repository. The repository ID is the model identity; the
// manifest may not restate it or pin a revision.
func LoadRepositoryModelProfile(
	familiesData, manifestData []byte,
	repositoryID, manifestCommit, label string,
) (ModelProfile, error) {
	if !hfModelID.MatchString(repositoryID) {
		return ModelProfile{}, fmt.Errorf("repository ID must have owner/name form; got %q", repositoryID)
	}
	if manifestCommit != "" && !hfCommit.MatchString(manifestCommit) {
		return ModelProfile{}, fmt.Errorf("manifest commit must be a 40-character SHA; got %q", manifestCommit)
	}
	families, err := loadFamilies(familiesData)
	if err != nil {
		return ModelProfile{}, err
	}
	document, err := parseYAMLDocument(manifestData, label)
	if err != nil {
		return ModelProfile{}, err
	}
	for _, key := range []string{"model", "revision"} {
		if _, exists := document[key]; exists {
			return ModelProfile{}, fmt.Errorf("%s must not set %s; the repository is authoritative", label, key)
		}
	}
	_, resolved, familyName, err := resolveManifestDocument(families, manifestData, label, nil)
	if err != nil {
		return ModelProfile{}, err
	}
	_, name, _ := strings.Cut(repositoryID, "/")
	profile, err := decodeProfile(name, resolved, label)
	if err != nil {
		return ModelProfile{}, err
	}
	profile.Model = repositoryID
	profile.ManifestCommit = manifestCommit
	profile.Family = familyName
	return profile, nil
}

// LoadCatalogModelProfile parses a manifest stored in the catalog
// repository for a model whose weights live elsewhere. The manifest names
// the upstream repository and, when the model runs remote code, pins its
// commit.
func LoadCatalogModelProfile(
	familiesData, manifestData []byte,
	name, catalogCommit, label string,
) (ModelProfile, error) {
	if !hfRepositoryName.MatchString(name) {
		return ModelProfile{}, fmt.Errorf("catalog entry name %q is not a valid repository name", name)
	}
	if catalogCommit != "" && !hfCommit.MatchString(catalogCommit) {
		return ModelProfile{}, fmt.Errorf("manifest commit must be a 40-character SHA; got %q", catalogCommit)
	}
	families, err := loadFamilies(familiesData)
	if err != nil {
		return ModelProfile{}, err
	}
	document, resolved, familyName, err := resolveManifestDocument(families, manifestData, label, []string{"model", "revision"})
	if err != nil {
		return ModelProfile{}, err
	}
	root := section{document, label}
	model, err := root.stringOr("model", "")
	if err != nil {
		return ModelProfile{}, err
	}
	if !hfModelID.MatchString(model) {
		return ModelProfile{}, fmt.Errorf("%s.model must name the upstream repository in owner/name form", label)
	}
	revision, err := root.stringOr("revision", "")
	if err != nil {
		return ModelProfile{}, err
	}
	if revision != "" && !hfCommit.MatchString(revision) {
		return ModelProfile{}, fmt.Errorf("%s.revision must be a 40-character commit SHA", label)
	}
	profile, err := decodeProfile(name, resolved, label)
	if err != nil {
		return ModelProfile{}, err
	}
	if profile.Serving.TrustRemoteCode && revision == "" {
		return ModelProfile{}, fmt.Errorf(
			"%s enables trust_remote_code for an external repository and must pin a revision", label,
		)
	}
	profile.Model = model
	profile.Revision = revision
	profile.ManifestCommit = catalogCommit
	profile.Family = familyName
	return profile, nil
}

// ManifestNamesModel reports whether a manifest body carries a model field,
// which distinguishes a catalog entry from a repository-rooted manifest.
func ManifestNamesModel(manifestData []byte, label string) (bool, error) {
	document, err := parseYAMLDocument(manifestData, label)
	if err != nil {
		return false, err
	}
	_, ok := document["model"]
	return ok, nil
}

type DraftProfile struct {
	RepositoryID     string
	ManifestCommit   string
	Description      string
	Method           string
	Quantization     string
	CompatibleModels []string
}

func RepositoryManifestKind(data []byte, label string) (string, error) {
	document, err := parseYAMLDocument(data, label)
	if err != nil {
		return "", err
	}
	if err := schemaVersionOf(document, label); err != nil {
		return "", err
	}
	kind, ok := document["kind"].(string)
	if !ok || kind != "model" && kind != "draft" {
		return "", fmt.Errorf("%s.kind must be model or draft", label)
	}
	return kind, nil
}

func LoadRepositoryDraftProfile(
	manifestData []byte,
	repositoryID, manifestCommit, label string,
) (DraftProfile, error) {
	if !hfModelID.MatchString(repositoryID) {
		return DraftProfile{}, fmt.Errorf("repository ID must have owner/name form; got %q", repositoryID)
	}
	if manifestCommit != "" && !hfCommit.MatchString(manifestCommit) {
		return DraftProfile{}, fmt.Errorf("manifest commit must be a 40-character SHA; got %q", manifestCommit)
	}
	document, err := parseYAMLDocument(manifestData, label)
	if err != nil {
		return DraftProfile{}, err
	}
	if err := checkKeys(
		document,
		label,
		[]string{"schema_version", "kind", "description", "method", "quantization", "compatible_models"},
		nil,
	); err != nil {
		return DraftProfile{}, err
	}
	if err := schemaVersionOf(document, label); err != nil {
		return DraftProfile{}, err
	}
	kind, kindOK := document["kind"].(string)
	description, descriptionOK := document["description"].(string)
	method, methodOK := document["method"].(string)
	quantization, quantizationOK := document["quantization"].(string)
	compatibleRaw, compatibleOK := document["compatible_models"].([]any)
	if !kindOK || kind != "draft" {
		return DraftProfile{}, fmt.Errorf("%s.kind must be draft", label)
	}
	if !descriptionOK || description == "" || !methodOK || method != "dflash" || !quantizationOK || quantization == "" {
		return DraftProfile{}, fmt.Errorf("%s contains invalid draft metadata", label)
	}
	if !compatibleOK || len(compatibleRaw) == 0 {
		return DraftProfile{}, fmt.Errorf("%s.compatible_models must be a non-empty list", label)
	}
	compatible := make([]string, 0, len(compatibleRaw))
	for _, raw := range compatibleRaw {
		model, ok := raw.(string)
		if !ok || !hfModelID.MatchString(model) {
			return DraftProfile{}, fmt.Errorf("%s.compatible_models must contain Hugging Face model IDs", label)
		}
		compatible = append(compatible, model)
	}
	return DraftProfile{
		RepositoryID: repositoryID, ManifestCommit: manifestCommit, Description: description,
		Method: method, Quantization: quantization, CompatibleModels: compatible,
	}, nil
}

func SortedProfileNames(profiles map[string]ModelProfile) []string {
	names := make([]string, 0, len(profiles))
	for name := range profiles {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}
