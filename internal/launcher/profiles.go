// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package launcher

import (
	"bytes"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

var hfModelID = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$`)
var hfCommit = regexp.MustCompile(`^[0-9a-f]{40}$`)

type profileYAML struct {
	Description               string         `yaml:"description"`
	Model                     string         `yaml:"model"`
	ServedModelName           string         `yaml:"served_model_name"`
	ExpectedArchitectures     []string       `yaml:"expected_architectures"`
	AttentionHeads            int            `yaml:"attention_heads"`
	WeightBytes               int64          `yaml:"weight_bytes"`
	Quantization              *string        `yaml:"quantization"`
	LoadFormat                string         `yaml:"load_format"`
	DType                     string         `yaml:"dtype"`
	BlockSize                 *int           `yaml:"block_size"`
	AttentionBackend          *string        `yaml:"attention_backend"`
	LinearBackend             string         `yaml:"linear_backend"`
	MoEBackend                string         `yaml:"moe_backend"`
	ReasoningParser           string         `yaml:"reasoning_parser"`
	ToolCallParser            string         `yaml:"tool_call_parser"`
	TrustRemoteCode           bool           `yaml:"trust_remote_code"`
	MambaCacheMode            *string        `yaml:"mamba_cache_mode"`
	AsyncScheduling           bool           `yaml:"async_scheduling"`
	DisableFlashinferAutotune bool           `yaml:"disable_flashinfer_autotune"`
	GDNDecodeKernel           *string        `yaml:"gdn_decode_kernel"`
	ModelLoaderExtraConfig    map[string]any `yaml:"model_loader_extra_config"`
	MMEncoderTPMode           *string        `yaml:"mm_encoder_tp_mode"`
	MMProcessorCacheGB        *float64       `yaml:"mm_processor_cache_gb"`
	LimitMMPerPrompt          map[string]int `yaml:"limit_mm_per_prompt"`
	GenerationConfig          *string        `yaml:"generation_config"`
	HFOverrides               map[string]any `yaml:"hf_overrides"`
	LongPrefillTokenThreshold *int           `yaml:"long_prefill_token_threshold"`
	DefaultSpeculator         string         `yaml:"default_speculator"`
	MTPTokens                 *int           `yaml:"mtp_tokens"`
	DFlash2Tokens             *int           `yaml:"dflash2_tokens"`
	DFlash2Model              *string        `yaml:"dflash2_model"`
	MTPMoEQuantization        string         `yaml:"mtp_moe_quantization"`
	MTPAttentionBackend       *string        `yaml:"mtp_attention_backend"`
	MTPModel                  *string        `yaml:"mtp_model"`
	MTPDraftSampleMethod      *string        `yaml:"mtp_draft_sample_method"`
	CUDADeviceMaxConnections  *int           `yaml:"cuda_device_max_connections"`
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

func deepMerge(base, overlay map[string]any) map[string]any {
	merged := cloneValue(base).(map[string]any)
	for key, value := range overlay {
		if key == "extends" {
			continue
		}
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

func parentNames(value any, context string) ([]string, error) {
	if value == nil {
		return nil, nil
	}
	if parent, ok := value.(string); ok && parent != "" {
		return []string{parent}, nil
	}
	items, ok := value.([]any)
	if !ok || len(items) == 0 {
		return nil, fmt.Errorf("%s must be a base name or non-empty list", context)
	}
	parents := make([]string, 0, len(items))
	seen := map[string]bool{}
	for _, item := range items {
		parent, ok := item.(string)
		if !ok || parent == "" {
			return nil, fmt.Errorf("%s must be a base name or non-empty list", context)
		}
		if seen[parent] {
			return nil, fmt.Errorf("%s contains duplicate parents", context)
		}
		seen[parent] = true
		parents = append(parents, parent)
	}
	return parents, nil
}

func resolveProfileMaps(bases, models map[string]any, context string) (map[string]map[string]any, error) {
	resolvedBases := map[string]map[string]any{}
	visiting := []string{}
	var resolveBase func(string) (map[string]any, error)
	resolveBase = func(name string) (map[string]any, error) {
		if resolved, ok := resolvedBases[name]; ok {
			return resolved, nil
		}
		value, ok := bases[name]
		if !ok {
			return nil, fmt.Errorf("%s.bases has no base named %q", context, name)
		}
		for index, active := range visiting {
			if active == name {
				cycle := append(append([]string{}, visiting[index:]...), name)
				return nil, fmt.Errorf("profile inheritance cycle: %s", strings.Join(cycle, " -> "))
			}
		}
		visiting = append(visiting, name)
		raw, err := mapping(value, context+".bases."+name)
		if err != nil {
			return nil, err
		}
		parents, err := parentNames(raw["extends"], context+".bases."+name+".extends")
		if err != nil {
			return nil, err
		}
		merged := map[string]any{}
		for _, parent := range parents {
			resolved, err := resolveBase(parent)
			if err != nil {
				return nil, err
			}
			merged = deepMerge(merged, resolved)
		}
		merged = deepMerge(merged, raw)
		visiting = visiting[:len(visiting)-1]
		resolvedBases[name] = merged
		return merged, nil
	}

	resolvedModels := map[string]map[string]any{}
	for name, value := range models {
		raw, err := mapping(value, context+".models."+name)
		if err != nil {
			return nil, err
		}
		parents, err := parentNames(raw["extends"], context+".models."+name+".extends")
		if err != nil {
			return nil, err
		}
		merged := map[string]any{}
		for _, parent := range parents {
			resolved, err := resolveBase(parent)
			if err != nil {
				return nil, err
			}
			merged = deepMerge(merged, resolved)
		}
		resolvedModels[name] = deepMerge(merged, raw)
	}
	return resolvedModels, nil
}

func parseCapacity(value any, context string) (Capacity, error) {
	data, err := mapping(value, context)
	if err != nil {
		return Capacity{}, err
	}
	keys := []string{"kv_cache_memory_bytes", "max_model_len", "max_num_seqs", "max_num_batched_tokens"}
	if err := checkKeys(data, context, keys, nil); err != nil {
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

func parseLaunchSettings(value any, context string) (LaunchSettings, error) {
	data, err := mapping(value, context)
	if err != nil {
		return LaunchSettings{}, err
	}
	if err := checkKeys(data, context, []string{"capacity", "compilation_config", "environment"}, nil); err != nil {
		return LaunchSettings{}, err
	}
	capacity, err := parseCapacity(data["capacity"], context+".capacity")
	if err != nil {
		return LaunchSettings{}, err
	}
	var compilation map[string]any
	if data["compilation_config"] != nil {
		compilation, err = mapping(data["compilation_config"], context+".compilation_config")
		if err != nil {
			return LaunchSettings{}, err
		}
		compilation = cloneValue(compilation).(map[string]any)
	}
	environmentRaw, err := mapping(data["environment"], context+".environment")
	if err != nil {
		return LaunchSettings{}, err
	}
	allowedEnvironment := map[string]bool{
		"INSTANTTENSOR_BUFFER_SIZE": true, "INSTANTTENSOR_CHUNK_SIZE": true,
		"INSTANTTENSOR_CONCURRENCY": true, "INSTANTTENSOR_IO_DEPTH": true,
		"VLLM_PLE_CPU_OFFLOAD": true,
	}
	environment := map[string]string{}
	for key, raw := range environmentRaw {
		value, ok := raw.(string)
		if !ok {
			return LaunchSettings{}, fmt.Errorf("%s.environment values must be strings", context)
		}
		if !allowedEnvironment[key] {
			return LaunchSettings{}, fmt.Errorf("%s.environment contains unsupported key: %s", context, key)
		}
		environment[key] = value
	}
	return LaunchSettings{capacity, compilation, environment}, nil
}

func parseTopologyPolicy(value any, inherited map[string]any, context string) (TopologyLaunchPolicy, error) {
	data, err := mapping(value, context)
	if err != nil {
		return TopologyLaunchPolicy{}, err
	}
	if err := checkKeys(data, context, []string{"default_tp_size"}, []string{"defaults", "tp"}); err != nil {
		return TopologyLaunchPolicy{}, err
	}
	policy := TopologyLaunchPolicy{TP: map[int]LaunchSettings{}}
	switch value := data["default_tp_size"].(type) {
	case string:
		if value != "all" {
			return policy, fmt.Errorf("%s.default_tp_size must be a positive integer or 'all'", context)
		}
		policy.DefaultTPAll = true
	case int:
		if value <= 0 {
			return policy, fmt.Errorf("%s.default_tp_size must be a positive integer or 'all'", context)
		}
		policy.DefaultTPSize = value
	default:
		return policy, fmt.Errorf("%s.default_tp_size must be a positive integer or 'all'", context)
	}
	defaultsOverlay := map[string]any{}
	if value, ok := data["defaults"]; ok {
		defaultsOverlay, err = mapping(value, context+".defaults")
		if err != nil {
			return policy, err
		}
	}
	defaults := deepMerge(inherited, defaultsOverlay)
	policy.Defaults, err = parseLaunchSettings(defaults, context+".resolved_defaults")
	if err != nil {
		return policy, err
	}
	if rawTP, ok := data["tp"]; ok {
		tpData, err := mapping(rawTP, context+".tp")
		if err != nil {
			return policy, err
		}
		for rawSize, rawOverrides := range tpData {
			size, err := strconv.Atoi(rawSize)
			if err != nil || size <= 0 || strconv.Itoa(size) != rawSize {
				return policy, fmt.Errorf("%s.tp keys must be positive integers", context)
			}
			overrides, err := mapping(rawOverrides, fmt.Sprintf("%s.tp.%d", context, size))
			if err != nil {
				return policy, err
			}
			policy.TP[size], err = parseLaunchSettings(deepMerge(defaults, overrides), fmt.Sprintf("%s.tp.%d.resolved", context, size))
			if err != nil {
				return policy, err
			}
		}
	}
	return policy, nil
}

func parseLaunchPolicy(value any, context string) (LaunchPolicy, error) {
	data, err := mapping(value, context)
	if err != nil {
		return LaunchPolicy{}, err
	}
	if err := checkKeys(data, context, []string{"defaults", "local", "spark_rdma"}, nil); err != nil {
		return LaunchPolicy{}, err
	}
	defaults, err := mapping(data["defaults"], context+".defaults")
	if err != nil {
		return LaunchPolicy{}, err
	}
	local, err := parseTopologyPolicy(data["local"], defaults, context+".local")
	if err != nil {
		return LaunchPolicy{}, err
	}
	spark, err := parseTopologyPolicy(data["spark_rdma"], defaults, context+".spark_rdma")
	if err != nil {
		return LaunchPolicy{}, err
	}
	return LaunchPolicy{Local: local, SparkRDMA: spark}, nil
}

func decodeProfile(name string, value map[string]any, context string) (ModelProfile, error) {
	launchRaw, ok := value["launch"]
	if !ok {
		return ModelProfile{}, fmt.Errorf("%s is missing required key: launch", context)
	}
	withoutLaunch := cloneValue(value).(map[string]any)
	delete(withoutLaunch, "launch")
	encoded, err := yaml.Marshal(withoutLaunch)
	if err != nil {
		return ModelProfile{}, err
	}
	var raw profileYAML
	decoder := yaml.NewDecoder(bytes.NewReader(encoded))
	decoder.KnownFields(true)
	if err := decoder.Decode(&raw); err != nil {
		return ModelProfile{}, fmt.Errorf("invalid model profile %s: %w", context, err)
	}
	if !nonEmpty(raw.Description, raw.Model, raw.ServedModelName, raw.LoadFormat,
		raw.DType, raw.LinearBackend, raw.MoEBackend, raw.ReasoningParser, raw.ToolCallParser) {
		return ModelProfile{}, fmt.Errorf("%s contains an empty required string", context)
	}
	if !hfModelID.MatchString(raw.Model) {
		return ModelProfile{}, fmt.Errorf("%s.model must be a Hugging Face model ID in owner/name form", context)
	}
	if len(raw.ExpectedArchitectures) == 0 || raw.AttentionHeads <= 0 {
		return ModelProfile{}, fmt.Errorf("%s requires architectures and positive attention_heads", context)
	}
	if raw.WeightBytes < 0 {
		return ModelProfile{}, fmt.Errorf("%s.weight_bytes must be positive or zero", context)
	}
	for _, architecture := range raw.ExpectedArchitectures {
		if architecture == "" {
			return ModelProfile{}, fmt.Errorf("%s.expected_architectures contains an empty value", context)
		}
	}
	positivePointers := map[string]*int{
		"block_size":                   raw.BlockSize,
		"long_prefill_token_threshold": raw.LongPrefillTokenThreshold,
		"cuda_device_max_connections":  raw.CUDADeviceMaxConnections,
	}
	for field, pointer := range positivePointers {
		if pointer != nil && *pointer <= 0 {
			return ModelProfile{}, fmt.Errorf("%s.%s must be positive or null", context, field)
		}
	}
	if raw.MMProcessorCacheGB != nil && *raw.MMProcessorCacheGB < 0 {
		return ModelProfile{}, fmt.Errorf("%s.mm_processor_cache_gb must be non-negative or null", context)
	}
	defaultSpeculator := raw.DefaultSpeculator
	if defaultSpeculator == "" {
		defaultSpeculator = "mtp"
	}
	if defaultSpeculator != "mtp" && defaultSpeculator != "dflash2" && defaultSpeculator != "none" {
		return ModelProfile{}, fmt.Errorf("%s.default_speculator must be mtp, dflash2, or none", context)
	}
	mtpTokens := 3
	if raw.MTPTokens != nil {
		mtpTokens = *raw.MTPTokens
	}
	if mtpTokens < 0 || raw.DFlash2Tokens != nil && *raw.DFlash2Tokens < 0 {
		return ModelProfile{}, fmt.Errorf("%s speculative token counts must be non-negative", context)
	}
	if raw.MTPModel != nil && *raw.MTPModel != "target" {
		return ModelProfile{}, fmt.Errorf("%s.mtp_model must be target or null", context)
	}
	if raw.DFlash2Model != nil && !hfModelID.MatchString(*raw.DFlash2Model) {
		return ModelProfile{}, fmt.Errorf("%s.dflash2_model must be a Hugging Face model ID", context)
	}
	if raw.MTPMoEQuantization != "" && raw.MTPMoEQuantization != "nvfp4" &&
		raw.MTPMoEQuantization != "mxfp8" && raw.MTPMoEQuantization != "bf16" {
		return ModelProfile{}, fmt.Errorf("%s.mtp_moe_quantization must be nvfp4, mxfp8, bf16, or empty", context)
	}
	launch, err := parseLaunchPolicy(launchRaw, context+".launch")
	if err != nil {
		return ModelProfile{}, err
	}
	_, hfOverridesSet := value["hf_overrides"]
	return ModelProfile{
		Name: name, Description: raw.Description, Model: raw.Model,
		ServedModelName:       raw.ServedModelName,
		ExpectedArchitectures: raw.ExpectedArchitectures,
		AttentionHeads:        raw.AttentionHeads, WeightBytes: raw.WeightBytes,
		Launch:       launch,
		Quantization: raw.Quantization, LoadFormat: raw.LoadFormat,
		DType: raw.DType, BlockSize: raw.BlockSize,
		AttentionBackend: raw.AttentionBackend, LinearBackend: raw.LinearBackend,
		MoEBackend: raw.MoEBackend, ReasoningParser: raw.ReasoningParser,
		ToolCallParser: raw.ToolCallParser, TrustRemoteCode: raw.TrustRemoteCode,
		MambaCacheMode: raw.MambaCacheMode, AsyncScheduling: raw.AsyncScheduling,
		DisableFlashinferAutotune: raw.DisableFlashinferAutotune,
		GDNDecodeKernel:           raw.GDNDecodeKernel,
		ModelLoaderExtraConfig:    raw.ModelLoaderExtraConfig,
		MMEncoderTPMode:           raw.MMEncoderTPMode,
		MMProcessorCacheGB:        raw.MMProcessorCacheGB,
		LimitMMPerPrompt:          raw.LimitMMPerPrompt,
		GenerationConfig:          raw.GenerationConfig,
		HFOverrides:               raw.HFOverrides, HFOverridesSet: hfOverridesSet,
		LongPrefillTokenThreshold: raw.LongPrefillTokenThreshold,
		DefaultSpeculator:         defaultSpeculator, MTPTokens: mtpTokens,
		DFlash2Tokens: raw.DFlash2Tokens, DFlash2Model: raw.DFlash2Model,
		MTPMoEQuantization:  raw.MTPMoEQuantization,
		MTPAttentionBackend: raw.MTPAttentionBackend, MTPModel: raw.MTPModel,
		MTPDraftSampleMethod:     raw.MTPDraftSampleMethod,
		CUDADeviceMaxConnections: raw.CUDADeviceMaxConnections,
	}, nil
}

type ModelProfileFile struct {
	Label string
	Data  []byte
}

func loadModelProfileDocument(data []byte, label string) (map[string]any, map[string]any, error) {
	var loaded any
	if err := yaml.Unmarshal(data, &loaded); err != nil {
		return nil, nil, fmt.Errorf("invalid YAML in %s: %w", label, err)
	}
	normalized, err := normalizeYAML(loaded)
	if err != nil {
		return nil, nil, err
	}
	document, err := mapping(normalized, label)
	if err != nil {
		return nil, nil, err
	}
	if err := checkKeys(
		document,
		label,
		[]string{"schema_version"},
		[]string{"bases", "models"},
	); err != nil {
		return nil, nil, err
	}
	version, ok := document["schema_version"].(int)
	if !ok || version != schemaVersion {
		return nil, nil, fmt.Errorf(
			"%s.schema_version must be %d; got %v",
			label,
			schemaVersion,
			document["schema_version"],
		)
	}
	bases := map[string]any{}
	if value, ok := document["bases"]; ok {
		bases, err = mapping(value, label+".bases")
		if err != nil {
			return nil, nil, err
		}
	}
	models := map[string]any{}
	if value, ok := document["models"]; ok {
		models, err = mapping(value, label+".models")
		if err != nil {
			return nil, nil, err
		}
	}
	if len(bases) == 0 && len(models) == 0 {
		return nil, nil, fmt.Errorf("%s must define bases or models", label)
	}
	return bases, models, nil
}

func decodeModelProfiles(
	bases map[string]any,
	models map[string]any,
	context string,
) (map[string]ModelProfile, error) {
	if len(models) == 0 {
		return nil, fmt.Errorf("%s defines no models", context)
	}
	resolved, err := resolveProfileMaps(bases, models, context)
	if err != nil {
		return nil, err
	}
	profiles := make(map[string]ModelProfile, len(resolved))
	for name, value := range resolved {
		profile, err := decodeProfile(name, value, context+".models."+name)
		if err != nil {
			return nil, err
		}
		profiles[name] = profile
	}
	return profiles, nil
}

func LoadModelProfiles(data []byte, label string) (map[string]ModelProfile, error) {
	bases, models, err := loadModelProfileDocument(data, label)
	if err != nil {
		return nil, err
	}
	return decodeModelProfiles(bases, models, label)
}

func LoadModelProfileFiles(files []ModelProfileFile) (map[string]ModelProfile, error) {
	if len(files) == 0 {
		return nil, fmt.Errorf("no model profile YAML files were found")
	}
	ordered := append([]ModelProfileFile(nil), files...)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i].Label < ordered[j].Label })
	bases := map[string]any{}
	models := map[string]any{}
	for _, file := range ordered {
		fileBases, fileModels, err := loadModelProfileDocument(file.Data, file.Label)
		if err != nil {
			return nil, err
		}
		for name, value := range fileBases {
			if _, exists := bases[name]; exists {
				return nil, fmt.Errorf("base %q is defined more than once", name)
			}
			bases[name] = value
		}
		for name, value := range fileModels {
			if _, exists := models[name]; exists {
				return nil, fmt.Errorf("model %q is defined more than once", name)
			}
			models[name] = value
		}
	}
	return decodeModelProfiles(bases, models, "model profile files")
}

func LoadRepositoryModelProfile(
	basesData, manifestData []byte,
	repositoryID, manifestCommit, label string,
) (ModelProfile, error) {
	if !hfModelID.MatchString(repositoryID) {
		return ModelProfile{}, fmt.Errorf("repository ID must have owner/name form; got %q", repositoryID)
	}
	if manifestCommit != "" && !hfCommit.MatchString(manifestCommit) {
		return ModelProfile{}, fmt.Errorf("manifest commit must be a 40-character SHA; got %q", manifestCommit)
	}
	bases, baseModels, err := loadModelProfileDocument(basesData, "embedded model bases")
	if err != nil {
		return ModelProfile{}, err
	}
	if len(baseModels) != 0 {
		return ModelProfile{}, fmt.Errorf("embedded model bases must not define models")
	}
	var loaded any
	if err := yaml.Unmarshal(manifestData, &loaded); err != nil {
		return ModelProfile{}, fmt.Errorf("invalid YAML in %s: %w", label, err)
	}
	normalized, err := normalizeYAML(loaded)
	if err != nil {
		return ModelProfile{}, err
	}
	document, err := mapping(normalized, label)
	if err != nil {
		return ModelProfile{}, err
	}
	version, ok := document["schema_version"].(int)
	if !ok || version != schemaVersion {
		return ModelProfile{}, fmt.Errorf(
			"%s.schema_version must be %d; got %v",
			label, schemaVersion, document["schema_version"],
		)
	}
	kind, ok := document["kind"].(string)
	if !ok || kind != "model" {
		return ModelProfile{}, fmt.Errorf("%s.kind must be model", label)
	}
	if _, exists := document["model"]; exists {
		return ModelProfile{}, fmt.Errorf("%s must not set model; the repository ID is authoritative", label)
	}
	delete(document, "schema_version")
	delete(document, "kind")
	document["model"] = repositoryID
	_, name, _ := strings.Cut(repositoryID, "/")
	resolved, err := resolveProfileMaps(
		bases, map[string]any{name: document}, label,
	)
	if err != nil {
		return ModelProfile{}, err
	}
	profile, err := decodeProfile(name, resolved[name], label)
	if err != nil {
		return ModelProfile{}, err
	}
	profile.ManifestCommit = manifestCommit
	return profile, nil
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
	var loaded any
	if err := yaml.Unmarshal(data, &loaded); err != nil {
		return "", fmt.Errorf("invalid YAML in %s: %w", label, err)
	}
	normalized, err := normalizeYAML(loaded)
	if err != nil {
		return "", err
	}
	document, err := mapping(normalized, label)
	if err != nil {
		return "", err
	}
	version, ok := document["schema_version"].(int)
	if !ok || version != schemaVersion {
		return "", fmt.Errorf("%s.schema_version must be %d; got %v", label, schemaVersion, document["schema_version"])
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
	var document map[string]any
	if err := yaml.Unmarshal(manifestData, &document); err != nil {
		return DraftProfile{}, fmt.Errorf("invalid YAML in %s: %w", label, err)
	}
	if err := checkKeys(
		document,
		label,
		[]string{"schema_version", "kind", "description", "method", "quantization", "compatible_models"},
		nil,
	); err != nil {
		return DraftProfile{}, err
	}
	version, versionOK := document["schema_version"].(int)
	kind, kindOK := document["kind"].(string)
	description, descriptionOK := document["description"].(string)
	method, methodOK := document["method"].(string)
	quantization, quantizationOK := document["quantization"].(string)
	compatibleRaw, compatibleOK := document["compatible_models"].([]any)
	if !versionOK || version != schemaVersion || !kindOK || kind != "draft" {
		return DraftProfile{}, fmt.Errorf("%s must be a schema version %d draft manifest", label, schemaVersion)
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
