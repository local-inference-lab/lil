// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package launcher

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

var layerExperts = regexp.MustCompile(`(?:^|\.)layers\.(\d+)\.mlp\.experts(?:$|\.)`)

type MTPMoEBackendDecision struct {
	Quantization string
	Backend      string
	Evidence     []string
}

func MTPMoEBackendFromQuantization(quantization, evidence string) (MTPMoEBackendDecision, error) {
	backend, ok := map[string]string{
		"nvfp4": "b12x",
		"mxfp4": "b12x",
		"mxfp8": "triton",
		"bf16":  "b12x",
	}[quantization]
	if !ok {
		return MTPMoEBackendDecision{Quantization: quantization, Evidence: []string{evidence}},
			fmt.Errorf("no default MTP MoE backend for quantization %q; declare speculators.mtp.moe_backend", quantization)
	}
	return MTPMoEBackendDecision{quantization, backend, []string{evidence}}, nil
}

type CheckpointMemory struct {
	TotalBytes      int64
	MappedHostBytes int64
	Source          string
}

func resolveExistingPath(path string) string {
	abs, err := filepath.Abs(path)
	if err != nil {
		return filepath.Clean(path)
	}
	if resolved, err := filepath.EvalSymlinks(abs); err == nil {
		return resolved
	}
	return abs
}

func isFile(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.Mode().IsRegular()
}

func safetensorsHeader(path string) (map[string]any, error) {
	source, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("cannot inspect %s: %w", path, err)
	}
	defer source.Close()
	var headerSize uint64
	if err := binary.Read(source, binary.LittleEndian, &headerSize); err != nil {
		return nil, fmt.Errorf("cannot inspect %s: %w", path, err)
	}
	if headerSize > 1<<30 {
		return nil, fmt.Errorf("cannot inspect %s: implausible header size %d", path, headerSize)
	}
	headerBytes := make([]byte, headerSize)
	if _, err := io.ReadFull(source, headerBytes); err != nil {
		return nil, fmt.Errorf("cannot inspect %s: %w", path, err)
	}
	var header map[string]any
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		return nil, fmt.Errorf("cannot inspect %s: %w", path, err)
	}
	return header, nil
}

func tensorBytes(entry any, context string) (int64, error) {
	metadata, ok := entry.(map[string]any)
	if !ok {
		return 0, fmt.Errorf("%s has invalid tensor metadata", context)
	}
	offsets, ok := metadata["data_offsets"].([]any)
	if !ok || len(offsets) != 2 {
		return 0, fmt.Errorf("%s has invalid data offsets", context)
	}
	start, startOK := jsonInteger(offsets[0])
	end, endOK := jsonInteger(offsets[1])
	if !startOK || !endOK || start < 0 || end < start {
		return 0, fmt.Errorf("%s has invalid data offsets", context)
	}
	return end - start, nil
}

func jsonInteger(value any) (int64, bool) {
	switch number := value.(type) {
	case float64:
		integer := int64(number)
		return integer, float64(integer) == number
	case int64:
		return number, true
	case int:
		return int64(number), true
	case json.Number:
		value, err := number.Int64()
		return value, err == nil
	default:
		return 0, false
	}
}

func CheckpointMemoryInfo(modelPath string, pleMappedHost bool) (*CheckpointMemory, error) {
	indexPath := filepath.Join(modelPath, "model.safetensors.index.json")
	if !isFile(indexPath) {
		shards, _ := filepath.Glob(filepath.Join(modelPath, "*.safetensors"))
		if len(shards) == 0 {
			return nil, nil
		}
		var total int64
		for _, shard := range shards {
			info, err := os.Stat(shard)
			if err != nil {
				return nil, fmt.Errorf("cannot inspect %s: %w", shard, err)
			}
			total += info.Size()
		}
		return &CheckpointMemory{total, 0, "safetensors file sizes"}, nil
	}
	var index map[string]any
	data, err := os.ReadFile(indexPath)
	if err != nil || json.Unmarshal(data, &index) != nil {
		return nil, fmt.Errorf("cannot inspect %s", indexPath)
	}
	metadata, _ := index["metadata"].(map[string]any)
	var total int64
	if value, ok := metadata["total_size"]; ok {
		switch typed := value.(type) {
		case float64:
			total = int64(typed)
		case string:
			total, _ = strconv.ParseInt(typed, 10, 64)
		}
	}
	if total <= 0 {
		return nil, nil
	}
	var mappedHost int64
	if pleMappedHost {
		weightMap, ok := index["weight_map"].(map[string]any)
		if !ok {
			return nil, fmt.Errorf("%s.weight_map must be a mapping", indexPath)
		}
		targets := map[string][]string{}
		for tensor, rawShard := range weightMap {
			shard, ok := rawShard.(string)
			if ok && strings.Contains(tensor, ".ple.ple_embedding.ngram_embedding.") {
				targets[shard] = append(targets[shard], tensor)
			}
		}
		for shard, tensors := range targets {
			path := filepath.Join(modelPath, shard)
			header, err := safetensorsHeader(path)
			if err != nil {
				return nil, err
			}
			for _, tensor := range tensors {
				bytes, err := tensorBytes(header[tensor], path+":"+tensor)
				if err != nil {
					return nil, err
				}
				mappedHost += bytes
			}
		}
	}
	return &CheckpointMemory{total, mappedHost, indexPath}, nil
}

func loadConfig(modelPath string) (map[string]any, error) {
	path := filepath.Join(modelPath, "config.json")
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("cannot inspect %s: %w", path, err)
	}
	return ParseCheckpointConfig(data, path)
}

// ParseCheckpointConfig decodes a checkpoint config.json document.
func ParseCheckpointConfig(data []byte, label string) (map[string]any, error) {
	var config map[string]any
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("cannot parse %s: %w", label, err)
	}
	return config, nil
}

func modelConfig(config map[string]any) map[string]any {
	if text, ok := config["text_config"].(map[string]any); ok {
		return text
	}
	return config
}

// FactsFromConfig derives checkpoint facts from a parsed config.json and a
// stored weight size measured elsewhere.
func FactsFromConfig(config map[string]any, weightBytes int64, weightBytesSource, source string) (*CheckpointFacts, error) {
	rawArchitectures, ok := config["architectures"].([]any)
	if !ok || len(rawArchitectures) == 0 {
		return nil, fmt.Errorf("%s: config.json does not list architectures", source)
	}
	architectures := make([]string, 0, len(rawArchitectures))
	for _, raw := range rawArchitectures {
		value, ok := raw.(string)
		if !ok || value == "" {
			return nil, fmt.Errorf("%s: config.json architectures must be strings", source)
		}
		architectures = append(architectures, value)
	}
	heads, ok := jsonInteger(modelConfig(config)["num_attention_heads"])
	if !ok || heads <= 0 {
		return nil, fmt.Errorf("%s: config.json does not declare num_attention_heads", source)
	}
	if weightBytes <= 0 {
		return nil, fmt.Errorf("%s: stored weight size is unknown", source)
	}
	return &CheckpointFacts{
		Source: source, Architectures: architectures, AttentionHeads: int(heads),
		WeightBytes: weightBytes, WeightBytesSource: weightBytesSource, Config: config,
	}, nil
}

// LoadCheckpointFacts reads facts from a checkpoint directory. A directory
// holding only config.json and a safetensors index is accepted, which lets a
// metadata-only mirror of a repository stand in for the checkpoint.
func LoadCheckpointFacts(modelPath string) (*CheckpointFacts, error) {
	config, err := loadConfig(modelPath)
	if err != nil {
		return nil, err
	}
	memory, err := CheckpointMemoryInfo(modelPath, false)
	if err != nil {
		return nil, err
	}
	if memory == nil {
		return nil, fmt.Errorf(
			"%s has no model.safetensors.index.json or safetensors shards to size", modelPath,
		)
	}
	return FactsFromConfig(config, memory.TotalBytes, memory.Source, modelPath)
}

func mtpLayerIndices(config map[string]any) map[int64]bool {
	model := modelConfig(config)
	start, startOK := jsonInteger(model["num_hidden_layers"])
	count, countOK := jsonInteger(model["num_nextn_predict_layers"])
	if !countOK {
		count, countOK = jsonInteger(model["mtp_num_hidden_layers"])
	}
	result := map[int64]bool{}
	if !startOK || !countOK || count <= 0 {
		return result
	}
	for layer := start; layer < start+count; layer++ {
		result[layer] = true
	}
	return result
}

func isMTPExpertTarget(target string, layers map[int64]bool) bool {
	normalized := strings.ToLower(target)
	if strings.Contains(normalized, "mtp") && strings.Contains(normalized, ".experts") {
		return true
	}
	match := layerExperts.FindStringSubmatch(normalized)
	if match == nil {
		return false
	}
	layer, _ := strconv.ParseInt(match[1], 10, 64)
	return layers[layer]
}

func normalizeQuantization(value any) string {
	text, ok := value.(string)
	if !ok {
		return ""
	}
	normalized := strings.NewReplacer("_", "", "-", "").Replace(strings.ToLower(text))
	switch {
	case strings.Contains(normalized, "nvfp4"):
		return "nvfp4"
	case strings.Contains(normalized, "mxfp8"):
		return "mxfp8"
	case strings.Contains(normalized, "mxfp4"), normalized == "fp4":
		return "mxfp4"
	case normalized == "fp8", normalized == "e4m3", normalized == "float8e4m3fn":
		return "fp8"
	case normalized == "bf16" || normalized == "bfloat16":
		return "bf16"
	default:
		return ""
	}
}

func groupQuantization(name string, group map[string]any) string {
	if quantization := normalizeQuantization(name); quantization != "" {
		return quantization
	}
	weights, ok := group["weights"].(map[string]any)
	if !ok {
		return ""
	}
	bits, bitsOK := jsonInteger(weights["num_bits"])
	groupSize, sizeOK := jsonInteger(weights["group_size"])
	weightType, _ := weights["type"].(string)
	if bitsOK && sizeOK && bits == 4 && groupSize == 16 && weightType == "float" {
		return "nvfp4"
	}
	if bitsOK && sizeOK && bits == 8 && groupSize == 32 && weightType == "float" {
		return "mxfp8"
	}
	return ""
}

func unquantizedMTPDecision(config map[string]any, defaultDType string) (MTPMoEBackendDecision, error) {
	model := modelConfig(config)
	dtype := model["dtype"]
	if dtype == nil {
		dtype = model["torch_dtype"]
	}
	if dtype == nil {
		dtype = config["dtype"]
	}
	if dtype == nil {
		dtype = config["torch_dtype"]
	}
	if dtype == nil {
		dtype = defaultDType
	}
	normalized := strings.TrimPrefix(strings.ToLower(fmt.Sprint(dtype)), "torch.")
	if normalized != "bf16" && normalized != "bfloat16" {
		return MTPMoEBackendDecision{}, fmt.Errorf(
			"cannot select an MTP MoE backend for unquantized checkpoint dtype %q", dtype,
		)
	}
	return MTPMoEBackendFromQuantization("bf16", fmt.Sprintf("no quantized MTP expert target; dtype=%v", dtype))
}

// blockFP8CheckpointDecision handles checkpoints whose quantization_config
// declares block-scaled FP8 for every projection and an expert_dtype for the
// MoE experts, the layout DeepSeek publishes. The MTP experts share the
// routed experts' format.
func blockFP8CheckpointDecision(config, quantizationConfig map[string]any) (MTPMoEBackendDecision, bool, error) {
	method, _ := quantizationConfig["quant_method"].(string)
	if strings.ToLower(method) != "fp8" {
		return MTPMoEBackendDecision{}, false, nil
	}
	expertDType := "fp4"
	if value, ok := config["expert_dtype"].(string); ok && value != "" {
		expertDType = value
	}
	quantization := normalizeQuantization(expertDType)
	if quantization == "" {
		return MTPMoEBackendDecision{}, false, nil
	}
	evidence := fmt.Sprintf("quantization_config.quant_method=fp8 with expert_dtype=%s", expertDType)
	decision, err := MTPMoEBackendFromQuantization(quantization, evidence)
	return decision, true, err
}

// MTPMoEBackendFromConfig decides the MTP expert kernel backend from the
// quantization metadata of a parsed config.json.
func MTPMoEBackendFromConfig(config map[string]any, defaultDType string) (MTPMoEBackendDecision, error) {
	quantizationConfig, ok := config["quantization_config"].(map[string]any)
	if !ok {
		return unquantizedMTPDecision(config, defaultDType)
	}
	layers := mtpLayerIndices(config)
	type finding struct{ quantization, target string }
	var findings []finding
	if quantizedLayers, ok := quantizationConfig["quantized_layers"].(map[string]any); ok {
		for target, rawSettings := range quantizedLayers {
			if !isMTPExpertTarget(target, layers) {
				continue
			}
			settings, _ := rawSettings.(map[string]any)
			findings = append(findings, finding{normalizeQuantization(settings["quant_algo"]), target})
		}
	}
	if len(findings) == 0 {
		if groups, ok := quantizationConfig["config_groups"].(map[string]any); ok {
			for name, rawGroup := range groups {
				group, ok := rawGroup.(map[string]any)
				if !ok {
					continue
				}
				targets, ok := group["targets"].([]any)
				if !ok {
					continue
				}
				for _, rawTarget := range targets {
					target, ok := rawTarget.(string)
					if ok && isMTPExpertTarget(target, layers) {
						findings = append(findings, finding{groupQuantization(name, group), target})
					}
				}
			}
		}
	}
	if len(findings) == 0 {
		if decision, ok, err := blockFP8CheckpointDecision(config, quantizationConfig); ok {
			return decision, err
		}
		return unquantizedMTPDecision(config, defaultDType)
	}
	unknown := []string{}
	algorithms := map[string]bool{}
	for _, item := range findings {
		if item.quantization == "" {
			unknown = append(unknown, item.target)
		}
		algorithms[item.quantization] = true
	}
	if len(unknown) > 0 {
		sort.Strings(unknown)
		return MTPMoEBackendDecision{}, fmt.Errorf("cannot determine MTP MoE quantization for checkpoint targets: %s", strings.Join(unknown, ", "))
	}
	if len(algorithms) != 1 {
		parts := make([]string, 0, len(findings))
		for _, item := range findings {
			parts = append(parts, item.target+"="+item.quantization)
		}
		sort.Strings(parts)
		return MTPMoEBackendDecision{}, fmt.Errorf("MTP MoE checkpoint targets use conflicting quantization: %s", strings.Join(parts, ", "))
	}
	quantization := findings[0].quantization
	backendDecision, err := MTPMoEBackendFromQuantization(quantization, "checkpoint metadata")
	if err != nil {
		return MTPMoEBackendDecision{}, err
	}
	evidence := make([]string, 0, len(findings))
	for _, item := range findings {
		evidence = append(evidence, item.target)
	}
	sort.Strings(evidence)
	return MTPMoEBackendDecision{quantization, backendDecision.Backend, evidence}, nil
}

// ResolveMTPMoEBackend decides the MTP expert backend from a checkpoint directory.
func ResolveMTPMoEBackend(modelPath, defaultDType string) (MTPMoEBackendDecision, error) {
	config, err := loadConfig(modelPath)
	if err != nil {
		return MTPMoEBackendDecision{}, err
	}
	return MTPMoEBackendFromConfig(config, defaultDType)
}
