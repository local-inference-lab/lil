// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package launcher

import (
	"fmt"
	"sort"
)

const DefaultGPUMemoryUtilization = 0.95

// Default tensor-parallel selection policies stored in topology YAML.
const (
	DefaultTPFit = "fit"
	DefaultTPAll = "all"
)

type LocalTopology struct {
	Name              string
	Host              string
	Port              int
	DeviceMemoryBytes int64
	RepoRoot          string
	Python            string
	B12XRoot          string
	CUDAHome          string
	CuteDSLArch       string
	DevicePools       [][]int
	Environment       map[string]string
}

func (t *LocalTopology) MaxDevices() int {
	devices := 0
	for _, pool := range t.DevicePools {
		devices = max(devices, len(pool))
	}
	return devices
}

func (t *LocalTopology) SelectDevices(tpSize int) ([]int, error) {
	for _, pool := range t.DevicePools {
		if len(pool) >= tpSize {
			return append([]int(nil), pool[:tpSize]...), nil
		}
	}
	return nil, fmt.Errorf(
		"topology %q has at most %d local GPUs, but TP=%d was requested",
		t.Name, t.MaxDevices(), tpSize,
	)
}

type SparkNode struct {
	SSHHost           string
	Address           string
	EthernetInterface string
	RDMAInterfaces    []string
}

type CacheMount struct {
	Source string
	Target string
}

type SparkRDMATopology struct {
	Name                  string
	Host                  string
	Port                  int
	DeviceMemoryBytes     int64
	RepoRoot              string
	Python                string
	RuntimeRepoRoot       string
	RuntimePython         string
	VLLMBin               string
	B12XRoot              string
	RuntimeB12XRoot       string
	Nodes                 []SparkNode
	MasterPort            int
	Image                 string
	ContainerNamePrefix   string
	ContainerMemoryGB     int
	ContainerMemorySwapGB int
	ContainerShmGB        int
	ContainerPidsLimit    int
	ContainerNofileLimit  int
	CacheMounts           []CacheMount
	CUDAHome              string
	CuteDSLArch           string
	NCCLDebug             string
	NCCLIBGIDIndex        int
	NCCLIBMergeNICs       bool
	DeviceID              int
	Environment           map[string]string
}

type Topology struct {
	Kind                 string
	GPUMemoryUtilization float64
	DefaultTP            string
	Local                *LocalTopology
	Spark                *SparkRDMATopology
}

func (t Topology) MemoryUtilization() float64 {
	if t.GPUMemoryUtilization == 0 {
		return DefaultGPUMemoryUtilization
	}
	return t.GPUMemoryUtilization
}

func (t Topology) Name() string {
	if t.Local != nil {
		return t.Local.Name
	}
	return t.Spark.Name
}

func (t Topology) Host() string {
	if t.Local != nil {
		return t.Local.Host
	}
	return t.Spark.Host
}

func (t Topology) Port() int {
	if t.Local != nil {
		return t.Local.Port
	}
	return t.Spark.Port
}

func (t Topology) DeviceMemoryBytes() int64 {
	if t.Local != nil {
		return t.Local.DeviceMemoryBytes
	}
	return t.Spark.DeviceMemoryBytes
}

func (t Topology) RepoRoot() string {
	if t.Local != nil {
		return t.Local.RepoRoot
	}
	return t.Spark.RepoRoot
}

func (t Topology) Python() string {
	if t.Local != nil {
		return t.Local.Python
	}
	return t.Spark.Python
}

func (t Topology) B12XRoot() string {
	if t.Local != nil {
		return t.Local.B12XRoot
	}
	return t.Spark.B12XRoot
}

func (t Topology) CUDAHome() string {
	if t.Local != nil {
		return t.Local.CUDAHome
	}
	return t.Spark.CUDAHome
}

func (t Topology) CuteDSLArch() string {
	if t.Local != nil {
		return t.Local.CuteDSLArch
	}
	return t.Spark.CuteDSLArch
}

// Environment returns the host tuning recorded by discovery for this topology.
func (t Topology) Environment() map[string]string {
	if t.Local != nil {
		return t.Local.Environment
	}
	return t.Spark.Environment
}

// MaxTPSize is the largest tensor-parallel size the topology can host.
func (t Topology) MaxTPSize() int {
	if t.Local != nil {
		return t.Local.MaxDevices()
	}
	return len(t.Spark.Nodes)
}

type Capacity struct {
	GPUMemoryUtilization *float64 `json:"gpu_memory_utilization"`
	KVCacheMemoryBytes   *string  `json:"kv_cache_memory_bytes"`
	MaxModelLen          string   `json:"max_model_len"`
	MaxNumSeqs           int      `json:"max_num_seqs"`
	MaxNumBatchedTokens  int      `json:"max_num_batched_tokens"`
}

// LaunchLayer is one partial contribution to the resolved launch settings.
// Capacity and Compilation are raw mappings so that layers merge recursively
// before the resolved capacity is validated.
type LaunchLayer struct {
	Capacity    map[string]any
	Compilation map[string]any
	Environment map[string]string
}

// OverrideCondition selects the launches an override applies to. Zero-valued
// fields are unconstrained; set fields must all match.
type OverrideCondition struct {
	Kind string
	Arch string
	TP   int
}

func (c OverrideCondition) Matches(kind, arch string, tpSize int) bool {
	return (c.Kind == "" || c.Kind == kind) &&
		(c.Arch == "" || c.Arch == arch) &&
		(c.TP == 0 || c.TP == tpSize)
}

type LaunchOverride struct {
	When  OverrideCondition
	Layer LaunchLayer
}

type LaunchSettings struct {
	Capacity          Capacity
	CompilationConfig map[string]any
	Environment       map[string]string
}

type MultimodalPolicy struct {
	EncoderTPMode    *string
	ProcessorCacheGB *float64
	LimitPerPrompt   map[string]int
}

type ServingPolicy struct {
	ServedModelName           string
	TrustRemoteCode           bool
	ReasoningParser           string
	ToolCallParser            string
	AutoToolChoice            bool
	GenerationConfig          *string
	HFOverrides               map[string]any
	AsyncScheduling           bool
	PrefixCaching             bool
	ChunkedPrefill            bool
	LongPrefillTokenThreshold *int
	Multimodal                *MultimodalPolicy
}

type KernelPolicy struct {
	DType              string
	Quantization       *string
	Attention          *string
	Linear             string
	MoE                string
	GDNDecode          *string
	BlockSize          *int
	MambaCacheMode     *string
	FlashinferAutotune bool
	LoadFormat         string
	LoaderExtraConfig  map[string]any
}

type MTPPolicy struct {
	Tokens            int
	MoEQuantization   string
	Attention         *string
	DraftSampleMethod *string
	WeightsInTarget   bool
}

type DFlashPolicy struct {
	Tokens int
	Model  string
}

type SpeculatorPolicy struct {
	Default string
	MTP     *MTPPolicy
	DFlash  *DFlashPolicy
}

type Requirements struct {
	Arch []string
}

// CheckpointFacts are read from the checkpoint itself: config.json plus the
// stored weight size. They are never declared by a manifest.
type CheckpointFacts struct {
	Source            string
	Architectures     []string
	AttentionHeads    int
	WeightBytes       int64
	WeightBytesSource string
	Config            map[string]any
}

type ModelProfile struct {
	Name           string
	Description    string
	Model          string
	ManifestCommit string
	Family         string
	Serving        ServingPolicy
	Kernels        KernelPolicy
	Speculators    SpeculatorPolicy
	Base           LaunchLayer
	Overrides      []LaunchOverride
	Requires       Requirements
	Facts          *CheckpointFacts
}

type ProfilerOptions struct {
	OutputDir     string
	RecordShapes  bool
	WithMemory    bool
	WithStack     bool
	WithFlops     bool
	UseGzip       bool
	MaxIterations int
}

type EnvironmentOverride struct {
	Name  string
	Value string
}

type LaunchOptions struct {
	TPSize                    *int
	DeviceIDs                 []int
	ModelPath                 *string
	ServedModelName           *string
	Host                      *string
	Port                      *int
	GPUMemoryUtilization      *float64
	KVCacheMemoryBytes        *string
	KVCacheDType              string
	MaxModelLen               *string
	MaxNumSeqs                *int
	MaxNumBatchedTokens       *int
	DCPSize                   int
	DCPCommBackend            string
	Speculator                *string
	SpeculativeTokens         *int
	AdaptiveSpeculativeTokens bool
	AdaptiveWindow            int
	AdaptiveInitial           *int
	PLECPUOffload             *bool
	B12XPolicyMode            string
	Profiler                  *ProfilerOptions
	EnvironmentOverrides      []EnvironmentOverride
	ExtraVLLMArgs             []string
	Detach                    bool
	SyncCode                  bool
	SyncModel                 bool
}

type SparkNodeLaunch struct {
	Node               SparkNode         `json:"-"`
	Rank               int               `json:"rank"`
	ContainerName      string            `json:"container_name"`
	RuntimeEnvironment map[string]string `json:"runtime_environment"`
	VLLMArgv           []string          `json:"vllm_argv"`
	DockerArgv         []string          `json:"docker_argv"`
}

type LaunchSpec struct {
	Model                ModelProfile
	Topology             Topology
	TPSize               int
	ModelSource          string
	CheckpointPath       *string
	ServedModelName      string
	Host                 string
	Port                 int
	Detach               bool
	SyncCode             bool
	SyncModel            bool
	DownloadRepositories []string
	RuntimeEnvironment   map[string]string
	HostEnvironment      map[string]string
	UnsetEnvironment     []string
	VLLMArgv             []string
	CommandArgv          []string
	DeviceIDs            []int
	SparkNodes           []SparkNodeLaunch
	Metadata             map[string]any
}

type SparkTarget struct {
	Topology      *SparkRDMATopology
	ProfileName   string
	TPSize        int
	Port          int
	ContainerName string
	Nodes         []SparkNode
}

type SparkNodeStatus struct {
	Node      SparkNode
	Rank      int
	Exists    bool
	Running   bool
	Status    string
	StartedAt string
	ExitCode  int
}

type CheckResult struct {
	Name   string
	Status string
	Detail string
}

func (r CheckResult) Failed() bool { return r.Status == "FAIL" }

func sortedMapKeys[V any](values map[string]V) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}
