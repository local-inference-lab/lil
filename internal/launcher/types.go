// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package launcher

import "fmt"

const DefaultGPUMemoryUtilization = 0.95

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
}

func (t *LocalTopology) SelectDevices(tpSize int) ([]int, error) {
	available := 0
	for _, pool := range t.DevicePools {
		if len(pool) > available {
			available = len(pool)
		}
		if len(pool) >= tpSize {
			return append([]int(nil), pool[:tpSize]...), nil
		}
	}
	return nil, fmt.Errorf(
		"topology %q has at most %d local GPUs, but TP=%d was requested",
		t.Name, available, tpSize,
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
}

type Topology struct {
	Kind                 string
	GPUMemoryUtilization float64
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

type Capacity struct {
	GPUMemoryUtilization *float64 `json:"gpu_memory_utilization"`
	KVCacheMemoryBytes   *string  `json:"kv_cache_memory_bytes"`
	MaxModelLen          string   `json:"max_model_len"`
	MaxNumSeqs           int      `json:"max_num_seqs"`
	MaxNumBatchedTokens  int      `json:"max_num_batched_tokens"`
}

type LaunchSettings struct {
	Capacity          Capacity
	CompilationConfig map[string]any
	Environment       map[string]string
}

type TopologyLaunchPolicy struct {
	DefaultTPAll  bool
	DefaultTPSize int
	Defaults      LaunchSettings
	TP            map[int]LaunchSettings
}

func (p TopologyLaunchPolicy) Resolve(tpSize int) LaunchSettings {
	if settings, ok := p.TP[tpSize]; ok {
		return settings
	}
	return p.Defaults
}

type LaunchPolicy struct {
	Local     TopologyLaunchPolicy
	SparkRDMA TopologyLaunchPolicy
}

func (p LaunchPolicy) Topology(kind string) (TopologyLaunchPolicy, error) {
	switch kind {
	case "local":
		return p.Local, nil
	case "spark_rdma":
		return p.SparkRDMA, nil
	default:
		return TopologyLaunchPolicy{}, fmt.Errorf("unsupported topology kind: %q", kind)
	}
}

type ModelProfile struct {
	Name                      string
	Description               string
	Model                     string
	ManifestCommit            string
	ServedModelName           string
	ExpectedArchitectures     []string
	AttentionHeads            int
	WeightBytes               int64
	Launch                    LaunchPolicy
	Quantization              *string
	LoadFormat                string
	DType                     string
	BlockSize                 *int
	AttentionBackend          *string
	LinearBackend             string
	MoEBackend                string
	ReasoningParser           string
	ToolCallParser            string
	TrustRemoteCode           bool
	MambaCacheMode            *string
	AsyncScheduling           bool
	DisableFlashinferAutotune bool
	GDNDecodeKernel           *string
	ModelLoaderExtraConfig    map[string]any
	MMEncoderTPMode           *string
	MMProcessorCacheGB        *float64
	LimitMMPerPrompt          map[string]int
	GenerationConfig          *string
	HFOverrides               map[string]any
	HFOverridesSet            bool
	LongPrefillTokenThreshold *int
	DefaultSpeculator         string
	MTPTokens                 int
	DFlash2Tokens             *int
	DFlash2Model              *string
	MTPMoEQuantization        string
	MTPAttentionBackend       *string
	MTPModel                  *string
	MTPDraftSampleMethod      *string
	CUDADeviceMaxConnections  *int
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
