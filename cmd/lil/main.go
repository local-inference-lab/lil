// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/spf13/pflag"

	configassets "github.com/local-inference-lab/lil/configs"
	"github.com/local-inference-lab/lil/internal/hfhub"
	"github.com/local-inference-lab/lil/internal/launcher"
)

var version = "dev"

const modelRepositoryOwner = "local-inference-lab"

type launchFlags struct {
	config                    string
	modelsConfig              string
	tp                        int
	devices                   string
	modelPath                 string
	servedModelName           string
	host                      string
	port                      int
	gpuMemoryUtilization      float64
	kvCacheMemoryBytes        string
	kvCacheDType              string
	maxModelLen               string
	maxNumSeqs                int
	maxNumBatchedTokens       int
	dcp                       int
	dcpCommBackend            string
	speculator                string
	speculativeTokens         int
	adaptiveSpeculativeTokens bool
	adaptiveWindow            int
	adaptiveInitial           int
	pleCPUOffload             bool
	noPLECPUOffload           bool
	b12xPolicyMode            string
	profileDir                string
	profileRecordShapes       bool
	profileWithMemory         bool
	profileWithFlops          bool
	profileNoStack            bool
	profileNoGzip             bool
	profileMaxIterations      int
	environment               []string
	detach                    bool
	syncCode                  bool
	syncModel                 bool
}

func configureLaunchFlags(fs *pflag.FlagSet, values *launchFlags, includeSync bool) {
	fs.StringVar(&values.config, "config", "", "discovered topology name or YAML path")
	fs.StringVar(&values.modelsConfig, "models-config", "", "directory of <model>/lil.yaml manifests with config.json and safetensors index")
	fs.IntVar(&values.tp, "tp", 0, "tensor parallel size")
	fs.StringVar(&values.devices, "devices", "", "comma-separated local physical GPU IDs")
	fs.StringVar(&values.modelPath, "model-path", "", "local checkpoint path")
	fs.StringVar(&values.servedModelName, "served-model-name", "", "served API model name")
	fs.StringVar(&values.host, "host", "", "API bind address")
	fs.IntVar(&values.port, "port", 0, "API port")
	fs.Float64Var(&values.gpuMemoryUtilization, "gpu-memory-utilization", 0, "vLLM GPU memory utilization")
	fs.StringVar(&values.kvCacheMemoryBytes, "kv-cache-memory-bytes", "", "explicit KV allocation or auto")
	fs.StringVar(&values.kvCacheDType, "kv-cache-dtype", "fp8", "KV cache dtype")
	fs.StringVar(&values.maxModelLen, "max-model-len", "", "maximum model length")
	fs.IntVar(&values.maxNumSeqs, "max-num-seqs", 0, "maximum concurrent sequences")
	fs.IntVar(&values.maxNumBatchedTokens, "max-num-batched-tokens", 0, "maximum batched tokens")
	fs.IntVar(&values.dcp, "dcp", 1, "decode context parallel size")
	fs.StringVar(&values.dcpCommBackend, "dcp-comm-backend", "a2a", "DCP communication backend")
	fs.StringVar(&values.speculator, "speculator", "", "mtp, dflash, or none")
	fs.IntVar(&values.speculativeTokens, "speculative-tokens", 0, "speculative token count")
	fs.BoolVar(&values.adaptiveSpeculativeTokens, "adaptive-speculative-tokens", false, "enable adaptive MTP depth")
	fs.IntVar(&values.adaptiveWindow, "adaptive-window", 32, "adaptive MTP window")
	fs.IntVar(&values.adaptiveInitial, "adaptive-initial", 0, "adaptive MTP initial depth")
	fs.BoolVar(&values.pleCPUOffload, "ple-cpu-offload", false, "enable mapped-host PLE")
	fs.BoolVar(&values.noPLECPUOffload, "no-ple-cpu-offload", false, "disable mapped-host PLE")
	fs.StringVar(&values.b12xPolicyMode, "b12x-policy-mode", "auto", "B12X policy mode")
	fs.StringVar(&values.profileDir, "profile-dir", "", "torch profiler output directory")
	fs.BoolVar(&values.profileRecordShapes, "profile-record-shapes", false, "record profiler shapes")
	fs.BoolVar(&values.profileWithMemory, "profile-with-memory", false, "record profiler memory")
	fs.BoolVar(&values.profileWithFlops, "profile-with-flops", false, "record profiler FLOPs")
	fs.BoolVar(&values.profileNoStack, "profile-no-stack", false, "disable profiler stacks")
	fs.BoolVar(&values.profileNoGzip, "profile-no-gzip", false, "disable profiler gzip")
	fs.IntVar(&values.profileMaxIterations, "profile-max-iterations", 4, "maximum profiled iterations")
	fs.StringArrayVar(&values.environment, "env", nil, "NAME=VALUE environment entry overriding topology or manifest tuning")
	fs.BoolVar(&values.detach, "detach", false, "leave Spark containers running")
	if includeSync {
		fs.BoolVar(&values.syncCode, "sync-code", false, "replace remote vllm/ and b12x/ packages before preflight")
		fs.BoolVar(&values.syncModel, "sync-model", false, "mirror --model-path to every selected Spark rank before preflight")
	}
}

func pointerIfChanged[T any](fs *pflag.FlagSet, name string, value T) *T {
	if fs.Changed(name) {
		return &value
	}
	return nil
}

func parseDevices(value string) ([]int, error) {
	parts := strings.Split(value, ",")
	devices := make([]int, 0, len(parts))
	seen := map[int]bool{}
	for _, part := range parts {
		device, err := strconv.Atoi(part)
		if err != nil || device < 0 || seen[device] {
			return nil, fmt.Errorf("--devices must contain unique non-negative integers")
		}
		seen[device] = true
		devices = append(devices, device)
	}
	return devices, nil
}

func parseEnvironment(values []string) ([]launcher.EnvironmentOverride, error) {
	overrides := make([]launcher.EnvironmentOverride, 0, len(values))
	for _, value := range values {
		name, contents, ok := strings.Cut(value, "=")
		if !ok || name == "" {
			return nil, fmt.Errorf("--env must have NAME=VALUE form")
		}
		overrides = append(overrides, launcher.EnvironmentOverride{Name: name, Value: contents})
	}
	return overrides, nil
}

func (values launchFlags) options(fs *pflag.FlagSet, extraArgs []string) (launcher.LaunchOptions, error) {
	options := launcher.LaunchOptions{
		TPSize:               pointerIfChanged(fs, "tp", values.tp),
		ModelPath:            pointerIfChanged(fs, "model-path", values.modelPath),
		ServedModelName:      pointerIfChanged(fs, "served-model-name", values.servedModelName),
		Host:                 pointerIfChanged(fs, "host", values.host),
		Port:                 pointerIfChanged(fs, "port", values.port),
		GPUMemoryUtilization: pointerIfChanged(fs, "gpu-memory-utilization", values.gpuMemoryUtilization),
		KVCacheMemoryBytes:   pointerIfChanged(fs, "kv-cache-memory-bytes", values.kvCacheMemoryBytes),
		KVCacheDType:         values.kvCacheDType,
		MaxModelLen:          pointerIfChanged(fs, "max-model-len", values.maxModelLen),
		MaxNumSeqs:           pointerIfChanged(fs, "max-num-seqs", values.maxNumSeqs),
		MaxNumBatchedTokens:  pointerIfChanged(fs, "max-num-batched-tokens", values.maxNumBatchedTokens),
		DCPSize:              values.dcp, DCPCommBackend: values.dcpCommBackend,
		Speculator:                pointerIfChanged(fs, "speculator", values.speculator),
		SpeculativeTokens:         pointerIfChanged(fs, "speculative-tokens", values.speculativeTokens),
		AdaptiveSpeculativeTokens: values.adaptiveSpeculativeTokens,
		AdaptiveWindow:            values.adaptiveWindow,
		AdaptiveInitial:           pointerIfChanged(fs, "adaptive-initial", values.adaptiveInitial),
		B12XPolicyMode:            values.b12xPolicyMode, ExtraVLLMArgs: extraArgs,
		Detach: values.detach, SyncCode: values.syncCode, SyncModel: values.syncModel,
	}
	if fs.Changed("devices") {
		devices, err := parseDevices(values.devices)
		if err != nil {
			return options, err
		}
		options.DeviceIDs = devices
	}
	if fs.Changed("ple-cpu-offload") && fs.Changed("no-ple-cpu-offload") {
		return options, fmt.Errorf("--ple-cpu-offload and --no-ple-cpu-offload are mutually exclusive")
	}
	if fs.Changed("ple-cpu-offload") {
		enabled := true
		options.PLECPUOffload = &enabled
	}
	if fs.Changed("no-ple-cpu-offload") {
		enabled := false
		options.PLECPUOffload = &enabled
	}
	overrides, err := parseEnvironment(values.environment)
	if err != nil {
		return options, err
	}
	options.EnvironmentOverrides = overrides
	if fs.Changed("profile-dir") {
		options.Profiler = &launcher.ProfilerOptions{
			OutputDir: values.profileDir, RecordShapes: values.profileRecordShapes,
			WithMemory: values.profileWithMemory, WithStack: !values.profileNoStack,
			WithFlops: values.profileWithFlops, UseGzip: !values.profileNoGzip,
			MaxIterations: values.profileMaxIterations,
		}
	}
	return options, nil
}

func embeddedModelFamilies() ([]byte, error) {
	return configassets.ModelBases.ReadFile("models/_bases.yaml")
}

// hubFacts reads config.json and the safetensors shard sizes at the
// repository's resolved commit.
func hubFacts(ctx context.Context, client *hfhub.Client, repository hfhub.Repository) (*launcher.CheckpointFacts, error) {
	sizes, _, err := client.FetchSizes(ctx, repository)
	if err != nil {
		return nil, err
	}
	data, _, err := client.FetchFile(ctx, repository, "config.json", hfhub.ConfigLimit)
	if err != nil {
		return nil, err
	}
	source := repository.ID + "@" + repository.Revision[:12]
	config, err := launcher.ParseCheckpointConfig(data, source+" config.json")
	if err != nil {
		return nil, err
	}
	return launcher.FactsFromConfig(config, hfhub.SafetensorsBytes(sizes), "Hub safetensors file sizes", source)
}

func hubModelProfile(ctx context.Context, client *hfhub.Client, families []byte, repository hfhub.Repository) (launcher.ModelProfile, string, error) {
	label := repository.ID + "/lil.yaml"
	manifest, _, err := client.FetchManifest(ctx, repository)
	if err != nil {
		return launcher.ModelProfile{}, "", err
	}
	kind, err := launcher.RepositoryManifestKind(manifest, label)
	if err != nil {
		return launcher.ModelProfile{}, "", err
	}
	if kind == "draft" {
		_, err := launcher.LoadRepositoryDraftProfile(manifest, repository.ID, repository.Revision, label)
		return launcher.ModelProfile{}, kind, err
	}
	profile, err := launcher.LoadRepositoryModelProfile(families, manifest, repository.ID, repository.Revision, label)
	if err != nil {
		return launcher.ModelProfile{}, "", err
	}
	profile.Facts, err = hubFacts(ctx, client, repository)
	if err != nil {
		return launcher.ModelProfile{}, "", err
	}
	return profile, kind, nil
}

func loadProfiles(ctx context.Context, path string) (map[string]launcher.ModelProfile, error) {
	families, err := embeddedModelFamilies()
	if err != nil {
		return nil, err
	}
	if path == "" {
		client, err := hfhub.DefaultClient(modelRepositoryOwner)
		if err != nil {
			return nil, err
		}
		discovery, err := client.Discover(ctx)
		if err != nil {
			return nil, err
		}
		if discovery.Warning != "" {
			fmt.Fprintln(os.Stderr, "warning:", discovery.Warning)
		}
		profiles := map[string]launcher.ModelProfile{}
		for _, repository := range discovery.Repositories {
			if !repository.HasFile("lil.yaml") {
				continue
			}
			profile, kind, err := hubModelProfile(ctx, client, families, repository)
			if err != nil {
				return nil, err
			}
			if kind == "model" {
				profiles[profile.Name] = profile
			}
		}
		return profiles, nil
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(abs)
	if err != nil {
		return nil, fmt.Errorf("cannot read model manifest directory %s: %w", abs, err)
	}
	profiles := map[string]launcher.ModelProfile{}
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		directory := filepath.Join(abs, entry.Name())
		manifestPath := filepath.Join(directory, "lil.yaml")
		data, err := os.ReadFile(manifestPath)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return nil, fmt.Errorf("cannot read %s: %w", manifestPath, err)
		}
		kind, err := launcher.RepositoryManifestKind(data, manifestPath)
		if err != nil {
			return nil, err
		}
		repositoryID := modelRepositoryOwner + "/" + entry.Name()
		if kind == "draft" {
			if _, err := launcher.LoadRepositoryDraftProfile(data, repositoryID, "", manifestPath); err != nil {
				return nil, err
			}
			continue
		}
		profile, err := launcher.LoadRepositoryModelProfile(families, data, repositoryID, "", manifestPath)
		if err != nil {
			return nil, err
		}
		profile.Facts, err = launcher.LoadCheckpointFacts(directory)
		if err != nil {
			return nil, err
		}
		profiles[profile.Name] = profile
	}
	if len(profiles) == 0 {
		return nil, fmt.Errorf("no <model>/lil.yaml manifests were found under %s", abs)
	}
	return profiles, nil
}

func profileFromMap(profiles map[string]launcher.ModelProfile, model string) (launcher.ModelProfile, error) {
	name := model
	if _, basename, ok := strings.Cut(model, "/"); ok {
		name = basename
	}
	profile, ok := profiles[name]
	if !ok {
		return launcher.ModelProfile{}, fmt.Errorf(
			"unknown model profile %q; choose one of: %s",
			model, strings.Join(launcher.SortedProfileNames(profiles), ", "),
		)
	}
	return profile, nil
}

func resolveModelProfile(ctx context.Context, model, modelsConfig string) (launcher.ModelProfile, error) {
	if modelsConfig != "" {
		profiles, err := loadProfiles(ctx, modelsConfig)
		if err != nil {
			return launcher.ModelProfile{}, err
		}
		return profileFromMap(profiles, model)
	}
	client, err := hfhub.DefaultClient(modelRepositoryOwner)
	if err != nil {
		return launcher.ModelProfile{}, err
	}
	families, err := embeddedModelFamilies()
	if err != nil {
		return launcher.ModelProfile{}, err
	}
	repository, err := client.Resolve(ctx, model)
	if err != nil {
		return launcher.ModelProfile{}, err
	}
	profile, kind, err := hubModelProfile(ctx, client, families, repository)
	if err != nil {
		return launcher.ModelProfile{}, err
	}
	if kind == "draft" {
		return launcher.ModelProfile{}, fmt.Errorf("%s is a DFlash draft checkpoint and cannot be launched as a serving model", repository.ID)
	}
	return profile, nil
}

func validateHubDraftProfile(ctx context.Context, profile launcher.ModelProfile) error {
	if profile.Speculators.DFlash == nil {
		return fmt.Errorf("model %q does not define a DFlash draft repository", profile.Name)
	}
	client, err := hfhub.DefaultClient(modelRepositoryOwner)
	if err != nil {
		return err
	}
	repository, err := client.Resolve(ctx, profile.Speculators.DFlash.Model)
	if err != nil {
		return err
	}
	manifest, _, err := client.FetchManifest(ctx, repository)
	if err != nil {
		return err
	}
	draft, err := launcher.LoadRepositoryDraftProfile(
		manifest, repository.ID, repository.Revision, repository.ID+"/lil.yaml",
	)
	if err != nil {
		return err
	}
	for _, compatible := range draft.CompatibleModels {
		if compatible == profile.Model {
			return nil
		}
	}
	return fmt.Errorf(
		"DFlash repository %s is not compatible with %s according to its latest lil.yaml",
		draft.RepositoryID, profile.Model,
	)
}

func buildSpec(ctx context.Context, fs *pflag.FlagSet, values launchFlags) (launcher.LaunchSpec, error) {
	args := fs.Args()
	dash := fs.ArgsLenAtDash()
	if (dash < 0 && len(args) != 1) || (dash >= 0 && dash != 1) {
		return launcher.LaunchSpec{}, fmt.Errorf("exactly one MODEL is required; forward vLLM arguments after --")
	}
	extraArgs := []string{}
	if dash >= 0 {
		extraArgs = append(extraArgs, args[dash:]...)
	}
	profile, err := resolveModelProfile(ctx, args[0], values.modelsConfig)
	if err != nil {
		return launcher.LaunchSpec{}, err
	}
	topology, err := loadTopology(values.config)
	if err != nil {
		return launcher.LaunchSpec{}, err
	}
	options, err := values.options(fs, extraArgs)
	if err != nil {
		return launcher.LaunchSpec{}, err
	}
	if options.Detach && topology.Local != nil {
		return launcher.LaunchSpec{}, fmt.Errorf("--detach is supported only by Spark/RDMA")
	}
	if options.SyncCode && topology.Local != nil {
		return launcher.LaunchSpec{}, fmt.Errorf("--sync-code is supported only by Spark/RDMA")
	}
	if options.SyncModel && topology.Local != nil {
		return launcher.LaunchSpec{}, fmt.Errorf("--sync-model is supported only by Spark/RDMA")
	}
	spec, err := launcher.BuildLaunchSpec(profile, topology, options)
	if err != nil {
		return launcher.LaunchSpec{}, err
	}
	if values.modelsConfig == "" && spec.Metadata["speculator"] == "dflash" && spec.Metadata["speculative_tokens"] != 0 {
		if err := validateHubDraftProfile(ctx, profile); err != nil {
			return launcher.LaunchSpec{}, err
		}
	}
	return spec, nil
}

func newFlagSet(command string) *pflag.FlagSet {
	fs := pflag.NewFlagSet(command, pflag.ContinueOnError)
	fs.SetInterspersed(true)
	fs.SetOutput(os.Stderr)
	fs.Usage = func() {
		if command == "list" {
			fmt.Fprintln(os.Stderr, "Usage: lil list [--models-config PATH]")
		} else {
			fmt.Fprintf(
				os.Stderr,
				"Usage: lil %s MODEL [options] [-- VLLM_ARGUMENTS...]\n",
				command,
			)
		}
		fs.PrintDefaults()
	}
	return fs
}

func listCommand(ctx context.Context, args []string) error {
	fs := newFlagSet("list")
	modelsConfig := fs.String("models-config", "", "directory of <model>/lil.yaml manifests")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return fmt.Errorf("list takes no positional arguments")
	}
	profiles, err := loadProfiles(ctx, *modelsConfig)
	if err != nil {
		return err
	}
	return printList(profiles)
}

func printChecks(results []launcher.CheckResult) bool {
	passed := true
	for _, result := range results {
		fmt.Printf("%-4s  %s: %s\n", result.Status, result.Name, result.Detail)
		passed = passed && !result.Failed()
	}
	return passed
}

func renderCommand(ctx context.Context, args []string) error {
	fs := newFlagSet("render")
	var values launchFlags
	configureLaunchFlags(fs, &values, false)
	format := fs.String("format", "shell", "shell or json")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *format != "shell" && *format != "json" {
		return fmt.Errorf("--format must be shell or json")
	}
	spec, err := buildSpec(ctx, fs, values)
	if err != nil {
		return err
	}
	if *format == "json" {
		encoded, err := launcher.JSONRender(spec)
		if err != nil {
			return err
		}
		fmt.Println(string(encoded))
	} else {
		fmt.Println(launcher.ShellRender(spec))
	}
	return nil
}

func checkCommand(ctx context.Context, args []string) (bool, error) {
	fs := newFlagSet("check")
	var values launchFlags
	configureLaunchFlags(fs, &values, false)
	if err := fs.Parse(args); err != nil {
		return false, err
	}
	spec, err := buildSpec(ctx, fs, values)
	if err != nil {
		return false, err
	}
	return printChecks(launcher.RunChecks(ctx, spec, true)), nil
}

func runCommand(ctx context.Context, args []string) (int, error) {
	fs := newFlagSet("run")
	var values launchFlags
	configureLaunchFlags(fs, &values, true)
	if err := fs.Parse(args); err != nil {
		return 2, err
	}
	spec, err := buildSpec(ctx, fs, values)
	if err != nil {
		return 2, err
	}
	if spec.Topology.Spark != nil && spec.SyncCode && !launcher.SyncSparkSources(ctx, spec) {
		return 1, nil
	}
	if spec.Topology.Spark != nil && spec.SyncModel {
		if err := launcher.SyncSparkModel(ctx, spec); err != nil {
			return 1, err
		}
	}
	if spec.Topology.Spark != nil {
		if err := launcher.PrepareSparkProfilerDirectories(ctx, spec); err != nil {
			return 1, err
		}
	}
	if !printChecks(launcher.RunChecks(ctx, spec, true)) {
		return 1, nil
	}
	if err := launcher.SyncHuggingFaceCache(ctx, spec); err != nil {
		return 1, err
	}
	if spec.Topology.Spark != nil {
		if len(spec.DownloadRepositories) > 0 && !printChecks(launcher.RecheckSparkNodes(ctx, spec)) {
			return 1, nil
		}
		return launcher.RunSparkCluster(ctx, spec), nil
	}
	if outputDir, ok := launcher.ProfilerOutputDir(spec.VLLMArgv); ok {
		if err := os.MkdirAll(outputDir, 0o755); err != nil {
			return 1, err
		}
	}
	environment := map[string]string{}
	for _, value := range os.Environ() {
		name, contents, ok := strings.Cut(value, "=")
		if ok {
			environment[name] = contents
		}
	}
	for _, name := range spec.UnsetEnvironment {
		delete(environment, name)
	}
	for name, value := range spec.HostEnvironment {
		environment[name] = value
	}
	names := make([]string, 0, len(environment))
	for name := range environment {
		names = append(names, name)
	}
	sort.Strings(names)
	env := make([]string, 0, len(names))
	for _, name := range names {
		env = append(env, name+"="+environment[name])
	}
	if err := os.Chdir(spec.Topology.RepoRoot()); err != nil {
		return 1, err
	}
	if err := syscall.Exec(spec.CommandArgv[0], spec.CommandArgv, env); err != nil {
		return 127, err
	}
	return 127, nil
}

type clusterFlags struct {
	config       string
	modelsConfig string
	tp           int
	port         int
	rank         int
	follow       bool
	tail         int
	timeout      time.Duration
}

func clusterTarget(ctx context.Context, fs *pflag.FlagSet, values clusterFlags) (launcher.SparkTarget, error) {
	if fs.NArg() != 1 {
		return launcher.SparkTarget{}, fmt.Errorf("cluster operation requires exactly one MODEL")
	}
	profile, err := resolveModelProfile(ctx, fs.Arg(0), values.modelsConfig)
	if err != nil {
		return launcher.SparkTarget{}, err
	}
	config := values.config
	if config == "" {
		config = "spark"
	}
	topology, err := loadTopology(config)
	if err != nil {
		return launcher.SparkTarget{}, err
	}
	return launcher.ResolveSparkTarget(
		profile, topology,
		pointerIfChanged(fs, "tp", values.tp),
		pointerIfChanged(fs, "port", values.port),
	)
}

func clusterCommand(ctx context.Context, args []string) (int, error) {
	if len(args) == 0 {
		return 2, fmt.Errorf("cluster action is required: status, logs, wait, stop, profile-start, or profile-stop")
	}
	if args[0] == "--help" || args[0] == "-h" || args[0] == "help" {
		fmt.Fprintln(os.Stderr, `Usage: lil cluster ACTION MODEL [options]

Actions:
  status         Show container state on every selected rank.
  logs           Show or follow logs from one rank (default: the head).
  wait           Wait for every rank and the head API to become ready.
  stop           Stop containers on every selected rank and remove them.
  profile-start  Start the configured vLLM profiler.
  profile-stop   Stop the configured vLLM profiler.`)
		return 0, nil
	}
	action := args[0]
	defaultTimeout := 30 * time.Minute
	if action == "profile-start" {
		defaultTimeout = time.Minute
	}
	fs := pflag.NewFlagSet("cluster "+action, pflag.ContinueOnError)
	fs.SetInterspersed(true)
	fs.SetOutput(os.Stderr)
	values := clusterFlags{}
	fs.StringVar(&values.config, "config", "", "discovered Spark topology name or YAML path")
	fs.StringVar(&values.modelsConfig, "models-config", "", "directory of <model>/lil.yaml manifests")
	fs.IntVar(&values.tp, "tp", 0, "tensor parallel size used by the cluster")
	fs.IntVar(&values.port, "port", 0, "head API port")
	fs.IntVar(&values.rank, "rank", 0, "rank whose logs to show")
	fs.BoolVarP(&values.follow, "follow", "f", false, "follow container logs")
	fs.IntVar(&values.tail, "tail", 120, "log lines to show")
	fs.DurationVar(&values.timeout, "timeout", defaultTimeout, "readiness or profile request timeout")
	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: lil cluster %s MODEL [options]\n", action)
		fs.PrintDefaults()
	}
	if err := fs.Parse(args[1:]); err != nil {
		return 2, err
	}
	if action != "logs" && (fs.Changed("follow") || fs.Changed("tail") || fs.Changed("rank")) {
		return 2, fmt.Errorf("--follow, --tail, and --rank are valid only for cluster logs")
	}
	if action != "wait" && action != "profile-start" && action != "profile-stop" && fs.Changed("timeout") {
		return 2, fmt.Errorf("--timeout is valid only for cluster wait or profiler control")
	}
	target, err := clusterTarget(ctx, fs, values)
	if err != nil {
		return 2, err
	}
	switch action {
	case "status":
		statuses, err := launcher.InspectSparkTarget(ctx, target)
		if err != nil {
			return 1, err
		}
		printClusterStatus(target, statuses)
		return 0, nil
	case "logs":
		return launcher.RunSparkLogs(ctx, target, values.rank, values.follow, values.tail)
	case "wait":
		if err := launcher.WaitSparkReady(ctx, target, values.timeout); err != nil {
			return 1, err
		}
		fmt.Printf("ready  http://%s:%d\n", target.Nodes[0].SSHHost, target.Port)
		return 0, nil
	case "stop":
		if err := launcher.StopSparkTarget(ctx, target); err != nil {
			return 1, err
		}
		return 0, nil
	case "profile-start":
		if err := launcher.SparkProfileRequest(ctx, target, true, values.timeout); err != nil {
			return 1, err
		}
		return 0, nil
	case "profile-stop":
		if err := launcher.SparkProfileRequest(ctx, target, false, values.timeout); err != nil {
			return 1, err
		}
		return 0, nil
	default:
		return 2, fmt.Errorf("unknown cluster action %q", action)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, `Usage: lil COMMAND [options]

Build deterministic local or Spark/RDMA vLLM launches.

Commands:
  list      Discover launchable Hugging Face models and local topologies.
  discover  Discover and save a local or Spark/RDMA topology.
  render    Render the resolved environment and command without executing it.
  check     Run read-only launch preflight checks.
  run       Run preflight checks, then execute the launch.
  cluster   Inspect and control a Spark/RDMA launch.

Run "lil COMMAND --help" for command options.`)
}

func commandErrorStatus(err error) (int, error) {
	if errors.Is(err, pflag.ErrHelp) {
		return 0, nil
	}
	if err != nil {
		return 2, err
	}
	return 0, nil
}

func execute(ctx context.Context, args []string) (int, error) {
	if len(args) == 0 {
		usage()
		return 2, errors.New("a command is required")
	}
	if args[0] == "--help" || args[0] == "-h" || args[0] == "help" {
		usage()
		return 0, nil
	}
	if args[0] == "--version" || args[0] == "version" {
		fmt.Println("lil", version)
		return 0, nil
	}
	switch args[0] {
	case "list":
		return commandErrorStatus(listCommand(ctx, args[1:]))
	case "discover":
		return commandErrorStatus(discoverCommand(ctx, args[1:]))
	case "render":
		return commandErrorStatus(renderCommand(ctx, args[1:]))
	case "check":
		passed, err := checkCommand(ctx, args[1:])
		if errors.Is(err, pflag.ErrHelp) {
			return 0, nil
		}
		if err != nil {
			return 2, err
		}
		if !passed {
			return 1, nil
		}
		return 0, nil
	case "run":
		status, err := runCommand(ctx, args[1:])
		if errors.Is(err, pflag.ErrHelp) {
			return 0, nil
		}
		return status, err
	case "cluster":
		status, err := clusterCommand(ctx, args[1:])
		if errors.Is(err, pflag.ErrHelp) {
			return 0, nil
		}
		return status, err
	default:
		usage()
		return 2, fmt.Errorf("unknown command %q", args[0])
	}
}

// interruptContext cancels on the first interrupt and restores default
// signal handling afterwards, so a second interrupt terminates the process
// even while cleanup is in progress.
func interruptContext() (context.Context, context.CancelFunc) {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	go func() {
		<-ctx.Done()
		stop()
	}()
	return ctx, stop
}

func main() {
	ctx, stop := interruptContext()
	defer stop()
	status, err := execute(ctx, os.Args[1:])
	if err != nil {
		fmt.Fprintln(os.Stderr, "lil:", err)
	}
	os.Exit(status)
}
