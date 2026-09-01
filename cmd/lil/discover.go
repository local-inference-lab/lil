// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"

	"github.com/spf13/pflag"

	"github.com/local-inference-lab/lil/internal/launcher"
)

var topologyNamePattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.-]*$`)

type discoveryFlags struct {
	name                  string
	host                  string
	port                  int
	gpuMemoryUtilization  float64
	repoRoot              string
	python                string
	b12xRoot              string
	cudaHome              string
	output                string
	force                 bool
	nodes                 []string
	runtimeRepoRoot       string
	runtimePython         string
	vllmBin               string
	runtimeB12XRoot       string
	masterPort            int
	image                 string
	containerNamePrefix   string
	containerMemoryGB     int
	containerMemorySwapGB int
	containerShmGB        int
	containerPidsLimit    int
	containerNofileLimit  int
	ncclDebug             string
	deviceID              int
}

func configureDiscoveryFlags(fs *pflag.FlagSet, values *discoveryFlags) {
	fs.StringVar(&values.name, "name", "", "topology name")
	fs.StringVar(&values.host, "host", "0.0.0.0", "API bind address")
	fs.IntVar(&values.port, "port", 8000, "API port")
	fs.Float64Var(&values.gpuMemoryUtilization, "gpu-memory-utilization", launcher.DefaultGPUMemoryUtilization, "default vLLM GPU memory utilization")
	fs.StringVar(&values.repoRoot, "repo-root", "", "controller vLLM source root")
	fs.StringVar(&values.python, "python", "", "controller vLLM Python executable")
	fs.StringVar(&values.b12xRoot, "b12x-root", "", "controller B12X source root")
	fs.StringVar(&values.cudaHome, "cuda-home", "", "CUDA installation root")
	fs.StringVarP(&values.output, "output", "o", "", "output YAML path or - for stdout")
	fs.BoolVar(&values.force, "force", false, "replace an existing topology file")
	fs.StringArrayVar(&values.nodes, "node", nil, "Spark SSH host, repeated in rank order")
	fs.StringVar(&values.runtimeRepoRoot, "runtime-repo-root", "", "remote vLLM source root")
	fs.StringVar(&values.runtimePython, "runtime-python", "", "remote vLLM Python executable")
	fs.StringVar(&values.vllmBin, "vllm-bin", "", "remote vLLM executable")
	fs.StringVar(&values.runtimeB12XRoot, "runtime-b12x-root", "", "remote B12X source root")
	fs.IntVar(&values.masterPort, "master-port", 29638, "torch distributed master port")
	fs.StringVar(&values.image, "image", "", "Spark Docker image")
	fs.StringVar(&values.containerNamePrefix, "container-name-prefix", "vllm-fleet", "Spark container name prefix")
	fs.IntVar(&values.containerMemoryGB, "container-memory-gb", 108, "Spark container memory limit")
	fs.IntVar(&values.containerMemorySwapGB, "container-memory-swap-gb", 112, "Spark container memory plus swap limit")
	fs.IntVar(&values.containerShmGB, "container-shm-gb", 64, "Spark container shared memory size")
	fs.IntVar(&values.containerPidsLimit, "container-pids-limit", 4096, "Spark container PID limit")
	fs.IntVar(&values.containerNofileLimit, "container-nofile-limit", 1048576, "Spark container nofile limit")
	fs.StringVar(&values.ncclDebug, "nccl-debug", "INFO", "NCCL log level")
	fs.IntVar(&values.deviceID, "device-id", 0, "physical GPU ID on each Spark node")
}

func writeTopology(path string, data []byte, force bool) error {
	abs, err := filepath.Abs(path)
	if err != nil {
		return err
	}
	if !force {
		if _, err := os.Stat(abs); err == nil {
			return fmt.Errorf("topology already exists: %s (pass --force to replace it)", abs)
		} else if !os.IsNotExist(err) {
			return err
		}
	}
	if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
		return err
	}
	temporary, err := os.CreateTemp(filepath.Dir(abs), ".lil-topology-*.yaml")
	if err != nil {
		return err
	}
	temporaryName := temporary.Name()
	removeTemporary := true
	defer func() {
		if removeTemporary {
			_ = os.Remove(temporaryName)
		}
	}()
	if _, err := temporary.Write(data); err != nil {
		_ = temporary.Close()
		return err
	}
	if err := temporary.Chmod(0o644); err != nil {
		_ = temporary.Close()
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	if err := os.Rename(temporaryName, abs); err != nil {
		return err
	}
	removeTemporary = false
	return nil
}

func discoverCommand(args []string) error {
	fs := pflag.NewFlagSet("discover", pflag.ContinueOnError)
	fs.SetInterspersed(true)
	fs.SetOutput(os.Stderr)
	values := discoveryFlags{}
	configureDiscoveryFlags(fs, &values)
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "Usage: lil discover [local|spark] [options]")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return err
	}
	kind := "local"
	if fs.NArg() == 1 {
		kind = fs.Arg(0)
	} else if fs.NArg() > 1 {
		return fmt.Errorf("discover accepts at most one topology kind")
	}
	if kind == "spark_rdma" {
		kind = "spark"
	}
	if kind != "local" && kind != "spark" {
		return fmt.Errorf("topology kind must be local or spark")
	}
	var topology launcher.Topology
	var err error
	if kind == "local" {
		if len(values.nodes) != 0 {
			return fmt.Errorf("--node is valid only for Spark/RDMA discovery")
		}
		topology, err = launcher.DiscoverLocalTopology(launcher.LocalDiscoveryOptions{
			Name: values.name, Host: values.host, Port: values.port,
			GPUMemoryUtilization: values.gpuMemoryUtilization,
			RepoRoot:             values.repoRoot, Python: values.python,
			B12XRoot: values.b12xRoot, CUDAHome: values.cudaHome,
		})
	} else {
		topology, err = launcher.DiscoverSparkTopology(launcher.SparkDiscoveryOptions{
			Name: values.name, Host: values.host, Port: values.port,
			GPUMemoryUtilization: values.gpuMemoryUtilization,
			RepoRoot:             values.repoRoot, Python: values.python,
			B12XRoot: values.b12xRoot, RuntimeRepoRoot: values.runtimeRepoRoot,
			RuntimePython: values.runtimePython, VLLMBin: values.vllmBin,
			RuntimeB12XRoot: values.runtimeB12XRoot, Nodes: values.nodes,
			MasterPort: values.masterPort, Image: values.image,
			ContainerNamePrefix:   values.containerNamePrefix,
			ContainerMemoryGB:     values.containerMemoryGB,
			ContainerMemorySwapGB: values.containerMemorySwapGB,
			ContainerShmGB:        values.containerShmGB,
			ContainerPidsLimit:    values.containerPidsLimit,
			ContainerNofileLimit:  values.containerNofileLimit,
			CUDAHome:              values.cudaHome, NCCLDebug: values.ncclDebug,
			DeviceID: values.deviceID,
		})
	}
	if err != nil {
		return err
	}
	if !topologyNamePattern.MatchString(topology.Name()) {
		return fmt.Errorf("topology name %q is not safe as a filename", topology.Name())
	}
	data, err := launcher.MarshalTopology(topology)
	if err != nil {
		return err
	}
	if _, err := launcher.LoadTopology(data, "discovered topology", "."); err != nil {
		return fmt.Errorf("generated topology failed validation: %w", err)
	}
	if values.output == "-" {
		_, err = os.Stdout.Write(data)
		return err
	}
	output := values.output
	if output == "" {
		directory, err := topologyConfigDir()
		if err != nil {
			return err
		}
		output = filepath.Join(directory, topology.Name()+".yaml")
	}
	if err := writeTopology(output, data, values.force); err != nil {
		return err
	}
	abs, _ := filepath.Abs(output)
	printDiscoveryResult(topology, abs)
	return nil
}
