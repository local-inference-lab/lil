// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package launcher

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

func executableFile(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.Mode().IsRegular() && info.Mode().Perm()&0o111 != 0
}

func localHFCLI(spec LaunchSpec) string {
	candidate := filepath.Join(filepath.Dir(spec.Topology.Python()), "hf")
	if executableFile(candidate) {
		return candidate
	}
	path, err := exec.LookPath("hf")
	if err == nil {
		return path
	}
	return ""
}

func onlineEnvironment(spec LaunchSpec) []string {
	values := map[string]string{}
	for _, entry := range os.Environ() {
		name, value, ok := strings.Cut(entry, "=")
		if ok {
			values[name] = value
		}
	}
	for _, name := range spec.UnsetEnvironment {
		delete(values, name)
	}
	for name, value := range spec.HostEnvironment {
		values[name] = value
	}
	for _, name := range []string{
		"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_PROGRESS_BARS",
	} {
		delete(values, name)
	}
	environment := make([]string, 0, len(values))
	for _, name := range sortedMapKeys(values) {
		environment = append(environment, name+"="+values[name])
	}
	return environment
}

func runVisible(argv []string, environment []string) error {
	command := exec.Command(argv[0], argv[1:]...)
	command.Stdin, command.Stdout, command.Stderr = os.Stdin, os.Stdout, os.Stderr
	if environment != nil {
		command.Env = environment
	}
	return command.Run()
}

func warnUnmanagedDownload(repository, location string) {
	fmt.Fprintf(
		os.Stderr,
		"warning: hf is unavailable %s; vLLM may silently download %s during startup\n",
		location, repository,
	)
}

func syncLocalHuggingFaceCache(spec LaunchSpec) error {
	hf := localHFCLI(spec)
	if hf == "" {
		for _, repository := range spec.DownloadRepositories {
			warnUnmanagedDownload(repository, "next to the configured vLLM Python and on PATH")
		}
		return nil
	}
	for _, repository := range spec.DownloadRepositories {
		fmt.Fprintf(os.Stderr, "updating Hugging Face cache for %s\n", repository)
		if err := runVisible([]string{hf, "download", repository}, onlineEnvironment(spec)); err != nil {
			return fmt.Errorf("Hugging Face download failed for %s: %w", repository, err)
		}
	}
	return nil
}

func huggingFaceCacheMount(topology *SparkRDMATopology) *CacheMount {
	for index := range topology.CacheMounts {
		mount := &topology.CacheMounts[index]
		if mount.Target == "/root/.cache/huggingface" {
			return mount
		}
	}
	return nil
}

func sparkHostHFArgv(
	topology *SparkRDMATopology, hf string, arguments ...string,
) []string {
	mount := huggingFaceCacheMount(topology)
	if mount == nil || hf == "" {
		return nil
	}
	argv := []string{
		"env", "-u", "HF_HUB_OFFLINE", "-u", "TRANSFORMERS_OFFLINE",
		"-u", "HF_HUB_DISABLE_PROGRESS_BARS", "HF_HOME=" + mount.Source, hf,
	}
	return append(argv, arguments...)
}

func remoteHFCLI(host string, topology *SparkRDMATopology) string {
	candidates := []string{
		filepath.Join(filepath.Dir(topology.RuntimePython), "hf"),
		"hf",
	}
	for _, candidate := range candidates {
		argv := sparkHostHFArgv(topology, candidate, "--version")
		if argv == nil {
			return ""
		}
		_, status, err := remoteRun(host, argv, 30*time.Second)
		if err == nil && status == 0 {
			return candidate
		}
	}
	return ""
}

func sparkContainerHFArgv(topology *SparkRDMATopology, arguments ...string) []string {
	argv := []string{"docker", "run", "--rm", "--network", "host"}
	if mount := huggingFaceCacheMount(topology); mount != nil {
		argv = append(argv, "--mount", fmt.Sprintf(
			"type=bind,src=%s,dst=%s", mount.Source, mount.Target,
		))
	}
	argv = append(argv, "--entrypoint", "hf", topology.Image)
	return append(argv, arguments...)
}

func sparkContainerHFIsAvailable(host string, topology *SparkRDMATopology) bool {
	if huggingFaceCacheMount(topology) == nil {
		return false
	}
	argv := sparkContainerHFArgv(topology, "--version")
	_, status, err := remoteRun(host, argv, 2*time.Minute)
	return err == nil && status == 0
}

func syncSparkHuggingFaceCache(spec LaunchSpec) error {
	topology := spec.Topology.Spark
	for _, node := range spec.SparkNodes {
		hostHF := remoteHFCLI(node.Node.SSHHost, topology)
		useContainerHF := false
		if hostHF == "" {
			useContainerHF = sparkContainerHFIsAvailable(node.Node.SSHHost, topology)
		}
		for _, repository := range spec.DownloadRepositories {
			if hostHF == "" && !useContainerHF {
				warnUnmanagedDownload(repository, "on "+node.Node.SSHHost+" or in "+topology.Image)
				continue
			}
			fmt.Fprintf(
				os.Stderr, "updating Hugging Face cache for %s on %s\n",
				repository, node.Node.SSHHost,
			)
			argv := sparkHostHFArgv(topology, hostHF, "download", repository)
			if useContainerHF {
				argv = sparkContainerHFArgv(topology, "download", repository)
			}
			remote := RemoteArgv(node.Node.SSHHost, argv)
			if err := runVisible(remote, nil); err != nil {
				var exit *exec.ExitError
				if errors.As(err, &exit) && exit.ExitCode() == 127 {
					warnUnmanagedDownload(repository, "on "+node.Node.SSHHost)
					continue
				}
				return fmt.Errorf(
					"Hugging Face download failed for %s on %s: %w",
					repository, node.Node.SSHHost, err,
				)
			}
		}
	}
	return nil
}

// SyncHuggingFaceCache updates every unpinned Hub repository needed by a launch.
// An unavailable hf CLI is non-fatal because vLLM can still populate its cache.
func SyncHuggingFaceCache(spec LaunchSpec) error {
	if len(spec.DownloadRepositories) == 0 {
		return nil
	}
	if spec.Topology.Spark != nil {
		return syncSparkHuggingFaceCache(spec)
	}
	return syncLocalHuggingFaceCache(spec)
}
