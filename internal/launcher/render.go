// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package launcher

import (
	"encoding/json"
	"regexp"
	"strings"
)

var shellSafe = regexp.MustCompile(`^[A-Za-z0-9_@%+=:,./-]+$`)

func shellQuote(value string) string {
	if value == "" {
		return "''"
	}
	if shellSafe.MatchString(value) {
		return value
	}
	return "'" + strings.ReplaceAll(value, "'", `'"'"'`) + "'"
}

func shellJoin(argv []string) string {
	quoted := make([]string, len(argv))
	for index, argument := range argv {
		quoted[index] = shellQuote(argument)
	}
	return strings.Join(quoted, " ")
}

func PrettyShellArgv(argv []string) string {
	var groups [][]string
	index := 0
	for index < len(argv) {
		argument := argv[index]
		if len(groups) == 0 {
			groups = append(groups, []string{})
		}
		if !strings.HasPrefix(argument, "--") && len(groups) == 1 {
			groups[0] = append(groups[0], argument)
			index++
			continue
		}
		if !strings.HasPrefix(argument, "--") {
			var group []string
			for index < len(argv) && !strings.HasPrefix(argv[index], "--") {
				group = append(group, argv[index])
				index++
			}
			groups = append(groups, group)
			continue
		}
		group := []string{argument}
		if !strings.Contains(argument, "=") && index+1 < len(argv) && !strings.HasPrefix(argv[index+1], "--") {
			group = append(group, argv[index+1])
			index++
		}
		groups = append(groups, group)
		index++
	}
	lines := make([]string, 0, len(groups))
	for _, group := range groups {
		if len(group) > 0 {
			lines = append(lines, shellJoin(group))
		}
	}
	return strings.Join(lines, " \\\n  ")
}

func ShellRender(spec LaunchSpec) string {
	if spec.Topology.Spark != nil {
		commands := make([]string, 0, len(spec.SparkNodes))
		for index := len(spec.SparkNodes) - 1; index >= 1; index-- {
			node := spec.SparkNodes[index]
			remote := RemoteArgv(node.Node.SSHHost, node.DockerArgv)
			remote[len(remote)-1] = PrettyShellArgv(node.DockerArgv)
			commands = append(commands, PrettyShellArgv(remote))
		}
		head := spec.SparkNodes[0]
		remote := RemoteArgv(head.Node.SSHHost, head.DockerArgv)
		remote[len(remote)-1] = PrettyShellArgv(head.DockerArgv)
		commands = append(commands, PrettyShellArgv(remote))
		return strings.Join(commands, "\n\n")
	}
	lines := make([]string, 0, len(spec.HostEnvironment)+2)
	for _, name := range spec.UnsetEnvironment {
		lines = append(lines, "unset "+shellQuote(name))
	}
	for _, name := range sortedMapKeys(spec.HostEnvironment) {
		lines = append(lines, "export "+name+"="+shellQuote(spec.HostEnvironment[name]))
	}
	lines = append(lines, "exec "+PrettyShellArgv(spec.CommandArgv))
	return strings.Join(lines, "\n")
}

type sparkNodeJSON struct {
	SSHHost            string            `json:"ssh_host"`
	Address            string            `json:"address"`
	Rank               int               `json:"rank"`
	ContainerName      string            `json:"container_name"`
	RuntimeEnvironment map[string]string `json:"runtime_environment"`
	VLLMArgv           []string          `json:"vllm_argv"`
	DockerArgv         []string          `json:"docker_argv"`
}

type launchSpecJSON struct {
	Model                string            `json:"model"`
	Repository           string            `json:"repository"`
	ManifestCommit       string            `json:"manifest_commit"`
	Family               string            `json:"family"`
	ModelSource          string            `json:"model_source"`
	CheckpointPath       *string           `json:"checkpoint_path"`
	ServedModelName      string            `json:"served_model_name"`
	Topology             string            `json:"topology"`
	TopologyKind         string            `json:"topology_kind"`
	TensorParallelSize   int               `json:"tensor_parallel_size"`
	DeviceIDs            []int             `json:"device_ids"`
	Host                 string            `json:"host"`
	Port                 int               `json:"port"`
	Detach               bool              `json:"detach"`
	SyncCode             bool              `json:"sync_code"`
	SyncModel            bool              `json:"sync_model"`
	DownloadRepositories []string          `json:"download_repositories"`
	UnsetEnvironment     []string          `json:"unset_environment"`
	RuntimeEnvironment   map[string]string `json:"runtime_environment"`
	HostEnvironment      map[string]string `json:"host_environment"`
	VLLMArgv             []string          `json:"vllm_argv"`
	CommandArgv          []string          `json:"command_argv"`
	SparkNodes           []sparkNodeJSON   `json:"spark_nodes"`
	Metadata             map[string]any    `json:"metadata"`
}

func JSONRender(spec LaunchSpec) ([]byte, error) {
	nodes := make([]sparkNodeJSON, 0, len(spec.SparkNodes))
	for _, node := range spec.SparkNodes {
		nodes = append(nodes, sparkNodeJSON{
			SSHHost: node.Node.SSHHost, Address: node.Node.Address, Rank: node.Rank,
			ContainerName:      node.ContainerName,
			RuntimeEnvironment: node.RuntimeEnvironment,
			VLLMArgv:           node.VLLMArgv, DockerArgv: node.DockerArgv,
		})
	}
	deviceIDs := spec.DeviceIDs
	if deviceIDs == nil && spec.Topology.Local != nil {
		deviceIDs = []int{}
	}
	unset := spec.UnsetEnvironment
	if unset == nil {
		unset = []string{}
	}
	return json.MarshalIndent(launchSpecJSON{
		Model: spec.Model.Name, Repository: spec.Model.Model,
		ManifestCommit: spec.Model.ManifestCommit, Family: spec.Model.Family,
		ModelSource:    spec.ModelSource,
		CheckpointPath: spec.CheckpointPath, ServedModelName: spec.ServedModelName,
		Topology: spec.Topology.Name(), TopologyKind: spec.Topology.Kind,
		TensorParallelSize: spec.TPSize, DeviceIDs: deviceIDs,
		Host: spec.Host, Port: spec.Port, Detach: spec.Detach, SyncCode: spec.SyncCode,
		SyncModel: spec.SyncModel, DownloadRepositories: spec.DownloadRepositories,
		UnsetEnvironment: unset, RuntimeEnvironment: spec.RuntimeEnvironment,
		HostEnvironment: spec.HostEnvironment, VLLMArgv: spec.VLLMArgv,
		CommandArgv: spec.CommandArgv, SparkNodes: nodes, Metadata: spec.Metadata,
	}, "", "  ")
}
