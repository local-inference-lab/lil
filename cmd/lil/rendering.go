// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"

	"github.com/local-inference-lab/lil/internal/launcher"
)

const listContentWidth = 76

func wrapListText(value string) string {
	return ansi.Wordwrap(value, listContentWidth, " ")
}

func wrapListPath(value string) string {
	return ansi.Hardwrap(value, listContentWidth, false)
}

var (
	headingStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#A78BFA"))
	nameStyle = lipgloss.NewStyle().
			Bold(true).
			Foreground(lipgloss.Color("#67E8F9"))
	metadataStyle = lipgloss.NewStyle().
			Foreground(lipgloss.Color("#94A3B8"))
	cardStyle = lipgloss.NewStyle().
			Border(lipgloss.ThickBorder(), false, false, false, true).
			BorderForeground(lipgloss.Color("#7C3AED")).
			PaddingLeft(1).
			MarginLeft(2)
)

func displayPath(path string) string {
	home, err := os.UserHomeDir()
	if err == nil && (path == home || strings.HasPrefix(path, home+string(filepath.Separator))) {
		return "~" + strings.TrimPrefix(path, home)
	}
	return path
}

func formatGiB(bytes int64) string {
	return fmt.Sprintf("%.1f GiB", float64(bytes)/float64(int64(1)<<30))
}

func renderList(profiles map[string]launcher.ModelProfile) string {
	var sections []string
	names := launcher.SortedProfileNames(profiles)
	modelCards := make([]string, 0, len(names))
	for _, name := range names {
		profile := profiles[name]
		metadata := fmt.Sprintf(
			"defaults  ·  local TP %s  ·  Spark/RDMA TP %s",
			defaultTP(profile.Launch.Local),
			defaultTP(profile.Launch.SparkRDMA),
		)
		body := strings.Join([]string{
			nameStyle.Render(name),
			wrapListText(profile.Description),
			metadataStyle.Render(wrapListText(metadata)),
		}, "\n")
		modelCards = append(modelCards, cardStyle.Render(body))
	}
	sections = append(sections,
		headingStyle.Render(fmt.Sprintf("Models  %d", len(names)))+"\n\n"+
			strings.Join(modelCards, "\n\n"),
	)

	topologies, err := discoveredTopologies()
	if err != nil {
		sections = append(sections,
			headingStyle.Render("Topologies")+"\n\n"+
				cardStyle.Render(metadataStyle.Render("unavailable: ")+err.Error()),
		)
		return strings.Join(sections, "\n\n") + "\n"
	}
	topologyCards := make([]string, 0, len(topologies))
	for _, item := range topologies {
		if item.Err != nil {
			body := strings.Join([]string{
				nameStyle.Render(strings.TrimSuffix(filepath.Base(item.Path), ".yaml")),
				metadataStyle.Render("invalid") + "  ·  " + wrapListText(item.Err.Error()),
				wrapListPath(displayPath(item.Path)),
			}, "\n")
			topologyCards = append(topologyCards, cardStyle.Render(body))
			continue
		}
		topology := item.Topology
		var summary string
		if topology.Local != nil {
			devices := 0
			for _, pool := range topology.Local.DevicePools {
				devices = max(devices, len(pool))
			}
			summary = fmt.Sprintf(
				"local  ·  TP 1–%d  ·  %s/GPU  ·  %s",
				devices, formatGiB(topology.DeviceMemoryBytes()), topology.CuteDSLArch(),
			)
		} else {
			summary = fmt.Sprintf(
				"Spark/RDMA  ·  TP 1–%d  ·  %s/rank  ·  %s",
				len(topology.Spark.Nodes), formatGiB(topology.DeviceMemoryBytes()),
				topology.CuteDSLArch(),
			)
		}
		body := strings.Join([]string{
			nameStyle.Render(topology.Name()),
			metadataStyle.Render(wrapListText(summary)),
			wrapListPath(displayPath(item.Path)),
		}, "\n")
		topologyCards = append(topologyCards, cardStyle.Render(body))
	}
	topologyContent := ""
	if len(topologyCards) == 0 {
		topologyContent = cardStyle.Render(strings.Join([]string{
			metadataStyle.Render("No discovered topologies"),
			"Run lil discover local or lil discover spark --node HOST ...",
		}, "\n"))
	} else {
		topologyContent = strings.Join(topologyCards, "\n\n")
	}
	sections = append(sections,
		headingStyle.Render(fmt.Sprintf("Topologies  %d", len(topologies)))+"\n\n"+
			topologyContent,
	)
	return strings.Join(sections, "\n\n") + "\n"
}

func printList(profiles map[string]launcher.ModelProfile) error {
	_, err := lipgloss.Fprint(os.Stdout, renderList(profiles))
	return err
}

func printDiscoveryResult(topology launcher.Topology, path string) {
	var capacity string
	if topology.Local != nil {
		devices := 0
		for _, pool := range topology.Local.DevicePools {
			devices = max(devices, len(pool))
		}
		capacity = fmt.Sprintf(
			"local  ·  TP 1–%d  ·  %s/GPU  ·  %s",
			devices, formatGiB(topology.DeviceMemoryBytes()), topology.CuteDSLArch(),
		)
	} else {
		capacity = fmt.Sprintf(
			"Spark/RDMA  ·  TP 1–%d  ·  %s/rank  ·  %s",
			len(topology.Spark.Nodes), formatGiB(topology.DeviceMemoryBytes()),
			topology.CuteDSLArch(),
		)
	}
	body := strings.Join([]string{
		nameStyle.Render(topology.Name()),
		metadataStyle.Render(capacity),
		displayPath(path),
	}, "\n")
	_, _ = lipgloss.Fprint(
		os.Stdout,
		headingStyle.Render("Topology discovered")+"\n\n"+cardStyle.Render(body)+"\n",
	)
}

func printClusterStatus(target launcher.SparkTarget, statuses []launcher.SparkNodeStatus) {
	cards := make([]string, 0, len(statuses))
	for _, status := range statuses {
		state := status.Status
		if status.Running {
			state = "running"
		} else if status.Exists {
			state = fmt.Sprintf("%s  ·  exit %d", status.Status, status.ExitCode)
		}
		body := strings.Join([]string{
			nameStyle.Render(fmt.Sprintf("rank %d  %s", status.Rank, status.Node.SSHHost)),
			metadataStyle.Render(status.Node.Address + "  ·  " + state),
		}, "\n")
		cards = append(cards, cardStyle.Render(body))
	}
	heading := fmt.Sprintf("%s  ·  TP %d  ·  %s", target.ProfileName, target.TPSize, target.ContainerName)
	_, _ = lipgloss.Fprint(
		os.Stdout,
		headingStyle.Render(heading)+"\n\n"+strings.Join(cards, "\n\n")+"\n",
	)
}
