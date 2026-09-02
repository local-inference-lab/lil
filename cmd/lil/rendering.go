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

// topologyFit describes how a model lands on one discovered topology.
func topologyFit(profile launcher.ModelProfile, item topologyFile) string {
	if item.Err != nil || profile.Facts == nil {
		return ""
	}
	topology := item.Topology
	if len(profile.Requires.Arch) > 0 && !contains(profile.Requires.Arch, topology.CuteDSLArch()) {
		return fmt.Sprintf("%s: requires %s", topology.Name(), strings.Join(profile.Requires.Arch, "/"))
	}
	tp, err := launcher.DefaultTPSize(profile.Facts, topology, topology.MemoryUtilization())
	if err != nil {
		return fmt.Sprintf("%s: does not fit", topology.Name())
	}
	return fmt.Sprintf("%s: TP %d", topology.Name(), tp)
}

func contains(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func profileSummary(profile launcher.ModelProfile) string {
	parts := []string{}
	if profile.Facts != nil {
		parts = append(parts, formatGiB(profile.Facts.WeightBytes)+" stored")
	}
	if profile.Family != "" {
		parts = append(parts, "family "+profile.Family)
	}
	parts = append(parts, "speculator "+profile.Speculators.Default)
	return strings.Join(parts, "  ·  ")
}

func renderList(profiles map[string]launcher.ModelProfile) string {
	topologies, topologyErr := discoveredTopologies()
	var sections []string
	names := launcher.SortedProfileNames(profiles)
	modelCards := make([]string, 0, len(names))
	for _, name := range names {
		profile := profiles[name]
		lines := []string{
			nameStyle.Render(name),
			wrapListText(profile.Description),
			metadataStyle.Render(wrapListText(profileSummary(profile))),
		}
		fits := []string{}
		for _, item := range topologies {
			if fit := topologyFit(profile, item); fit != "" {
				fits = append(fits, fit)
			}
		}
		if len(fits) > 0 {
			lines = append(lines, metadataStyle.Render(wrapListText("defaults  ·  "+strings.Join(fits, "  ·  "))))
		}
		modelCards = append(modelCards, cardStyle.Render(strings.Join(lines, "\n")))
	}
	sections = append(sections,
		headingStyle.Render(fmt.Sprintf("Models  %d", len(names)))+"\n\n"+
			strings.Join(modelCards, "\n\n"),
	)

	if topologyErr != nil {
		sections = append(sections,
			headingStyle.Render("Topologies")+"\n\n"+
				cardStyle.Render(metadataStyle.Render("unavailable: ")+topologyErr.Error()),
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
		body := strings.Join([]string{
			nameStyle.Render(item.Topology.Name()),
			metadataStyle.Render(wrapListText(topologySummary(item.Topology))),
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

func topologySummary(topology launcher.Topology) string {
	kind := "local"
	unit := "GPU"
	if topology.Spark != nil {
		kind = "Spark/RDMA"
		unit = "rank"
	}
	return fmt.Sprintf(
		"%s  ·  TP 1–%d  ·  %s/%s  ·  %s  ·  default TP %s",
		kind, topology.MaxTPSize(), formatGiB(topology.DeviceMemoryBytes()), unit,
		topology.CuteDSLArch(), topology.DefaultTP,
	)
}

func printList(profiles map[string]launcher.ModelProfile) error {
	_, err := lipgloss.Fprint(os.Stdout, renderList(profiles))
	return err
}

func printDiscoveryResult(topology launcher.Topology, path string) {
	body := strings.Join([]string{
		nameStyle.Render(topology.Name()),
		metadataStyle.Render(topologySummary(topology)),
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
