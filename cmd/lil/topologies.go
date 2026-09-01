// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/local-inference-lab/lil/internal/launcher"
)

type topologyFile struct {
	Path     string
	Topology launcher.Topology
	Err      error
}

func topologyConfigDir() (string, error) {
	if directory := os.Getenv("LIL_TOPOLOGY_DIR"); directory != "" {
		return filepath.Abs(directory)
	}
	root, err := os.UserConfigDir()
	if err != nil {
		return "", fmt.Errorf("resolve user configuration directory: %w", err)
	}
	return filepath.Join(root, "lil", "topologies"), nil
}

func readTopologyFile(path string) (launcher.Topology, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return launcher.Topology{}, err
	}
	data, err := os.ReadFile(abs)
	if err != nil {
		return launcher.Topology{}, fmt.Errorf("cannot read topology %s: %w", abs, err)
	}
	return launcher.LoadTopology(data, abs, filepath.Dir(abs))
}

func discoveredTopologies() ([]topologyFile, error) {
	directory, err := topologyConfigDir()
	if err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(directory)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read topology directory %s: %w", directory, err)
	}
	result := []topologyFile{}
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".yaml" {
			continue
		}
		path := filepath.Join(directory, entry.Name())
		topology, loadErr := readTopologyFile(path)
		result = append(result, topologyFile{Path: path, Topology: topology, Err: loadErr})
	}
	sort.Slice(result, func(i, j int) bool { return result[i].Path < result[j].Path })
	return result, nil
}

func uniqueDiscoveredTopology(
	topologies []topologyFile,
	predicate func(topologyFile) bool,
	description string,
) (launcher.Topology, error) {
	matches := []topologyFile{}
	for _, item := range topologies {
		if item.Err == nil && predicate(item) {
			matches = append(matches, item)
		}
	}
	if len(matches) == 1 {
		return matches[0].Topology, nil
	}
	if len(matches) == 0 {
		return launcher.Topology{}, fmt.Errorf(
			"no %s topology is configured; run lil discover", description,
		)
	}
	names := make([]string, len(matches))
	for index, item := range matches {
		names[index] = item.Topology.Name()
	}
	return launcher.Topology{}, fmt.Errorf(
		"%s topology is ambiguous (%s); select one with --config",
		description, strings.Join(names, ", "),
	)
}

func loadTopology(selection string) (launcher.Topology, error) {
	if selection == "" {
		selection = os.Getenv("LIL_TOPOLOGY")
	}
	if selection != "" && (strings.ContainsRune(selection, filepath.Separator) ||
		filepath.IsAbs(selection)) {
		return readTopologyFile(selection)
	}
	if selection != "" {
		if info, err := os.Stat(selection); err == nil && !info.IsDir() {
			return readTopologyFile(selection)
		}
	}
	topologies, err := discoveredTopologies()
	if err != nil {
		return launcher.Topology{}, err
	}
	if selection == "local" || selection == "spark" || selection == "spark_rdma" {
		kind := "local"
		if selection != "local" {
			kind = "spark_rdma"
		}
		return uniqueDiscoveredTopology(
			topologies,
			func(item topologyFile) bool { return item.Topology.Kind == kind },
			selection,
		)
	}
	if selection != "" {
		trimmed := strings.TrimSuffix(selection, ".yaml")
		for _, item := range topologies {
			base := strings.TrimSuffix(filepath.Base(item.Path), ".yaml")
			if base == trimmed || item.Err == nil && item.Topology.Name() == selection {
				if item.Err != nil {
					return launcher.Topology{}, item.Err
				}
				return item.Topology, nil
			}
		}
		return launcher.Topology{}, fmt.Errorf(
			"unknown discovered topology %q; run lil list or lil discover", selection,
		)
	}
	hostname, hostnameErr := os.Hostname()
	if hostnameErr == nil {
		for _, item := range topologies {
			if item.Err == nil && item.Topology.Kind == "local" &&
				(item.Topology.Name() == hostname ||
					strings.TrimSuffix(filepath.Base(item.Path), ".yaml") == hostname) {
				return item.Topology, nil
			}
		}
	}
	return uniqueDiscoveredTopology(
		topologies,
		func(item topologyFile) bool { return item.Topology.Kind == "local" },
		"local",
	)
}
