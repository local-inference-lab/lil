// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/local-inference-lab/lil/internal/hfhub"
	"github.com/local-inference-lab/lil/internal/launcher"
	"github.com/spf13/pflag"
	"gopkg.in/yaml.v3"
)

var catalogNamePattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.-]*$`)

func localManifestPath(name string) (string, error) {
	if !catalogNamePattern.MatchString(name) {
		return "", fmt.Errorf("invalid catalog entry name %q", name)
	}
	root, err := os.UserConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(root, "lil", "models", name, "lil.yaml"), nil
}

func catalogManifest(ctx context.Context, client *hfhub.Client, repository hfhub.Repository, name string) ([]byte, string, string, error) {
	path, err := localManifestPath(name)
	if err != nil {
		return nil, "", "", err
	}
	data, err := os.ReadFile(path)
	if err == nil {
		return data, path, "", nil
	}
	if !os.IsNotExist(err) {
		return nil, "", "", fmt.Errorf("read local manifest %s: %w", path, err)
	}
	data, _, err = client.FetchFile(ctx, repository, name+"/lil.yaml", hfhub.ManifestLimit)
	return data, repository.ID + "/" + name + "/lil.yaml", repository.Revision, err
}

func editCommand(ctx context.Context, args []string) error {
	fs := pflag.NewFlagSet("edit", pflag.ContinueOnError)
	repository := fs.Bool("repository", false, "edit and upload the upstream catalog manifest")
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "Usage: lil edit NAME [--repository]")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		return fmt.Errorf("edit requires exactly one catalog NAME")
	}
	if owner, _, qualified := strings.Cut(fs.Arg(0), "/"); qualified && !catalogNamePattern.MatchString(owner) {
		return fmt.Errorf("invalid catalog entry name %q", fs.Arg(0))
	}
	client, err := hfhub.DefaultClient(modelRepositoryOwner)
	if err != nil {
		return err
	}
	return editManifest(ctx, client, launchName(fs.Arg(0)), *repository)
}

func validateEditedManifest(data []byte, name, path string) error {
	if len(data) >= hfhub.ManifestLimit {
		return fmt.Errorf("manifest must be smaller than %d bytes", hfhub.ManifestLimit)
	}
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	var document yaml.Node
	if err := decoder.Decode(&document); err != nil {
		return fmt.Errorf("invalid YAML in %s: %w", path, err)
	}
	if err := decoder.Decode(&document); err != io.EOF {
		return fmt.Errorf("%s must contain exactly one YAML document", path)
	}
	kind, err := launcher.ManifestKind(data, path)
	if err != nil {
		return err
	}
	if kind == "draft" {
		_, err = launcher.LoadDraftEntry(data, name, "", path)
		return err
	}
	families, err := embeddedModelFamilies()
	if err != nil {
		return err
	}
	_, err = launcher.LoadModelEntry(families, data, name, "", path, modelRepositoryOwner)
	return err
}

func editManifest(ctx context.Context, client *hfhub.Client, name string, upstream bool) error {
	path, err := localManifestPath(name)
	if err != nil {
		return err
	}
	editor := strings.TrimSpace(os.Getenv("EDITOR"))
	if editor == "" {
		return fmt.Errorf("EDITOR is not set; set it to an editor command (for example, vim or 'code --wait')")
	}
	if upstream && strings.TrimSpace(client.Token) == "" {
		return fmt.Errorf("--repository requires Hugging Face authentication; run hf auth login or set HF_TOKEN with write access to %s", client.CatalogID())
	}
	var original []byte
	var repository hfhub.Repository
	existed := false
	if !upstream {
		original, err = os.ReadFile(path)
		existed = err == nil
		if err != nil && !os.IsNotExist(err) {
			return err
		}
	}
	if !existed {
		repository, err = resolveCatalogRepository(ctx, client)
		if err != nil {
			return err
		}
		original, _, err = client.FetchFile(ctx, repository, name+"/lil.yaml", hfhub.ManifestLimit)
		if err != nil {
			return err
		}
	}
	directory := ""
	if !upstream {
		directory = filepath.Dir(path)
		if err := os.MkdirAll(directory, 0o755); err != nil {
			return err
		}
	}
	temporary, err := os.CreateTemp(directory, "lil-edit-*.yaml")
	if err != nil {
		return err
	}
	temporaryPath := temporary.Name()
	if _, err := temporary.Write(original); err != nil {
		temporary.Close()
		os.Remove(temporaryPath)
		return err
	}
	if err := temporary.Close(); err != nil {
		os.Remove(temporaryPath)
		return err
	}
	preserve := func(err error) error { return fmt.Errorf("%w; edited file retained at %s", err, temporaryPath) }
	fmt.Fprintf(os.Stderr, "Editing %s/%s/lil.yaml", client.CatalogID(), name)
	if !upstream {
		fmt.Fprintf(os.Stderr, " (local override: %s)", path)
	}
	fmt.Fprintln(os.Stderr)
	// The editor command may include quoted arguments; pass the filename separately.
	command := exec.CommandContext(ctx, "sh", "-c", editor+` "$@"`, "lil-editor", temporaryPath)
	command.Stdin, command.Stdout, command.Stderr = os.Stdin, os.Stdout, os.Stderr
	if err := command.Run(); err != nil {
		return preserve(fmt.Errorf("editor failed: %w", err))
	}
	edited, err := os.ReadFile(temporaryPath)
	if err != nil {
		return preserve(err)
	}
	if bytes.Equal(original, edited) {
		os.Remove(temporaryPath)
		fmt.Fprintln(os.Stderr, "No changes.")
		return nil
	}
	if err := validateEditedManifest(edited, name, temporaryPath); err != nil {
		return preserve(err)
	}
	if upstream {
		commit, err := client.UploadManifest(ctx, repository, name+"/lil.yaml", edited)
		if err != nil {
			return preserve(err)
		}
		os.Remove(temporaryPath)
		fmt.Fprintf(os.Stderr, "Uploaded %s/%s/lil.yaml at %s\n", repository.ID, name, commit)
		if _, err := os.Stat(path); err == nil {
			fmt.Fprintf(os.Stderr, "Local override still shadows upstream: %s\n", path)
		}
		return nil
	}
	current, readErr := os.ReadFile(path)
	if existed && (readErr != nil || !bytes.Equal(current, original)) || !existed && !os.IsNotExist(readErr) {
		return preserve(fmt.Errorf("local manifest changed while editing; refusing to overwrite %s", path))
	}
	if err := os.Rename(temporaryPath, path); err != nil {
		return preserve(err)
	}
	fmt.Fprintf(os.Stderr, "Saved local override: %s\n", path)
	return nil
}
