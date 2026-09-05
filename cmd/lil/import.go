// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package main

import (
	"context"
	"fmt"
	"os"

	"github.com/local-inference-lab/lil/internal/hfhub"
)

func importCommand(ctx context.Context, args []string) error {
	fs := newFlagSet("import")
	var options hfhub.ImportOptions
	fs.StringVar(&options.Revision, "revision", "", "40-character Hub commit SHA (default: main)")
	fs.StringVar(&options.CacheDir, "cache-dir", "", "Hub cache directory (default: HF_HUB_CACHE, HF_HOME/hub, or ~/.cache/huggingface/hub)")
	fs.BoolVar(&options.DryRun, "dry-run", false, "verify files and report the import without writing to the cache")
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, `Usage: lil import OWNER/REPOSITORY LOCAL_DIR [options]

Import matching local files into the Hugging Face cache without downloading
file contents or copying weights. LOCAL_DIR must contain the repository files.
Hardlinks require the source and cache to share a filesystem. Treat linked
files as immutable: edits through either path affect both.

Files missing locally and in the cache are reported and left for hf download.`)
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 2 {
		return fmt.Errorf("import requires OWNER/REPOSITORY and LOCAL_DIR")
	}
	client, err := hfhub.DefaultClient(modelRepositoryOwner)
	if err != nil {
		return err
	}
	options.Progress = os.Stderr
	result, err := client.Import(ctx, fs.Arg(0), fs.Arg(1), options)
	if err != nil {
		return err
	}
	action := "Imported"
	if options.DryRun {
		action = "Would import"
	}
	fmt.Fprintf(os.Stderr, "%s %d files (%d bytes) using hardlinks; %d files already cached.\n", action, result.LinkedFiles, result.LinkedBytes, result.CachedFiles)
	for _, name := range result.MissingFiles {
		fmt.Fprintf(os.Stderr, "Missing locally and in snapshot: %s\n", name)
	}
	if len(result.MissingFiles) > 0 {
		fmt.Fprintf(os.Stderr, "Snapshot is incomplete: %d repository files missing.\n", len(result.MissingFiles))
	}
	fmt.Fprintln(os.Stdout, result.SnapshotPath)
	return nil
}
