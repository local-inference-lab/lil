// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package hfhub

import (
	"context"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"syscall"
)

type ImportOptions struct {
	Revision string
	CacheDir string
	DryRun   bool
	Progress io.Writer
}

type ImportResult struct {
	Revision     string
	SnapshotPath string
	LinkedFiles  int
	LinkedBytes  int64
	CachedFiles  int
	MissingFiles []string
}

// HubCacheDir follows huggingface_hub's cache environment precedence.
func HubCacheDir() (string, error) {
	for _, name := range []string{"HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"} {
		if value := os.Getenv(name); value != "" {
			return value, nil
		}
	}
	home := os.Getenv("HF_HOME")
	if home == "" {
		cache := os.Getenv("XDG_CACHE_HOME")
		if cache == "" {
			userHome, err := os.UserHomeDir()
			if err != nil {
				return "", err
			}
			cache = filepath.Join(userHome, ".cache")
		}
		home = filepath.Join(cache, "huggingface")
	}
	return filepath.Join(home, "hub"), nil
}

type importFile struct {
	sibling Sibling
	source  string
	info    os.FileInfo
	blob    string
	pointer string
}

func (sibling Sibling) digest() (string, error) {
	digest, length := sibling.BlobID, 40
	if sibling.LFS != nil {
		digest, length = sibling.LFS.SHA256, 64
		if sibling.LFS.Size != sibling.Size {
			return "", fmt.Errorf("inconsistent LFS size for %s", sibling.Filename)
		}
	}
	decoded, err := hex.DecodeString(digest)
	if err != nil || len(decoded)*2 != length || strings.ToLower(digest) != digest || sibling.Size < 0 {
		return "", fmt.Errorf("missing or invalid content hash/size for %s", sibling.Filename)
	}
	return digest, nil
}

func unchanged(path string, before os.FileInfo) error {
	after, err := os.Stat(path)
	if err != nil {
		return err
	}
	if !os.SameFile(before, after) || before.Size() != after.Size() || !before.ModTime().Equal(after.ModTime()) {
		return fmt.Errorf("file changed during import: %s", path)
	}
	return nil
}

func verifyImportFile(ctx context.Context, path string, sibling Sibling) (os.FileInfo, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() || info.Size() != sibling.Size {
		return nil, fmt.Errorf("%s: expected a regular file of %d bytes, got %s (%d bytes)", path, sibling.Size, info.Mode(), info.Size())
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	info, err = file.Stat()
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() || info.Size() != sibling.Size {
		return nil, fmt.Errorf("%s: expected a regular file of %d bytes, got %s (%d bytes)", path, sibling.Size, info.Mode(), info.Size())
	}
	digest, err := sibling.digest()
	if err != nil {
		return nil, err
	}
	var h hash.Hash = sha256.New()
	if sibling.LFS == nil {
		h = sha1.New()
		fmt.Fprintf(h, "blob %d\x00", sibling.Size)
	}
	buffer := make([]byte, 1<<20)
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		n, err := file.Read(buffer)
		h.Write(buffer[:n])
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
	}
	if hex.EncodeToString(h.Sum(nil)) != digest {
		return nil, fmt.Errorf("content hash mismatch for %s (expected %s)", path, digest)
	}
	return info, unchanged(path, info)
}

// Cache roots may contain user-configured symlinks; directories inside them must
// be real directories so importing cannot write through a snapshot symlink.
func checkCacheParents(root, path string) error {
	relative, err := filepath.Rel(root, filepath.Dir(path))
	if err != nil || !filepath.IsLocal(relative) {
		return fmt.Errorf("cache path escapes %s: %s", root, path)
	}
	parent := root
	for _, part := range strings.Split(relative, string(filepath.Separator)) {
		parent = filepath.Join(parent, part)
		info, err := os.Lstat(parent)
		if os.IsNotExist(err) {
			return nil
		}
		if err != nil {
			return err
		}
		if !info.IsDir() {
			return fmt.Errorf("cache parent must be a directory, not a symlink or file: %s", parent)
		}
	}
	return nil
}

func existingAncestor(path string) (os.FileInfo, error) {
	for {
		info, err := os.Stat(path)
		if !os.IsNotExist(err) || filepath.Dir(path) == path {
			return info, err
		}
		path = filepath.Dir(path)
	}
}

// Import verifies local repository files and registers them in the standard Hub
// cache using hardlinks and snapshot symlinks. It only requests Hub metadata.
// Files absent from both the source and snapshot are reported, never downloaded.
func (client *Client) Import(ctx context.Context, id, source string, options ImportOptions) (ImportResult, error) {
	var result ImportResult
	source, err := filepath.EvalSymlinks(source)
	if err != nil {
		return result, err
	}
	source, err = filepath.Abs(source)
	if err != nil {
		return result, err
	}
	info, err := os.Stat(source)
	if err != nil {
		return result, err
	}
	if !info.IsDir() {
		return result, fmt.Errorf("source must be a directory: %s", source)
	}
	cache := options.CacheDir
	if cache == "" {
		cache, err = HubCacheDir()
		if err != nil {
			return result, err
		}
	}
	cache, err = filepath.Abs(cache)
	if err != nil {
		return result, err
	}
	cacheInfo, err := existingAncestor(cache)
	if err != nil {
		return result, err
	}
	repository, err := client.ResolveAt(ctx, id, options.Revision)
	if err != nil {
		return result, err
	}
	result.Revision = repository.Revision
	repoName := "models--" + strings.ReplaceAll(id, "/", "--")
	repoDir := filepath.Join(cache, repoName)
	result.SnapshotPath = filepath.Join(repoDir, "snapshots", repository.Revision)
	ref := filepath.Join(repoDir, "refs", "main")
	if err := checkCacheParents(cache, ref); err != nil {
		return result, err
	}
	if options.Progress != nil {
		fmt.Fprintf(options.Progress, "Verifying %s@%s against %s\n", id, repository.Revision, source)
	}
	files := make([]importFile, 0, len(repository.Siblings))
	seen := map[string]bool{}
	localFiles := 0
	for _, sibling := range repository.Siblings {
		name := sibling.Filename
		if !fs.ValidPath(name) || name == "." || strings.Contains(name, "\\") || seen[name] {
			return result, fmt.Errorf("invalid or duplicate repository path %q", name)
		}
		seen[name] = true
		digest, err := sibling.digest()
		if err != nil {
			return result, err
		}
		entry := importFile{
			sibling: sibling, source: filepath.Join(source, name),
			blob: filepath.Join(repoDir, "blobs", digest), pointer: filepath.Join(result.SnapshotPath, name),
		}
		for _, path := range []string{entry.blob, entry.pointer, filepath.Join(cache, ".locks", repoName, digest+".lock")} {
			if err := checkCacheParents(cache, path); err != nil {
				return result, err
			}
		}
		if options.Progress != nil {
			fmt.Fprintf(options.Progress, "[%d/%d] %s (%d bytes)\n", len(seen), len(repository.Siblings), name, sibling.Size)
		}
		if _, err := os.Lstat(entry.source); err == nil {
			entry.source, err = filepath.EvalSymlinks(entry.source)
			if err != nil {
				return result, err
			}
			entry.info, err = verifyImportFile(ctx, entry.source, sibling)
			if err != nil {
				return result, err
			}
			localFiles++
		} else if !os.IsNotExist(err) {
			return result, err
		}
		pointerExists, blobExists := false, false
		for _, path := range []string{entry.blob, entry.pointer} {
			stat, err := os.Lstat(path)
			if os.IsNotExist(err) {
				continue
			}
			if err != nil {
				return result, err
			}
			if path == entry.blob && !stat.Mode().IsRegular() {
				return result, fmt.Errorf("cache blob must be a regular file: %s", path)
			}
			stat, err = os.Stat(path)
			if err != nil {
				return result, fmt.Errorf("cache conflict: %w", err)
			}
			if entry.info == nil || !os.SameFile(entry.info, stat) {
				if _, err := verifyImportFile(ctx, path, sibling); err != nil {
					return result, fmt.Errorf("cache conflict: %w", err)
				}
			} else if err := unchanged(path, entry.info); err != nil {
				return result, err
			}
			if path == entry.pointer {
				pointerExists = true
			} else {
				blobExists = true
			}
		}
		if pointerExists {
			result.CachedFiles++
			continue
		}
		if blobExists {
			entry.source = entry.blob
			entry.info, err = os.Stat(entry.blob)
			if err != nil {
				return result, err
			}
			result.CachedFiles++
		} else if entry.info == nil {
			result.MissingFiles = append(result.MissingFiles, name)
			continue
		} else {
			if entry.info.Sys().(*syscall.Stat_t).Dev != cacheInfo.Sys().(*syscall.Stat_t).Dev {
				return result, fmt.Errorf("cannot hardlink %s into %s: different filesystems; choose --cache-dir on the source filesystem", entry.source, cache)
			}
			result.LinkedFiles++
			result.LinkedBytes += sibling.Size
		}
		files = append(files, entry)
	}
	if localFiles == 0 {
		return result, fmt.Errorf("no repository files found in %s; pass the directory containing the model files", source)
	}
	if err := ctx.Err(); err != nil {
		return result, err
	}
	if options.DryRun {
		return result, nil
	}
	for _, entry := range files {
		if err := ctx.Err(); err != nil {
			return result, err
		}
		if err := importCacheFile(ctx, cache, repoName, entry); err != nil {
			return result, err
		}
	}
	if options.Revision == "" {
		if err := ctx.Err(); err != nil {
			return result, err
		}
		if err := checkCacheParents(cache, ref); err != nil {
			return result, err
		}
		if err := atomicWrite(ref, []byte(repository.Revision)); err != nil {
			return result, err
		}
	}
	return result, nil
}

func importCacheFile(ctx context.Context, cache, repoName string, entry importFile) error {
	lockPath := filepath.Join(cache, ".locks", repoName, filepath.Base(entry.blob)+".lock")
	for _, path := range []string{lockPath, entry.blob, entry.pointer} {
		if err := checkCacheParents(cache, path); err != nil {
			return err
		}
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			return err
		}
	}
	lock, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR|syscall.O_NOFOLLOW, 0o644)
	if err != nil {
		return err
	}
	defer lock.Close()
	if err := syscall.Flock(int(lock.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		return fmt.Errorf("cache blob is locked by another process (%s); retry after it finishes: %w", entry.sibling.Filename, err)
	}
	defer syscall.Flock(int(lock.Fd()), syscall.LOCK_UN)
	if err := unchanged(entry.source, entry.info); err != nil {
		return err
	}
	if entry.source != entry.blob {
		if err := os.Link(entry.source, entry.blob); err != nil {
			if !errors.Is(err, fs.ErrExist) {
				return fmt.Errorf("cannot hardlink %s: %w", entry.sibling.Filename, err)
			}
			if _, err := verifyImportFile(ctx, entry.blob, entry.sibling); err != nil {
				return fmt.Errorf("cache conflict: %w", err)
			}
		}
	}
	if err := unchanged(entry.blob, entry.info); err != nil {
		// An existing verified blob can have a different inode from the source.
		if _, err := verifyImportFile(ctx, entry.blob, entry.sibling); err != nil {
			return err
		}
	}
	target, err := filepath.Rel(filepath.Dir(entry.pointer), entry.blob)
	if err != nil {
		return err
	}
	if err := os.Symlink(target, entry.pointer); err != nil {
		if !errors.Is(err, fs.ErrExist) {
			return err
		}
		if _, err := verifyImportFile(ctx, entry.pointer, entry.sibling); err != nil {
			return fmt.Errorf("cache conflict: %w", err)
		}
	}
	return nil
}
