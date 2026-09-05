// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package hfhub

import (
	"context"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync/atomic"
	"syscall"
	"testing"
)

func importFixture(t *testing.T) (*Client, string, string, Repository) {
	t.Helper()
	source, cache := t.TempDir(), filepath.Join(t.TempDir(), "hub")
	contents := map[string]string{
		".gitattributes": "*.safetensors filter=lfs\n",
		"config.json":    "{}\n", "nested/tokenizer/vocab.json": "{\"a\": 1}\n",
		"model.safetensors": "test weights\n",
	}
	repository := Repository{ID: "lab/Model", Revision: testRevision}
	for _, name := range []string{".gitattributes", "config.json", "nested/tokenizer/vocab.json", "model.safetensors"} {
		data := contents[name]
		file := Sibling{Filename: name, Size: int64(len(data)), BlobID: fmt.Sprintf("%x", sha1.Sum([]byte(fmt.Sprintf("blob %d\x00%s", len(data), data))))}
		if strings.HasSuffix(name, ".safetensors") {
			metadata := fmt.Sprintf(`{"sha256":"%x","size":%d}`, sha256.Sum256([]byte(data)), len(data))
			if err := json.Unmarshal([]byte(metadata), &file.LFS); err != nil {
				t.Fatal(err)
			}
		}
		repository.Siblings = append(repository.Siblings, file)
		writeImportFile(t, filepath.Join(source, name), data)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if (r.URL.Path != "/api/models/lab/Model" && r.URL.Path != "/api/models/lab/Model/revision/"+testRevision) || r.URL.Query().Get("blobs") != "true" {
			t.Errorf("unexpected request (import must fetch only metadata): %s", r.URL)
			http.NotFound(w, r)
			return
		}
		json.NewEncoder(w).Encode(repository)
	}))
	t.Cleanup(server.Close)
	return &Client{Endpoint: server.URL, HTTPClient: server.Client()}, source, cache, repository
}

func writeImportFile(t *testing.T, path, data string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestImportVerifiesAndHardlinksHubSnapshotWithoutDownloading(t *testing.T) {
	client, source, cache, repository := importFixture(t)
	options := ImportOptions{CacheDir: cache, DryRun: true}
	result, err := client.Import(context.Background(), repository.ID, source, options)
	if err != nil || result.LinkedFiles != 4 || len(result.MissingFiles) != 0 {
		t.Fatalf("dry run: %+v %v", result, err)
	}
	if _, err := os.Stat(cache); !os.IsNotExist(err) {
		t.Fatalf("dry run wrote the cache: %v", err)
	}
	options.DryRun = false
	result, err = client.Import(context.Background(), repository.ID, source, options)
	if err != nil {
		t.Fatal(err)
	}
	for _, file := range repository.Siblings {
		pointer := filepath.Join(result.SnapshotPath, file.Filename)
		target, err := os.Readlink(pointer)
		if err != nil || filepath.IsAbs(target) {
			t.Fatalf("snapshot entry is not a relative symlink: %s %v", target, err)
		}
		digest, _ := file.digest()
		blob := filepath.Join(cache, "models--lab--Model", "blobs", digest)
		if filepath.Clean(filepath.Join(filepath.Dir(pointer), target)) != blob {
			t.Fatalf("snapshot target: %s", target)
		}
		original, _ := os.Stat(filepath.Join(source, file.Filename))
		linked, err := os.Stat(blob)
		if err != nil || !os.SameFile(original, linked) {
			t.Fatalf("file was copied instead of hardlinked: %s %v", file.Filename, err)
		}
	}
	ref := filepath.Join(cache, "models--lab--Model", "refs", "main")
	if data, err := os.ReadFile(ref); err != nil || string(data) != testRevision {
		t.Fatalf("main ref: %q %v", data, err)
	}
	result, err = client.Import(context.Background(), repository.ID, source, options)
	if err != nil || result.LinkedFiles != 0 || result.CachedFiles != 4 {
		t.Fatalf("repeat import: %+v %v", result, err)
	}
	writeImportFile(t, ref, secondTestRevision)
	options.Revision = testRevision
	if _, err := client.Import(context.Background(), repository.ID, source, options); err != nil {
		t.Fatal(err)
	}
	if data, _ := os.ReadFile(ref); string(data) != secondTestRevision {
		t.Fatalf("pinned import rewrote main: %s", data)
	}
	if err := os.Remove(filepath.Join(source, "model.safetensors")); err != nil {
		t.Fatal(err)
	}
	if data, err := os.ReadFile(filepath.Join(result.SnapshotPath, "model.safetensors")); err != nil || string(data) != "test weights\n" {
		t.Fatalf("deleting source broke snapshot: %q %v", data, err)
	}
}

func TestImportRejectsMismatchBeforeWritingAnyCacheEntries(t *testing.T) {
	for _, data := range []string{"wrong weight\n", "truncated"} {
		t.Run(data, func(t *testing.T) {
			client, source, cache, repository := importFixture(t)
			writeImportFile(t, filepath.Join(source, "model.safetensors"), data)
			_, err := client.Import(context.Background(), repository.ID, source, ImportOptions{CacheDir: cache})
			if err == nil || !strings.Contains(err.Error(), "model.safetensors") {
				t.Fatalf("mismatch accepted: %v", err)
			}
			if _, err := os.Stat(cache); !os.IsNotExist(err) {
				t.Fatalf("cache changed before validation completed: %v", err)
			}
		})
	}
}

func TestImportPreservesConflictingCacheAndIncompleteDownloads(t *testing.T) {
	for _, conflict := range []string{"blob", "snapshot", "parent symlink"} {
		t.Run(conflict, func(t *testing.T) {
			client, source, cache, repository := importFixture(t)
			repo := filepath.Join(cache, "models--lab--Model")
			file := repository.Siblings[len(repository.Siblings)-1]
			digest, _ := file.digest()
			path := filepath.Join(repo, "blobs", digest)
			if conflict == "snapshot" {
				path = filepath.Join(repo, "snapshots", testRevision, file.Filename)
			}
			if conflict == "parent symlink" {
				path = filepath.Join(repo, "snapshots", testRevision)
				if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
					t.Fatal(err)
				}
				if err := os.Symlink(source, path); err != nil {
					t.Fatal(err)
				}
			} else {
				writeImportFile(t, path, "wrong weight\n")
			}
			partial := filepath.Join(repo, "blobs", digest+".incomplete")
			writeImportFile(t, partial, "partial download")
			_, err := client.Import(context.Background(), repository.ID, source, ImportOptions{CacheDir: cache})
			if err == nil {
				t.Fatal("cache conflict accepted")
			}
			if conflict != "parent symlink" {
				if data, _ := os.ReadFile(path); string(data) != "wrong weight\n" {
					t.Fatal("conflicting file was overwritten")
				}
			}
			if data, _ := os.ReadFile(partial); string(data) != "partial download" {
				t.Fatal("partial download was overwritten")
			}
			if _, err := os.Stat(filepath.Join(repo, "refs", "main")); !os.IsNotExist(err) {
				t.Fatal("failed import updated main")
			}
		})
	}
}

func TestImportReportsMissingFilesAndReusesCachedBlobs(t *testing.T) {
	client, source, cache, repository := importFixture(t)
	for _, name := range []string{".gitattributes", "config.json"} {
		if err := os.Remove(filepath.Join(source, name)); err != nil {
			t.Fatal(err)
		}
	}
	digest, _ := repository.Siblings[1].digest()
	writeImportFile(t, filepath.Join(cache, "models--lab--Model", "blobs", digest), "{}\n")
	result, err := client.Import(context.Background(), repository.ID, source, ImportOptions{CacheDir: cache})
	if err != nil || !reflect.DeepEqual(result.MissingFiles, []string{".gitattributes"}) || result.CachedFiles != 1 || result.LinkedFiles != 2 {
		t.Fatalf("partial import: %+v %v", result, err)
	}
	if data, err := os.ReadFile(filepath.Join(result.SnapshotPath, "config.json")); err != nil || string(data) != "{}\n" {
		t.Fatalf("cached blob was not linked into snapshot: %q %v", data, err)
	}
}

func TestImportRespectsDownloadLock(t *testing.T) {
	client, source, cache, repository := importFixture(t)
	digest, _ := repository.Siblings[0].digest()
	path := filepath.Join(cache, ".locks", "models--lab--Model", digest+".lock")
	writeImportFile(t, path, "")
	lock, err := os.OpenFile(path, os.O_RDWR, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer lock.Close()
	if err := syscall.Flock(int(lock.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		t.Fatal(err)
	}
	defer syscall.Flock(int(lock.Fd()), syscall.LOCK_UN)
	_, err = client.Import(context.Background(), repository.ID, source, ImportOptions{CacheDir: cache})
	if err == nil || !strings.Contains(err.Error(), "locked by another process") {
		t.Fatalf("import ignored download lock: %v", err)
	}
}

func TestHubCacheEnvironmentPrecedence(t *testing.T) {
	for _, name := range []string{"HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"} {
		t.Setenv(name, "")
	}
	for _, item := range []struct{ name, value, want string }{
		{"XDG_CACHE_HOME", "/xdg", "/xdg/huggingface/hub"},
		{"HF_HOME", "/hf", "/hf/hub"},
		{"HUGGINGFACE_HUB_CACHE", "/legacy", "/legacy"},
		{"HF_HUB_CACHE", "/explicit", "/explicit"},
	} {
		t.Setenv(item.name, item.value)
		if got, err := HubCacheDir(); err != nil || got != item.want {
			t.Fatalf("%s: got %q, %v; want %q", item.name, got, err, item.want)
		}
	}
}

func TestImportRejectsInvalidHubMetadataAndEmptySource(t *testing.T) {
	for _, invalid := range []string{"path traversal", "hash", "LFS size", "empty source"} {
		t.Run(invalid, func(t *testing.T) {
			client, source, cache, repository := importFixture(t)
			switch invalid {
			case "path traversal":
				repository.Siblings[0].Filename = "../escaped"
			case "hash":
				repository.Siblings[0].BlobID = ""
			case "LFS size":
				repository.Siblings[len(repository.Siblings)-1].LFS.Size++
			case "empty source":
				source = t.TempDir()
			}
			if _, err := client.Import(context.Background(), repository.ID, source, ImportOptions{CacheDir: cache}); err == nil {
				t.Fatalf("accepted %s", invalid)
			}
			if _, err := os.Stat(cache); !os.IsNotExist(err) {
				t.Fatalf("invalid import wrote the cache: %v", err)
			}
		})
	}
}

func TestImportRejectsCrossFilesystemHardlinksWithoutCopying(t *testing.T) {
	client, source, _, repository := importFixture(t)
	other, err := os.MkdirTemp("/dev/shm", "lil-import-test-*")
	if err != nil {
		t.Skipf("no second filesystem available: %v", err)
	}
	t.Cleanup(func() { os.RemoveAll(other) })
	sourceInfo, _ := os.Stat(source)
	otherInfo, _ := os.Stat(other)
	if sourceInfo.Sys().(*syscall.Stat_t).Dev == otherInfo.Sys().(*syscall.Stat_t).Dev {
		t.Skip("/dev/shm and the source share a filesystem")
	}
	cache := filepath.Join(other, "hub")
	_, err = client.Import(context.Background(), repository.ID, source, ImportOptions{CacheDir: cache})
	if err == nil || !strings.Contains(err.Error(), "different filesystems") {
		t.Fatalf("cross-filesystem import: %v", err)
	}
	if _, err := os.Stat(cache); !os.IsNotExist(err) {
		t.Fatalf("cross-filesystem import wrote the cache: %v", err)
	}
}

const testRevision = "0123456789abcdef0123456789abcdef01234567"
const secondTestRevision = "89abcdef0123456789abcdef0123456789abcdef"

func TestResolveFollowsHeadPinsRevisionsAndCachesFiles(t *testing.T) {
	var resolves, fetches atomic.Int32
	var fail atomic.Bool
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if fail.Load() {
			http.Error(writer, "unavailable", http.StatusServiceUnavailable)
			return
		}
		if request.Header.Get("Authorization") != "Bearer test-token" {
			http.Error(writer, "missing token", http.StatusUnauthorized)
			return
		}
		if request.URL.Query().Get("blobs") != "true" && strings.HasPrefix(request.URL.Path, "/api/") {
			http.Error(writer, "sizes require blobs=true", http.StatusBadRequest)
			return
		}
		listing := func(revision string) string {
			return fmt.Sprintf(`{"id":"lab/lil-catalog","sha":"%s","siblings":[{"rfilename":"README.md","size":10},{"rfilename":"Alpha/lil.yaml","size":40},{"rfilename":"Beta/lil.yaml","size":40},{"rfilename":"Alpha/nested/lil.yaml","size":1}]}`, revision)
		}
		switch request.URL.Path {
		case "/api/models/lab/lil-catalog":
			count := resolves.Add(1)
			revision := testRevision
			if count > 1 {
				revision = secondTestRevision
			}
			fmt.Fprint(writer, listing(revision))
		case "/api/models/lab/lil-catalog/revision/" + testRevision:
			fmt.Fprint(writer, listing(testRevision))
		case "/api/models/other/Weights":
			fmt.Fprintf(writer, `{"id":"other/Weights","sha":"%s","siblings":[{"rfilename":"config.json","size":12},{"rfilename":"model-00001.safetensors","size":1000},{"rfilename":"model-00002.safetensors","size":2000}]}`, testRevision)
		case "/lab/lil-catalog/resolve/" + testRevision + "/Alpha/lil.yaml":
			fetches.Add(1)
			fmt.Fprint(writer, "schema_version: 1\nkind: model\ndescription: first\n")
		case "/lab/lil-catalog/resolve/" + secondTestRevision + "/Alpha/lil.yaml":
			fmt.Fprint(writer, "schema_version: 1\nkind: model\ndescription: second\n")
		case "/other/Weights/resolve/" + testRevision + "/config.json":
			fmt.Fprint(writer, `{"architectures":["X"]}`)
		default:
			http.NotFound(writer, request)
		}
	}))
	defer server.Close()

	client := &Client{
		Endpoint: server.URL, Owner: "lab", Token: "test-token",
		CacheDir: t.TempDir(), HTTPClient: server.Client(),
	}
	if client.CatalogID() != "lab/lil-catalog" {
		t.Fatalf("catalog id: %s", client.CatalogID())
	}
	first, err := client.Resolve(context.Background(), client.CatalogID())
	if err != nil {
		t.Fatal(err)
	}
	if got := first.CatalogEntries(); !reflect.DeepEqual(got, []string{"Alpha", "Beta"}) {
		t.Fatalf("catalog entries: %v", got)
	}
	manifest, cached, err := client.FetchFile(context.Background(), first, "Alpha/lil.yaml", ManifestLimit)
	if err != nil || cached || !strings.Contains(string(manifest), "first") {
		t.Fatalf("first fetch: cached=%t manifest=%q error=%v", cached, manifest, err)
	}
	_, cached, err = client.FetchFile(context.Background(), first, "Alpha/lil.yaml", ManifestLimit)
	if err != nil || !cached || fetches.Load() != 1 {
		t.Fatalf("cached fetch: cached=%t fetches=%d error=%v", cached, fetches.Load(), err)
	}
	pinned, err := client.ResolveAt(context.Background(), client.CatalogID(), testRevision)
	if err != nil || pinned.Revision != testRevision {
		t.Fatalf("pinned resolve: %+v %v", pinned, err)
	}
	second, err := client.Resolve(context.Background(), client.CatalogID())
	if err != nil || second.Revision != secondTestRevision {
		t.Fatalf("head was not refreshed: %+v %v", second, err)
	}
	if manifest, _, err = client.FetchFile(context.Background(), second, "Alpha/lil.yaml", ManifestLimit); err != nil || !strings.Contains(string(manifest), "second") {
		t.Fatalf("head manifest: %q %v", manifest, err)
	}
	weights, err := client.ResolveAt(context.Background(), "other/Weights", "")
	if err != nil || weights.SafetensorsBytes() != 3000 {
		t.Fatalf("foreign repository: %+v %v", weights, err)
	}
	if config, _, err := client.FetchFile(context.Background(), weights, "config.json", ConfigLimit); err != nil || !strings.Contains(string(config), "architectures") {
		t.Fatalf("config fetch: %q %v", config, err)
	}
	for _, name := range []string{"../etc/passwd", "Alpha/nested/lil.yaml", "/absolute"} {
		if _, _, err := client.FetchFile(context.Background(), first, name, ConfigLimit); err == nil {
			t.Errorf("file name %q was accepted", name)
		}
	}
	if _, err := client.ResolveAt(context.Background(), "lab/missing", ""); !errors.Is(err, ErrNotFound) {
		t.Fatalf("missing repository: %v", err)
	}
	fail.Store(true)
	if _, err := client.Resolve(context.Background(), client.CatalogID()); err == nil {
		t.Fatal("resolution silently succeeded while the Hub was unavailable")
	}
}
