// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package hfhub

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
)

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
