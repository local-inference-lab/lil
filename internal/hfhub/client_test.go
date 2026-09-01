// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package hfhub

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

const testRevision = "0123456789abcdef0123456789abcdef01234567"
const secondTestRevision = "89abcdef0123456789abcdef0123456789abcdef"

func TestDiscoverPaginatesAuthenticatesAndCachesManifest(t *testing.T) {
	var manifestRequests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.Header.Get("Authorization") != "Bearer test-token" {
			http.Error(writer, "missing token", http.StatusUnauthorized)
			return
		}
		switch request.URL.Path {
		case "/api/models":
			if request.URL.Query().Get("cursor") == "second" {
				fmt.Fprintf(writer, `[{"id":"local-inference-lab/Zeta","sha":"%s","siblings":[]}]`, testRevision)
				return
			}
			writer.Header().Set("Link", "<"+serverURL(request)+"/api/models?cursor=second>; rel=\"next\"")
			fmt.Fprintf(writer, `[{"id":"local-inference-lab/Alpha","sha":"%s","siblings":[{"rfilename":"lil.yaml"}]}]`, testRevision)
		case "/local-inference-lab/Alpha/resolve/" + testRevision + "/lil.yaml":
			manifestRequests.Add(1)
			fmt.Fprint(writer, "schema_version: 1\nkind: model\n")
		default:
			http.NotFound(writer, request)
		}
	}))
	defer server.Close()

	client := &Client{
		Endpoint: server.URL, Owner: "local-inference-lab", Token: "test-token",
		CacheDir: t.TempDir(), HTTPClient: server.Client(),
	}
	discovery, err := client.Discover(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if discovery.Cached || len(discovery.Repositories) != 2 || discovery.Repositories[0].ID != "local-inference-lab/Alpha" {
		t.Fatalf("unexpected discovery: %+v", discovery)
	}
	manifest, cached, err := client.FetchManifest(context.Background(), discovery.Repositories[0])
	if err != nil || cached || !strings.Contains(string(manifest), "kind: model") {
		t.Fatalf("first manifest fetch: cached=%t manifest=%q error=%v", cached, manifest, err)
	}
	_, cached, err = client.FetchManifest(context.Background(), discovery.Repositories[0])
	if err != nil || !cached || manifestRequests.Load() != 1 {
		t.Fatalf("cached manifest fetch: cached=%t requests=%d error=%v", cached, manifestRequests.Load(), err)
	}

	offline := *client
	offline.Offline = true
	offlineDiscovery, err := offline.Discover(context.Background())
	if err != nil || !offlineDiscovery.Cached || len(offlineDiscovery.Repositories) != 2 {
		t.Fatalf("offline discovery: %+v error=%v", offlineDiscovery, err)
	}
}

func serverURL(request *http.Request) string {
	return "http://" + request.Host
}

func TestNormalizeRepositoryRestrictsOwner(t *testing.T) {
	client := &Client{Owner: "local-inference-lab"}
	if got, err := client.NormalizeRepository("Model"); err != nil || got != "local-inference-lab/Model" {
		t.Fatalf("normalized repository: %q, %v", got, err)
	}
	if _, err := client.NormalizeRepository("someone-else/Model"); err == nil {
		t.Fatal("foreign owner was accepted")
	}
}

func TestResolveChecksRepositoryHeadAndFetchesItsManifest(t *testing.T) {
	var resolves atomic.Int32
	var fail atomic.Bool
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if fail.Load() {
			http.Error(writer, "unavailable", http.StatusServiceUnavailable)
			return
		}
		switch request.URL.Path {
		case "/api/models/local-inference-lab/Model":
			count := resolves.Add(1)
			revision := testRevision
			if count > 1 {
				revision = secondTestRevision
			}
			fmt.Fprintf(
				writer,
				`{"id":"local-inference-lab/Model","sha":"%s","siblings":[{"rfilename":"lil.yaml"}]}`,
				revision,
			)
		case "/local-inference-lab/Model/resolve/" + testRevision + "/lil.yaml":
			fmt.Fprint(writer, "schema_version: 1\nkind: model\ndescription: first\n")
		case "/local-inference-lab/Model/resolve/" + secondTestRevision + "/lil.yaml":
			fmt.Fprint(writer, "schema_version: 1\nkind: model\ndescription: second\n")
		default:
			http.NotFound(writer, request)
		}
	}))
	defer server.Close()

	client := &Client{
		Endpoint: server.URL, Owner: "local-inference-lab",
		CacheDir: t.TempDir(), HTTPClient: server.Client(),
	}
	first, err := client.Resolve(context.Background(), "Model")
	if err != nil {
		t.Fatal(err)
	}
	firstManifest, _, err := client.FetchManifest(context.Background(), first)
	if err != nil {
		t.Fatal(err)
	}
	second, err := client.Resolve(context.Background(), "Model")
	if err != nil {
		t.Fatal(err)
	}
	secondManifest, _, err := client.FetchManifest(context.Background(), second)
	if err != nil {
		t.Fatal(err)
	}
	if first.Revision != testRevision || second.Revision != secondTestRevision ||
		!strings.Contains(string(firstManifest), "first") ||
		!strings.Contains(string(secondManifest), "second") {
		t.Fatalf(
			"repository head was not refreshed: first=%+v second=%+v manifests=%q/%q",
			first, second, firstManifest, secondManifest,
		)
	}

	fail.Store(true)
	if _, err := client.Resolve(context.Background(), "Model"); err == nil {
		t.Fatal("repository resolution silently fell back to cached metadata")
	}
}
