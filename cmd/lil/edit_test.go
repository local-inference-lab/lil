// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/local-inference-lab/lil/internal/hfhub"
)

func TestEditManifest(t *testing.T) {
	for _, mode := range []string{"local", "repository", "unchanged", "invalid", "editor-failure", "unauthenticated", "conflict", "denied", "repository-unchanged", "multiple-documents"} {
		t.Run(mode, func(t *testing.T) {
			t.Setenv("XDG_CONFIG_HOME", t.TempDir())
			name := "GLM-5.3-NVFP4"
			original, err := os.ReadFile(filepath.Join(testModelsConfig(), name, "lil.yaml"))
			if err != nil {
				t.Fatal(err)
			}
			updated := append(bytes.Clone(original), []byte("\n# editor change.\n")...)
			sha := strings.Repeat("a", 40)
			posts := 0
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch {
				case r.Method == "POST":
					posts++
					if r.URL.Path != "/api/models/local-inference-lab/lil-catalog/commit/main" || r.Header.Get("Authorization") != "Bearer test-token" || r.Header.Get("Content-Type") != "application/x-ndjson" {
						t.Errorf("unexpected upload: %s %v", r.URL, r.Header)
					}
					decoder := json.NewDecoder(r.Body)
					var header struct {
						Key   string
						Value struct{ ParentCommit string }
					}
					var file struct {
						Key   string
						Value struct{ Path, Encoding, Content string }
					}
					if err := decoder.Decode(&header); err != nil {
						t.Error(err)
					}
					if err := decoder.Decode(&file); err != nil {
						t.Error(err)
					}
					contents, err := base64.StdEncoding.DecodeString(file.Value.Content)
					if err != nil || header.Key != "header" || header.Value.ParentCommit != sha || file.Key != "file" || file.Value.Path != name+"/lil.yaml" || file.Value.Encoding != "base64" || !bytes.Equal(contents, updated) {
						t.Errorf("unexpected commit: %+v %+v", header, file)
					}
					if mode == "denied" {
						w.WriteHeader(http.StatusForbidden)
						fmt.Fprint(w, `{"error":"write access required"}`)
						return
					}
					if mode == "conflict" {
						w.WriteHeader(http.StatusConflict)
						fmt.Fprint(w, `{"error":"parent commit changed"}`)
						return
					}
					fmt.Fprintf(w, `{"commitOid":%q}`, strings.Repeat("b", 40))
				case strings.HasPrefix(r.URL.Path, "/api/models/"):
					json.NewEncoder(w).Encode(hfhub.Repository{ID: "local-inference-lab/lil-catalog", Revision: sha, Siblings: []hfhub.Sibling{{Filename: name + "/lil.yaml"}}})
				default:
					w.Write(original)
				}
			}))
			defer server.Close()
			client := &hfhub.Client{Endpoint: server.URL, Owner: modelRepositoryOwner, Token: "test-token", CacheDir: t.TempDir()}
			path, err := localManifestPath(name)
			if err != nil {
				t.Fatal(err)
			}
			// Repository editing must ignore an existing local override.
			if mode == "repository" {
				os.MkdirAll(filepath.Dir(path), 0o755)
				os.WriteFile(path, []byte("local override"), 0o600)
			}
			script := filepath.Join(t.TempDir(), "editor script")
			scriptBody := "#!/bin/sh\n[ \"$1\" = 'argument with spaces' ] || exit 8\nprintf '\\n# editor change.\\n' >> \"$2\"\n"
			switch mode {
			case "unchanged", "repository-unchanged":
				scriptBody = "#!/bin/sh\nexit 0\n"
			case "multiple-documents":
				scriptBody = "#!/bin/sh\nprintf '\\n---\\nkind: draft\\n' >> \"$2\"\n"
			case "invalid":
				scriptBody = "#!/bin/sh\nprintf 'invalid: [' > \"$2\"\n"
			case "editor-failure":
				scriptBody = "#!/bin/sh\nexit 9\n"
			case "unauthenticated":
				client.Token = ""
			}
			if err := os.WriteFile(script, []byte(scriptBody), 0o700); err != nil {
				t.Fatal(err)
			}
			t.Setenv("EDITOR", fmt.Sprintf("'%s' 'argument with spaces'", script))
			upstream := mode == "repository" || mode == "conflict" || mode == "unauthenticated" || mode == "denied" || mode == "repository-unchanged"
			err = editManifest(context.Background(), client, name, upstream)
			wantError := mode == "invalid" || mode == "editor-failure" || mode == "unauthenticated" || mode == "conflict" || mode == "denied" || mode == "multiple-documents"
			if (err != nil) != wantError {
				t.Fatalf("error = %v", err)
			}
			if wantError && mode != "unauthenticated" {
				_, retained, ok := strings.Cut(err.Error(), "edited file retained at ")
				if !ok {
					t.Fatalf("missing recovery path: %v", err)
				}
				if _, err := os.Stat(retained); err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { os.Remove(retained) })
			}
			expectedPosts := 0
			if mode == "repository" || mode == "conflict" || mode == "denied" {
				expectedPosts = 1
			}
			if posts != expectedPosts {
				t.Fatalf("uploads = %d, want %d", posts, expectedPosts)
			}
			data, readErr := os.ReadFile(path)
			switch mode {
			case "local":
				if readErr != nil || !bytes.Equal(data, updated) {
					t.Fatalf("local contents: %s, %v", data, readErr)
				}
				// Every Hub entry reader uses this resolution path.
				manifest, label, commit, err := catalogManifest(context.Background(), client, hfhub.Repository{}, name)
				if err != nil || !bytes.Equal(manifest, updated) || label != path || commit != "" {
					t.Fatalf("override resolution: %s %s %v", label, commit, err)
				}
				t.Setenv("EDITOR", "true")
				client.Endpoint = "http://127.0.0.1:1"
				if err := editManifest(context.Background(), client, name, false); err != nil {
					t.Fatalf("edit existing override requires network: %v", err)
				}
			case "repository":
				if string(data) != "local override" {
					t.Fatalf("overwrote local override: %q", data)
				}
			default:
				if !os.IsNotExist(readErr) {
					t.Fatalf("unexpected local override: %v", readErr)
				}
			}
		})
	}
}

func TestEditArguments(t *testing.T) {
	for _, args := range [][]string{{"edit"}, {"edit", "one", "two"}, {"edit", "../bad"}} {
		status, err := execute(context.Background(), args)
		if status == 0 || err == nil {
			t.Fatalf("accepted %v", args)
		}
	}
	status, err := execute(context.Background(), []string{"edit", "--help"})
	if status != 0 || err != nil {
		t.Fatalf("help: %d %v", status, err)
	}
}

func TestLocalManifestUsedByHubModelLoader(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	name := "GLM-5.3-NVFP4"
	manifest, err := os.ReadFile(filepath.Join(testModelsConfig(), name, "lil.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	// A manifest identity override must also select the corresponding checkpoint facts.
	manifest = bytes.ReplaceAll(manifest, []byte("local-inference-lab/"+name), []byte("local-inference-lab/custom-weights"))
	config, err := os.ReadFile(filepath.Join(testModelsConfig(), name, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	path, err := localManifestPath(name)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, manifest, 0o600); err != nil {
		t.Fatal(err)
	}
	sha := strings.Repeat("c", 40)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/models/local-inference-lab/custom-weights":
			json.NewEncoder(w).Encode(hfhub.Repository{ID: "local-inference-lab/custom-weights", Revision: sha, Siblings: []hfhub.Sibling{{Filename: "config.json"}, {Filename: "model.safetensors", Size: 123456}}})
		case "/local-inference-lab/custom-weights/resolve/" + sha + "/config.json":
			w.Write(config)
		default:
			t.Errorf("unexpected upstream request: %s", r.URL)
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	client := &hfhub.Client{Endpoint: server.URL, CacheDir: t.TempDir()}
	families, err := embeddedModelFamilies()
	if err != nil {
		t.Fatal(err)
	}
	kind, err := hubEntryKind(context.Background(), client, hfhub.Repository{}, name)
	if err != nil || kind != "model" {
		t.Fatalf("kind = %s: %v", kind, err)
	}
	profile, err := hubModelEntry(context.Background(), client, families, hfhub.Repository{}, name)
	if err != nil {
		t.Fatal(err)
	}
	if profile.Model != "local-inference-lab/custom-weights" || profile.ManifestCommit != "" || profile.Facts == nil {
		t.Fatalf("override not applied: %+v", profile)
	}
	directory, err := loadDirectoryCatalog(testModelsConfig())
	if err != nil {
		t.Fatal(err)
	}
	if directory.models[name].Model != "local-inference-lab/"+name {
		t.Fatal("local override changed explicit catalog")
	}
}
