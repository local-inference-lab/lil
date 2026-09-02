// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package hfhub

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
)

const defaultEndpoint = "https://huggingface.co"

// Size caps for files fetched from a repository at a resolved commit.
const (
	ManifestLimit = 1 << 20
	ConfigLimit   = 16 << 20
)

// CatalogRepositoryName is the repository under the owner that holds one
// <model>/lil.yaml manifest per launchable model or draft.
const CatalogRepositoryName = "lil-catalog"

// ErrNotFound reports a repository, revision, or file the Hub does not have.
var ErrNotFound = errors.New("not found")

var (
	repositoryIDPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$`)
	revisionPattern     = regexp.MustCompile(`^[0-9a-f]{40}$`)
	repositoryFilePath  = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.-]*(/[A-Za-z0-9][A-Za-z0-9_.-]*)?$`)
)

type Sibling struct {
	Filename string `json:"rfilename"`
	Size     int64  `json:"size,omitempty"`
}

// Repository is a listing of one repository at one commit, with file sizes.
type Repository struct {
	ID           string    `json:"id"`
	Revision     string    `json:"sha"`
	Private      bool      `json:"private"`
	LastModified time.Time `json:"lastModified"`
	Siblings     []Sibling `json:"siblings"`
}

func (repository Repository) HasFile(filename string) bool {
	for _, sibling := range repository.Siblings {
		if sibling.Filename == filename {
			return true
		}
	}
	return false
}

// CatalogEntries lists the names that have a <name>/lil.yaml manifest in
// this repository.
func (repository Repository) CatalogEntries() []string {
	entries := []string{}
	for _, sibling := range repository.Siblings {
		directory, file, ok := strings.Cut(sibling.Filename, "/")
		if ok && file == "lil.yaml" && !strings.Contains(directory, "/") {
			entries = append(entries, directory)
		}
	}
	sort.Strings(entries)
	return entries
}

// SafetensorsBytes sums the stored size of every safetensors shard.
func (repository Repository) SafetensorsBytes() int64 {
	var total int64
	for _, sibling := range repository.Siblings {
		if strings.HasSuffix(sibling.Filename, ".safetensors") {
			total += sibling.Size
		}
	}
	return total
}

type Client struct {
	Endpoint   string
	Owner      string
	Token      string
	CacheDir   string
	HTTPClient *http.Client
}

func DefaultClient(owner string) (*Client, error) {
	cacheRoot := os.Getenv("LIL_CACHE_DIR")
	if cacheRoot == "" {
		if xdg := os.Getenv("XDG_CACHE_HOME"); xdg != "" {
			cacheRoot = filepath.Join(xdg, "lil")
		} else {
			home, err := os.UserHomeDir()
			if err != nil {
				return nil, err
			}
			cacheRoot = filepath.Join(home, ".cache", "lil")
		}
	}
	return &Client{
		Endpoint:   defaultEndpoint,
		Owner:      owner,
		Token:      huggingFaceToken(),
		CacheDir:   filepath.Join(cacheRoot, "huggingface"),
		HTTPClient: &http.Client{Timeout: 30 * time.Second},
	}, nil
}

// CatalogID names the owner's catalog repository.
func (client *Client) CatalogID() string {
	return client.Owner + "/" + CatalogRepositoryName
}

func huggingFaceToken() string {
	for _, name := range []string{"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"} {
		if token := strings.TrimSpace(os.Getenv(name)); token != "" {
			return token
		}
	}
	hfHome := os.Getenv("HF_HOME")
	if hfHome == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return ""
		}
		hfHome = filepath.Join(home, ".cache", "huggingface")
	}
	data, err := os.ReadFile(filepath.Join(hfHome, "token"))
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}

func (client *Client) httpClient() *http.Client {
	if client.HTTPClient != nil {
		return client.HTTPClient
	}
	return &http.Client{Timeout: 30 * time.Second}
}

func (client *Client) endpoint() string {
	if client.Endpoint == "" {
		return defaultEndpoint
	}
	return strings.TrimRight(client.Endpoint, "/")
}

func (client *Client) request(ctx context.Context, method, target string) (*http.Response, error) {
	request, err := http.NewRequestWithContext(ctx, method, target, nil)
	if err != nil {
		return nil, err
	}
	request.Header.Set("Accept", "application/json")
	request.Header.Set("User-Agent", "lil")
	if client.Token != "" {
		request.Header.Set("Authorization", "Bearer "+client.Token)
	}
	return client.httpClient().Do(request)
}

func responseError(response *http.Response) error {
	if response.StatusCode == http.StatusNotFound {
		return ErrNotFound
	}
	message := strings.TrimSpace(response.Header.Get("X-Error-Message"))
	if message == "" {
		data, _ := io.ReadAll(io.LimitReader(response.Body, 64<<10))
		var payload struct {
			Error string `json:"error"`
		}
		if json.Unmarshal(data, &payload) == nil {
			message = strings.TrimSpace(payload.Error)
		}
	}
	if message == "" {
		message = response.Status
	}
	return errors.New(message)
}

func atomicWrite(path string, data []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	temporary, err := os.CreateTemp(filepath.Dir(path), ".lil-cache-*")
	if err != nil {
		return err
	}
	temporaryPath := temporary.Name()
	defer os.Remove(temporaryPath)
	if _, err := temporary.Write(data); err != nil {
		temporary.Close()
		return err
	}
	if err := temporary.Chmod(0o644); err != nil {
		temporary.Close()
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	return os.Rename(temporaryPath, path)
}

// Resolve fetches a repository's listing at its head commit. Every launch
// command resolves through here, so a network failure fails the command
// instead of reusing stale policy.
func (client *Client) Resolve(ctx context.Context, id string) (Repository, error) {
	return client.ResolveAt(ctx, id, "")
}

// ResolveAt fetches the listing of any accessible repository at its head,
// or at a pinned commit when revision is set. File sizes are included and
// cached under the resolved commit.
func (client *Client) ResolveAt(ctx context.Context, id, revision string) (Repository, error) {
	if !repositoryIDPattern.MatchString(id) {
		return Repository{}, fmt.Errorf("repository must have owner/name form; got %q", id)
	}
	if revision != "" && !revisionPattern.MatchString(revision) {
		return Repository{}, fmt.Errorf("revision for %s must be a 40-character commit SHA", id)
	}
	owner, name, _ := strings.Cut(id, "/")
	target := client.endpoint() + "/api/models/" + url.PathEscape(owner) + "/" + url.PathEscape(name)
	if revision != "" {
		target += "/revision/" + revision
	}
	target += "?blobs=true"
	response, err := client.request(ctx, http.MethodGet, target)
	if err != nil {
		return Repository{}, fmt.Errorf("cannot resolve Hugging Face repository %s: %w", id, err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return Repository{}, fmt.Errorf("cannot resolve Hugging Face repository %s: %w", id, responseError(response))
	}
	var repository Repository
	if err := json.NewDecoder(response.Body).Decode(&repository); err != nil {
		return Repository{}, fmt.Errorf("cannot decode Hugging Face repository %s: %w", id, err)
	}
	if repository.ID != id || !revisionPattern.MatchString(repository.Revision) {
		return Repository{}, fmt.Errorf("Hugging Face returned invalid metadata for %s", id)
	}
	if revision != "" && repository.Revision != revision {
		return Repository{}, fmt.Errorf("Hugging Face resolved %s@%s to %s", id, revision[:12], repository.Revision[:12])
	}
	return repository, nil
}

func (client *Client) revisionDir(repository Repository) (string, error) {
	if !repositoryIDPattern.MatchString(repository.ID) || !revisionPattern.MatchString(repository.Revision) {
		return "", fmt.Errorf("invalid Hugging Face repository metadata")
	}
	return filepath.Join(
		client.CacheDir, "repositories",
		strings.ReplaceAll(repository.ID, "/", "--"), repository.Revision,
	), nil
}

// FetchFile returns a file from the repository at its resolved commit,
// reusing bytes cached for that immutable commit. Paths may be a root file
// or one directory deep.
func (client *Client) FetchFile(ctx context.Context, repository Repository, name string, limit int64) ([]byte, bool, error) {
	if !repositoryFilePath.MatchString(name) {
		return nil, false, fmt.Errorf("unsupported repository file name %q", name)
	}
	if !repository.HasFile(name) {
		return nil, false, fmt.Errorf("%s does not contain %s at %s", repository.ID, name, repository.Revision[:12])
	}
	directory, err := client.revisionDir(repository)
	if err != nil {
		return nil, false, err
	}
	path := filepath.Join(directory, name)
	if data, err := os.ReadFile(path); err == nil {
		return data, true, nil
	}
	owner, repositoryName, _ := strings.Cut(repository.ID, "/")
	target := client.endpoint() + "/" + url.PathEscape(owner) + "/" + url.PathEscape(repositoryName) +
		"/resolve/" + repository.Revision + "/" + name
	response, err := client.request(ctx, http.MethodGet, target)
	if err != nil {
		return nil, false, fmt.Errorf("cannot fetch %s/%s: %w", repository.ID, name, err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return nil, false, fmt.Errorf("cannot fetch %s/%s: %w", repository.ID, name, responseError(response))
	}
	data, err := io.ReadAll(io.LimitReader(response.Body, limit))
	if err != nil {
		return nil, false, err
	}
	if int64(len(data)) == limit {
		return nil, false, fmt.Errorf("%s/%s exceeds %d bytes", repository.ID, name, limit)
	}
	if err := atomicWrite(path, data); err != nil {
		return nil, false, fmt.Errorf("cannot cache %s/%s: %w", repository.ID, name, err)
	}
	return data, false, nil
}
