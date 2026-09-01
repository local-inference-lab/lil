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

var (
	repositoryIDPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$`)
	revisionPattern     = regexp.MustCompile(`^[0-9a-f]{40}$`)
	nextLinkPattern     = regexp.MustCompile(`<([^>]+)>;\s*rel="next"`)
)

type Repository struct {
	ID           string    `json:"id"`
	Revision     string    `json:"sha"`
	Private      bool      `json:"private"`
	LastModified time.Time `json:"lastModified"`
	Siblings     []struct {
		Filename string `json:"rfilename"`
	} `json:"siblings"`
}

func (repository Repository) HasFile(filename string) bool {
	for _, sibling := range repository.Siblings {
		if sibling.Filename == filename {
			return true
		}
	}
	return false
}

type Discovery struct {
	Repositories []Repository
	Cached       bool
	Warning      string
}

type Client struct {
	Endpoint   string
	Owner      string
	Token      string
	CacheDir   string
	Offline    bool
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

func (client *Client) catalogPath() string {
	return filepath.Join(client.CacheDir, "catalog.json")
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

func (client *Client) readCatalog() ([]Repository, error) {
	data, err := os.ReadFile(client.catalogPath())
	if err != nil {
		return nil, err
	}
	var repositories []Repository
	if err := json.Unmarshal(data, &repositories); err != nil {
		return nil, fmt.Errorf("invalid cached Hugging Face catalog: %w", err)
	}
	return repositories, nil
}

func (client *Client) writeCatalog(repositories []Repository) error {
	data, err := json.MarshalIndent(repositories, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return atomicWrite(client.catalogPath(), data)
}

func (client *Client) Discover(ctx context.Context) (Discovery, error) {
	if client.Offline {
		repositories, err := client.readCatalog()
		if err != nil {
			return Discovery{}, fmt.Errorf("Hugging Face is offline and no cached model catalog is available: %w", err)
		}
		return Discovery{Repositories: repositories, Cached: true}, nil
	}
	query := url.Values{"author": {client.Owner}, "full": {"true"}, "limit": {"100"}}
	next := client.endpoint() + "/api/models?" + query.Encode()
	seen := map[string]bool{}
	var repositories []Repository
	for next != "" {
		if seen[next] || len(seen) >= 100 {
			return Discovery{}, fmt.Errorf("invalid Hugging Face pagination while discovering %s", client.Owner)
		}
		seen[next] = true
		response, err := client.request(ctx, http.MethodGet, next)
		if err != nil {
			return client.cachedDiscovery(err)
		}
		if response.StatusCode != http.StatusOK {
			responseErr := responseError(response)
			response.Body.Close()
			return client.cachedDiscovery(responseErr)
		}
		var page []Repository
		decodeErr := json.NewDecoder(response.Body).Decode(&page)
		response.Body.Close()
		if decodeErr != nil {
			return client.cachedDiscovery(decodeErr)
		}
		repositories = append(repositories, page...)
		next = ""
		if match := nextLinkPattern.FindStringSubmatch(response.Header.Get("Link")); match != nil {
			next = match[1]
		}
	}
	sort.Slice(repositories, func(i, j int) bool { return repositories[i].ID < repositories[j].ID })
	if err := client.writeCatalog(repositories); err != nil {
		return Discovery{}, fmt.Errorf("cannot cache Hugging Face model catalog: %w", err)
	}
	return Discovery{Repositories: repositories}, nil
}

func (client *Client) cachedDiscovery(cause error) (Discovery, error) {
	repositories, err := client.readCatalog()
	if err != nil {
		return Discovery{}, fmt.Errorf("cannot discover Hugging Face models: %w", cause)
	}
	return Discovery{
		Repositories: repositories,
		Cached:       true,
		Warning:      "Hugging Face discovery failed; using cached catalog: " + cause.Error(),
	}, nil
}

func (client *Client) NormalizeRepository(model string) (string, error) {
	model = strings.TrimSpace(model)
	if !strings.Contains(model, "/") {
		model = client.Owner + "/" + model
	}
	if !repositoryIDPattern.MatchString(model) {
		return "", fmt.Errorf("model must be a Hugging Face repository name or owner/name ID")
	}
	owner, _, _ := strings.Cut(model, "/")
	if owner != client.Owner {
		return "", fmt.Errorf("model repository must belong to %s; got %s", client.Owner, model)
	}
	return model, nil
}

func (client *Client) Resolve(ctx context.Context, model string) (Repository, error) {
	id, err := client.NormalizeRepository(model)
	if err != nil {
		return Repository{}, err
	}
	if client.Offline {
		return client.resolveCached(id)
	}
	owner, name, _ := strings.Cut(id, "/")
	target := client.endpoint() + "/api/models/" + url.PathEscape(owner) + "/" + url.PathEscape(name)
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
	if err := validateRepository(repository, id); err != nil {
		return Repository{}, err
	}
	client.mergeCatalog(repository)
	return repository, nil
}

func validateRepository(repository Repository, expectedID string) error {
	if repository.ID != expectedID || !revisionPattern.MatchString(repository.Revision) {
		return fmt.Errorf("Hugging Face returned invalid metadata for %s", expectedID)
	}
	return nil
}

func (client *Client) resolveCached(id string) (Repository, error) {
	repositories, err := client.readCatalog()
	if err != nil {
		return Repository{}, fmt.Errorf("no cached Hugging Face metadata for %s", id)
	}
	for _, repository := range repositories {
		if repository.ID == id {
			return repository, validateRepository(repository, id)
		}
	}
	return Repository{}, fmt.Errorf("no cached Hugging Face metadata for %s", id)
}

func (client *Client) mergeCatalog(repository Repository) {
	repositories, _ := client.readCatalog()
	found := false
	for index := range repositories {
		if repositories[index].ID == repository.ID {
			repositories[index] = repository
			found = true
			break
		}
	}
	if !found {
		repositories = append(repositories, repository)
	}
	sort.Slice(repositories, func(i, j int) bool { return repositories[i].ID < repositories[j].ID })
	_ = client.writeCatalog(repositories)
}

func (client *Client) manifestPath(repository Repository) (string, error) {
	if !repositoryIDPattern.MatchString(repository.ID) || !revisionPattern.MatchString(repository.Revision) {
		return "", fmt.Errorf("invalid Hugging Face repository metadata")
	}
	return filepath.Join(
		client.CacheDir,
		"manifests",
		strings.ReplaceAll(repository.ID, "/", "--"),
		repository.Revision,
		"lil.yaml",
	), nil
}

func (client *Client) FetchManifest(ctx context.Context, repository Repository) ([]byte, bool, error) {
	if !repository.HasFile("lil.yaml") {
		return nil, false, fmt.Errorf("%s does not contain lil.yaml", repository.ID)
	}
	path, err := client.manifestPath(repository)
	if err != nil {
		return nil, false, err
	}
	if data, err := os.ReadFile(path); err == nil {
		return data, true, nil
	}
	if client.Offline {
		return nil, false, fmt.Errorf("lil.yaml for %s@%s is not cached", repository.ID, repository.Revision)
	}
	owner, name, _ := strings.Cut(repository.ID, "/")
	target := client.endpoint() + "/" + url.PathEscape(owner) + "/" + url.PathEscape(name) +
		"/resolve/" + repository.Revision + "/lil.yaml"
	response, err := client.request(ctx, http.MethodGet, target)
	if err != nil {
		return nil, false, fmt.Errorf("cannot fetch %s: %w", repository.ID+"/lil.yaml", err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return nil, false, fmt.Errorf("cannot fetch %s: %w", repository.ID+"/lil.yaml", responseError(response))
	}
	data, err := io.ReadAll(io.LimitReader(response.Body, 1<<20))
	if err != nil {
		return nil, false, err
	}
	if len(data) == 1<<20 {
		return nil, false, fmt.Errorf("%s/lil.yaml exceeds 1 MiB", repository.ID)
	}
	if err := atomicWrite(path, data); err != nil {
		return nil, false, fmt.Errorf("cannot cache %s/lil.yaml: %w", repository.ID, err)
	}
	return data, false, nil
}
