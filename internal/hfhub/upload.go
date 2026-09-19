// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package hfhub

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// UploadManifest commits one small catalog manifest to main, conditional on
// the repository still having the commit from which the edit started.
func (client *Client) UploadManifest(ctx context.Context, repository Repository, name string, data []byte) (string, error) {
	if strings.TrimSpace(client.Token) == "" {
		return "", fmt.Errorf("Hugging Face authentication is required to upload")
	}
	if !repositoryIDPattern.MatchString(repository.ID) || !revisionPattern.MatchString(repository.Revision) {
		return "", fmt.Errorf("invalid Hugging Face repository metadata")
	}
	if !repositoryFilePath.MatchString(name) || !strings.HasSuffix(name, "/lil.yaml") || len(data) >= ManifestLimit {
		return "", fmt.Errorf("upload requires a <name>/lil.yaml manifest smaller than %d bytes", ManifestLimit)
	}
	var body bytes.Buffer
	encoder := json.NewEncoder(&body)
	for _, entry := range []map[string]any{
		{"key": "header", "value": map[string]string{"summary": "Update " + name, "description": "", "parentCommit": repository.Revision}},
		{"key": "file", "value": map[string]string{"path": name, "encoding": "base64", "content": base64.StdEncoding.EncodeToString(data)}},
	} {
		if err := encoder.Encode(entry); err != nil {
			return "", err
		}
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, client.endpoint()+"/api/models/"+repository.ID+"/commit/main", &body)
	if err != nil {
		return "", err
	}
	request.Header.Set("Authorization", "Bearer "+client.Token)
	request.Header.Set("Content-Type", "application/x-ndjson")
	request.Header.Set("User-Agent", "lil")
	response, err := client.httpClient().Do(request)
	if err != nil {
		return "", fmt.Errorf("upload %s/%s: %w", repository.ID, name, err)
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return "", fmt.Errorf("upload %s/%s (HTTP %d): %w", repository.ID, name, response.StatusCode, responseError(response))
	}
	var result struct {
		CommitOID string `json:"commitOid"`
	}
	if err := json.NewDecoder(response.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("decode upload response: %w", err)
	}
	if !revisionPattern.MatchString(result.CommitOID) {
		return "", fmt.Errorf("Hub returned an invalid commit ID after upload")
	}
	return result.CommitOID, nil
}
