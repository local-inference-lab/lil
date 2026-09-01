// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the lil project

package configs

import "embed"

// ModelBases contains the shared model-profile inheritance hierarchy.
//
//go:embed models/_bases.yaml
var ModelBases embed.FS
