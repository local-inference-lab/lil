# lil

`lil` is a standalone Go launcher for local and Spark/RDMA vLLM deployments in
the local inference lab. It discovers launchable models from the
`local-inference-lab` Hugging Face account, resolves each repository's
`lil.yaml`, combines it with inherited family policy and a discovered machine
topology, and produces a typed vLLM invocation. Shell launcher scripts, Ray,
Python wrappers, and external queueing layers are not part of the execution path.

## Install

```bash
git clone git@github.com:local-inference-lab/lil.git
cd lil
make check
make install
lil --version
```

The default install path is `~/.local/bin/lil`. Set `PREFIX` to install
elsewhere:

```bash
make install PREFIX=/usr/local
```

The executable embeds shared model-family policy. Per-model manifests come
from Hugging Face at runtime, and topology YAML is generated locally under
`~/.config/lil/topologies`.

## Discover the launch topology

Local topology discovery records the vLLM checkout, CUDA installation, B12X
checkout, GPU memory, compute capability, and every valid local GPU pool:

```bash
lil discover local \
  --repo-root ~/projects/vllm-hh-rebase \
  --b12x-root ~/projects/b12x
```

Spark/RDMA discovery runs from the controller and reaches ranks over SSH. The
controller does not need to be a Spark node:

```bash
lil discover spark \
  --node tachyon \
  --node luxon \
  --repo-root ~/projects/vllm-hh-rebase \
  --b12x-root ~/projects/b12x
```

Repeated `--node` flags define rank order. Discovery validates remote GPU,
RDMA, interface, path, image, and cache compatibility before writing YAML. Use
`--output -` to inspect the result or `--force` to replace an existing topology.

## Run models

List models carrying a valid `lil.yaml` and all discovered topologies:

```bash
lil list
```

Pretty-print the exact environment and vLLM command without downloading model
weights or executing it:

```bash
lil render GLM-5.3-NVFP4 --config local --tp 8
```

Run read-only preflight validation:

```bash
lil check GLM-5.3-NVFP4 --config local --tp 8
```

Update the standard Hugging Face cache with visible progress and launch:

```bash
lil run GLM-5.3-NVFP4 --config local --tp 8
```

For Spark/RDMA, `lil run` updates the cache on every selected rank before
starting workers and then the head:

```bash
lil run GLM-5.3-Flash-NVFP4-Spark \
  --config spark \
  --tp 2 \
  --detach
```

Spark preflight also hashes the importable `vllm/` and `b12x/` package trees on
the controller and every selected rank. A mismatch fails closed before model
download or container startup. `lil run --sync-code` explicitly reconciles
those two remote package directories with `rsync --delete`, then reruns the
same preflight; without that flag, `lil` never changes remote source trees.

`--config local` and `--config spark` select the unique discovered topology of
that kind. A topology name or YAML path selects an exact configuration.

Use an explicit checkpoint directory without changing the model profile:

```bash
lil run GLM-5.3-NVFP4 \
  --config local \
  --model-path /data/models/GLM-5.3-NVFP4 \
  --tp 8
```

The local checkpoint is inspected to verify its architecture, stored size, and
MTP expert quantization. A local `lil.yaml` is not used; model policy still
comes from the repository's newest manifest. For a Spark topology,
`--sync-model` incrementally copies an explicit `--model-path` to every rank.

Arguments after `--` are forwarded verbatim. Launcher-managed vLLM flags,
including `--revision`, are rejected there so the command cannot contain
contradictory policy:

```bash
lil render GLM-5.3-NVFP4 --tp 8 -- --disable-log-requests
```

## Latest-model and cache contract

`render`, `check`, `run`, and cluster commands resolve the requested repository
through the Hugging Face API on every invocation. `lil` retrieves `lil.yaml`
for the repository's head commit; bytes already cached for that immutable
commit may be reused. Network resolution errors fail the command rather than
silently using stale model policy.

For DFlash, the draft repository head is resolved independently, its newest
manifest is validated as `kind: draft`, and its compatibility list must contain
the serving model.

`lil` never emits a model or draft `--revision`. Before a Hub-backed launch,
`lil run` invokes `hf download OWNER/REPOSITORY` without a revision. The HF CLI
checks repository head, downloads missing or changed files into the standard
cache, and streams its progress directly to the terminal. DFlash launches
update both the serving model and selected draft repositories. An explicit
`--model-path` skips the serving-model download but still updates any Hub-backed
draft.

For Spark/RDMA, `lil` updates each rank's cache through SSH. It prefers the `hf`
executable beside the topology's runtime Python, then checks `PATH`, with
`HF_HOME` set to the configured cache mount source. If neither exists, it tries
the configured Docker image's `hf` executable with that cache mounted. If no
`hf` executable is available, `lil` prints a warning and continues because vLLM
can still download during startup; that fallback may not expose useful
progress. A present `hf` command returning an authentication, network, or
filesystem error fails the run.

`list` may show a cached catalog with an explicit warning when Hugging Face
discovery is temporarily unavailable. Launch commands do not use that fallback.

## Model manifests and inheritance

Every launchable repository in `local-inference-lab` owns one `lil.yaml` at its
root. Identity is derived from the repository; manifests do not repeat a model
path or contain revisions. A typical serving manifest is deliberately small:

```yaml
schema_version: 1
kind: model
extends: glm-5.3
description: GLM-5.3 with NVFP4 routed experts and a BF16 MTP expert layer
weight_bytes: 464823066832
launch:
  local:
    default_tp_size: 8
  spark_rdma:
    default_tp_size: all
quantization: null
mtp_moe_quantization: bf16
```

The embedded `_bases.yaml` holds cross-model and family policy such as parsers,
load format, multimodal flags, speculative defaults, and kernel selection.
Remote manifests extend those semantic bases and contain only checkpoint facts
or genuine per-model differences. Mappings merge recursively; scalars and
lists replace inherited values. Unknown fields, duplicate definitions, missing
parents, inheritance cycles, and invalid resolved types fail closed.

Draft repositories use a separate schema and do not appear as launchable
models:

```yaml
schema_version: 1
kind: draft
description: MXFP8 DFlash speculative-decoding draft for GLM-5.3 Flash
method: dflash
quantization: mxfp8
compatible_models:
  - local-inference-lab/GLM-5.3-Flash-NVFP4
```

`--models-config` accepts an explicit repository-layout manifest directory or
legacy combined YAML for development and tests. It is never the default model
source.

## Policy ownership

| Concern | Owner |
| --- | --- |
| Checkpoint identity, architecture, stored size, quantization, parsers, multimodal behavior, and speculation | repository `lil.yaml` plus inherited family base |
| GPU inventory, CUDA path, memory-utilization ceiling, API bind defaults, local pools, SSH/RDMA layout, image, and cache mounts | discovered topology YAML |
| TP, local checkpoint, bind overrides, KV dtype, scheduler limits, capacity, profiling, and speculation overrides | typed `lil` CLI |
| Environment construction, kernel selection, graph capture sizes, local PCIe policy, RDMA policy, and native commands | Go launcher |
| Loaded checkpoint truth, runtime imports, GPU state, paths, ports, HCAs, and vLLM argument compatibility | preflight checks |

Every model can use every topology-supported TP size that divides its attention
heads. Local discovery builds TP 1-N device pools; Spark selects the first N
one-GPU ranks. `--kv-cache-dtype` defaults to `fp8`, `--max-model-len` defaults
to `auto`, and scheduler concurrency defaults to eight rather than one. The
topology owns `--gpu-memory-utilization`, with `0.95` written by discovery unless
overridden.

Multimodal arguments are emitted only by multimodal family profiles. MTP expert
quantization chooses its backend: NVFP4 uses B12X; MXFP8 and BF16 use Triton. An
explicit local checkpoint is inspected and must agree with the manifest hint.

CUDA graph capture sizes are calculated from resolved scheduler and speculation
settings. The set contains mixed batch sizes and every uniform decode batch
through `max_num_seqs`; with MTP depth K, uniform verification shapes are
sequence count multiplied by K+1. This includes shapes such as 12, 20, and 28
for K=3 instead of relying on a static table.

## Capacity

`weight_bytes` lets `lil` estimate stored weights, post-TP-sharding device
weights per rank, mapped-host PLE bytes, runtime reserve, and an advisory safe
KV budget before downloading a checkpoint. An explicit local checkpoint uses
its safetensors index instead. JSON rendering exposes every estimate term:

```bash
lil render GLM-5.3-NVFP4 --tp 8 --format json
```

By default, `lil` emits the topology's `--gpu-memory-utilization` and leaves
`--kv-cache-memory-bytes` absent so vLLM profiles the actual model, activation
peak, and graph footprint. `--kv-cache-memory-bytes` requests a fixed allocation;
`auto` restores runtime profiling.

## Cluster lifecycle

Detached Spark launches have deterministic container names:

```bash
lil cluster status GLM-5.3-Flash-NVFP4-Spark --config spark --tp 2
lil cluster logs GLM-5.3-Flash-NVFP4-Spark --config spark --tp 2 --follow
lil cluster wait GLM-5.3-Flash-NVFP4-Spark --config spark --tp 2 --timeout 30m
lil cluster profile-start GLM-5.3-Flash-NVFP4-Spark --config spark --tp 2
lil cluster profile-stop GLM-5.3-Flash-NVFP4-Spark --config spark --tp 2
lil cluster stop GLM-5.3-Flash-NVFP4-Spark --config spark --tp 2
```

The controller performs lifecycle and profiler requests over SSH. The head API
does not need to be exposed to the controller network.
