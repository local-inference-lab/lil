# lil

`lil` is a standalone Go launcher for local and Spark/RDMA vLLM deployments in
the local inference lab. It reads launch manifests from the catalog
repository `local-inference-lab/lil-catalog`, reads each model's checkpoint
metadata from the repository that holds its weights, combines them with a
discovered machine topology, and produces a typed vLLM invocation. Shell
launcher scripts, Ray, Python wrappers, and external queueing layers are not
part of the execution path.

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

The executable embeds the model families in `configs/models/_bases.yaml`.
Catalog entries and checkpoint facts come from Hugging Face at runtime, and
topology YAML is generated locally under `~/.config/lil/topologies`.

## Discover the launch topology

A topology records hardware facts and host tuning. Local discovery records the
vLLM checkout, CUDA installation, B12X checkout, GPU memory, compute
capability, every valid local GPU pool, and the environment variables a launch
on this host exports:

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

Every topology carries two policy fields that discovery fills with defaults:

- `default_tp` selects how a launch picks its tensor-parallel size when `--tp`
  is absent. `fit` chooses the smallest size whose sharded weights leave KV
  headroom on one rank; `all` uses every rank. Local discovery writes `fit`,
  Spark discovery writes `all`, and `--default-tp` overrides either.
- `environment` holds the host tuning exported by every launch on that
  topology: allocator settings, thread counts, NCCL protocol and PCIe
  all-reduce policy on local hosts, loader buffer sizes and RDMA routing on
  Spark nodes. The block is required, so a topology written before it existed
  must be rediscovered or edited. Values the launcher derives from topology
  facts, such as `CUDA_HOME`, `CUTE_DSL_ARCH`, `CUDA_VISIBLE_DEVICES`,
  `LD_LIBRARY_PATH`, and the per-rank interface variables, cannot appear in
  it.

Discovery also records `nvrtc_library_dir`, the venv directory holding the
CUDA 13 NVRTC builtins, when the runtime Python ships them. Launches that
select the Humming MoE backend put it on `LD_LIBRARY_PATH` and fail closed
when it is absent.

## Run models

List the catalog's models, their stored size, and the default tensor-parallel
size on every discovered topology:

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

For Spark/RDMA, `lil run` updates the cache on every selected rank, repeats the
per-node host probes so that a conflict that appeared during the download is
caught, then starts workers and the head:

```bash
lil run GLM-5.3-Flash-NVFP4-Spark \
  --config spark \
  --tp 2 \
  --detach
```

`--config local` and `--config spark` select the unique discovered topology of
that kind. A topology name or YAML path selects an exact configuration.

Use an explicit checkpoint directory without changing the model profile:

```bash
lil run GLM-5.3-NVFP4 \
  --config local \
  --model-path /data/models/GLM-5.3-NVFP4 \
  --tp 8
```

The local checkpoint must declare the same architectures and attention heads
as the repository at its resolved commit, and its MTP expert quantization must
agree with any assertion in the manifest. Its safetensors index sizes the
memory estimate. For a Spark topology, `--sync-model` incrementally copies an
explicit `--model-path` to every rank.

Arguments after `--` are forwarded verbatim. Launcher-managed vLLM flags,
including `--revision`, are rejected there so the command cannot contain
contradictory policy:

```bash
lil render GLM-5.3-NVFP4 --tp 8 -- --disable-log-requests
```

`--env NAME=VALUE` overrides a topology or manifest tuning value for one
launch. Variables the launcher derives, and the offline switches it clears,
cannot be overridden.

## Latest-model and cache contract

Every command resolves the catalog repository's head commit through the
Hugging Face API on each invocation and reads the requested entry there. It
then resolves the entry's weight repository, at the pinned revision when the
entry has one and at head otherwise, and reads `config.json` and the size of
every safetensors shard at that commit. Bytes already cached for an immutable
commit are reused. Network resolution errors fail the command rather than
silently using stale model policy.

For DFlash, the draft's catalog entry must name the draft repository and list
the serving model as compatible.

An unpinned entry never emits `--revision`, and `lil run` invokes
`hf download OWNER/REPOSITORY` without one: the HF CLI checks repository head,
downloads missing or changed files into the standard cache, and streams its
progress to the terminal. A pinned entry emits `--revision` and downloads that
commit. DFlash launches update both the serving model and the draft
repository. An explicit `--model-path` skips the serving-model download but
still updates any Hub-backed draft.

For Spark/RDMA, `lil` updates each rank's cache through SSH. It prefers the `hf`
executable beside the topology's runtime Python, then checks `PATH`, with
`HF_HOME` set to the configured cache mount source. If neither exists, it tries
the configured Docker image's `hf` executable with that cache mounted. If no
`hf` executable is available, `lil` prints a warning and continues because vLLM
can still download during startup; that fallback may not expose useful
progress. A present `hf` command returning an authentication, network, or
filesystem error fails the run.

## The catalog

Every launchable model is one entry in `local-inference-lab/lil-catalog`: a
directory named for the model holding a `lil.yaml`. The directory name is the
launch name. The manifest names the repository that holds the weights and
states only what the checkpoint cannot state about itself. Architectures,
attention heads, stored weight size, and the quantization of the MTP expert
layer are read from `config.json` and the shard sizes at the resolved commit.

```yaml
schema_version: 1
kind: model
model: local-inference-lab/GLM-5.3-NVFP4
family: glm
description: GLM-5.3 with NVFP4 routed experts and an unquantized BF16 MTP expert layer
serving:
  served_model_name: GLM-5.3
  trust_remote_code: true
  async_scheduling: true
  generation_config: vllm
  long_prefill_token_threshold: 2048
speculators:
  default: mtp
  mtp:
    tokens: 3
    moe_quantization: bf16
    attention: B12X
    draft_sample_method: probabilistic
    model: target
compilation:
  cudagraph_mode: FULL_AND_PIECEWISE
  custom_ops:
    - all
environment:
  CUDA_DEVICE_MAX_CONNECTIONS: "32"
```

The sections:

- `model` names the weight repository in owner/name form; it may belong to
  any account. `revision` optionally pins a commit. A repository outside
  `local-inference-lab` that enables `trust_remote_code` must be pinned,
  because a third-party head is not a trusted input. A pinned entry emits
  `--revision`, downloads that commit, reads its checkpoint facts there, and
  carries the pin into a speculative config whose draft lives in the target
  checkpoint.
- `family` names one entry in the embedded families file. Family mappings
  merge under the manifest; a scalar, list, or explicit `null` in the manifest
  replaces the family value.
- `serving` holds the API-facing policy: served name, remote code, tokenizer
  mode, parsers, tool choice, generation config, default chat-template
  arguments, scheduler switches including the prefill schedule interval,
  prefix-cache retention, usage and request-ID reporting, and multimodal
  limits. `auto_tool_choice` defaults to true when a
  tool parser is set. Prefix caching and chunked prefill default to on. A
  manifest without parsers or a tool parser emits none of those flags.
- `kernels` holds dtype, KV cache dtype, quantization, attention, linear, MoE,
  and GDN decode backends, block size, Mamba cache mode, FlashInfer autotuning,
  and the weight loader. Defaults are bfloat16 weights, an fp8 KV cache, the
  B12X linear and MoE backends, and the instanttensor loader.
  `--kv-cache-dtype` overrides the KV cache dtype for one launch.
- `speculators` names a `default` of `mtp`, `dflash`, `dspark`, or `none`
  (the default) and a section per method. `mtp.moe_quantization` is an
  assertion checked against the checkpoint; it is required only when the
  checkpoint metadata cannot decide, and `mtp.moe_backend` overrides the
  backend the quantization implies. NVFP4, MXFP4, and BF16 experts use the
  B12X backend, MXFP8 experts use Humming, and block-FP8 experts need an
  explicit backend. A Humming launch adds the topology's discovered NVRTC
  library directory, which the Humming kernels load at runtime. A DSpark
  draft always ships inside the target checkpoint.
- `capacity`, `compilation`, and `environment` are the base launch layer.
  Capacity defaults are `max_model_len: auto`, eight sequences, and 4096
  batched tokens; `capacity.gpu_memory_utilization` replaces the topology's
  default for this model. Every capacity value is a default that the matching
  command-line flag overrides. A compilation mapping enables
  `--compilation-config` with computed CUDA graph capture sizes.
- `overrides` is an ordered list of layers applied when every condition in
  `when` matches the launch. Conditions are `kind` (`local` or `spark_rdma`),
  `arch` (the topology's CuTe DSL architecture, such as `sm_121a`), `tp`, and
  `speculator` (the method that actually runs, with a zero-depth launch
  counting as `none`). Later overrides win.
- `requires.arch` lists the architectures a checkpoint may run on. A launch on
  any other topology fails, and `lil list` marks the topology.

Draft checkpoints are entries of `kind: draft`. They are not listed as
serving models; a serving entry's `speculators.dflash.model` must match a
draft entry whose compatibility list contains the serving model's repository:

```yaml
schema_version: 1
kind: draft
model: local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8
description: MXFP8 DFlash speculative-decoding draft for GLM-5.3 Flash
method: dflash
quantization: mxfp8
compatible_models:
  - local-inference-lab/GLM-5.3-Flash-NVFP4
```

`--models-config DIRECTORY` reads a local catalog laid out the same way, with
each model entry's `config.json` and `model.safetensors.index.json` beside its
manifest, for development and tests. The fixtures under
`internal/launcher/testdata/model-manifests` are the reference copies of the
published catalog. Unknown keys, missing families, inheritance cycles, and
invalid types fail closed.

## Policy ownership

| Concern | Owner |
| --- | --- |
| Architectures, attention heads, stored size, MTP expert quantization | weight repository `config.json` and shard sizes at the resolved commit |
| Weight repository, pinned revision, serving identity, parsers, kernel selection, speculation, capacity and compilation layers, conditional overrides | catalog entry plus its family |
| GPU inventory, CUDA path, memory-utilization ceiling, default TP policy, host tuning environment, API bind defaults, local pools, SSH/RDMA layout, image, and cache mounts | discovered topology YAML |
| TP, local checkpoint, bind overrides, KV dtype, scheduler limits, capacity, profiling, speculation overrides, and one-off environment overrides | typed `lil` CLI |
| Derived environment, default TP fit, graph capture sizes, native commands | Go launcher |
| Loaded checkpoint truth, runtime imports, GPU state, paths, ports, HCAs, and vLLM argument compatibility | preflight checks |

A launch may use any tensor-parallel size the topology can host that divides
the checkpoint's attention heads. Local launches take the first device pool
large enough; Spark launches take the first N one-GPU ranks.
The topology owns the default `--gpu-memory-utilization`, with `0.95` written
by discovery; a catalog entry may replace it and the command line overrides
both.

CUDA graph capture sizes are calculated from resolved scheduler and speculation
settings. The set contains mixed batch sizes and every uniform decode batch
through `max_num_seqs`; with MTP or DSpark depth K, uniform verification
shapes are sequence count multiplied by K+1. The mixed ladder extends to twice
the uniform maximum, except for DSpark, whose verifier never exceeds one
sampled token plus its drafts per request.

## Capacity

The stored weight size lets `lil` estimate post-sharding device weights per
rank, mapped-host PLE bytes for an explicit checkpoint, a runtime reserve, and
an advisory safe KV budget before downloading a checkpoint. The same
arithmetic picks the default tensor-parallel size under the `fit` policy. JSON
rendering exposes every term, the supported TP sizes, the checkpoint facts and
their source, and the manifest commit:

```bash
lil render GLM-5.3-NVFP4 --tp 8 --format json
```

By default, `lil` emits the topology's `--gpu-memory-utilization` and leaves
`--kv-cache-memory-bytes` absent so vLLM profiles the actual model, activation
peak, and graph footprint. `--kv-cache-memory-bytes` requests a fixed allocation;
`auto` restores runtime profiling.

## Spark/RDMA lifecycle

Rank containers are named deterministically from the topology prefix, model,
and TP, and they are kept after exit. A crashed rank leaves its logs and exit
code behind; the next launch removes an exited container of the same name and
refuses a running one. Preflight parses each rank's arguments inside the launch
image with the launch mounts and environment, so the interpreter and imports
validated are the ones the container runs. Spark preflight also hashes the
importable `vllm/` and `b12x/` Python trees on the controller and every
selected rank; native extensions are not compared.

Without `--detach`, the controller starts workers, then the head, streams the
head's logs, and polls every rank. It tears the cluster down only when it
observes a rank exit or is interrupted. A worker exit stops the head and prints
the worker's last log lines; a head exit returns its exit code. Losing contact
with the ranks never stops a running cluster: the controller reports the
outage, keeps trying for ten minutes, then gives up and leaves the containers
running for `lil cluster` to manage. Ctrl-C stops and removes every rank; a
second Ctrl-C exits immediately.

With `--detach`, the controller confirms every rank is still running a few
seconds after start and returns.

```bash
lil cluster status GLM-5.3-Flash-NVFP4-Spark --config spark --tp 2
lil cluster logs GLM-5.3-Flash-NVFP4-Spark --config spark --tp 2 --follow
lil cluster logs GLM-5.3-Flash-NVFP4-Spark --config spark --tp 2 --rank 1
lil cluster wait GLM-5.3-Flash-NVFP4-Spark --config spark --tp 2 --timeout 30m
lil cluster profile-start GLM-5.3-Flash-NVFP4-Spark --config spark --tp 2
lil cluster profile-stop GLM-5.3-Flash-NVFP4-Spark --config spark --tp 2
lil cluster stop GLM-5.3-Flash-NVFP4-Spark --config spark --tp 2
```

`cluster stop` stops and removes every rank. The controller performs lifecycle
and profiler requests over SSH with keepalives. The head API does not need to
be exposed to the controller network.

`lil run --sync-code` reconciles the two remote Python package directories
with `rsync --delete`, then reruns preflight; without that flag, `lil` never
changes remote source trees.
