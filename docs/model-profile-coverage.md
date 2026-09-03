# Hugging Face model coverage

Model coverage is dynamic. A model is launchable when the catalog repository
`local-inference-lab/lil-catalog` holds a `<model>/lil.yaml` entry of
`kind: model` and the weight repository the entry names has readable
`config.json` and safetensors shards at the resolved commit. `lil list` is
the authoritative inventory and shows the default tensor-parallel size on
every discovered topology.

Entries of `kind: draft` describe speculative-decoding checkpoints. They are
validated when the catalog loads but are not listed as serving targets. A
serving entry names its draft repository, and the draft entry lists the
serving repositories it is compatible with.

The executable contains no per-model inventory, no checkpoint facts, and no
model revisions. It embeds only the model families in
`configs/models/_bases.yaml`, which describe contracts shared by several
entries: parsers, kernel selection, and environment a family needs. A
checkpoint name, model version, or one-entry convenience does not belong in a
family.

An entry contains only what the checkpoint cannot state about itself:

- the weight repository and, when qualified or required, its pinned commit;
- the family, when one applies;
- the served model name and API-facing serving policy;
- kernel and loader selection that differs from the launcher defaults;
- speculation methods, their depth, and the draft repository;
- launch layers and the conditions under which they apply;
- architectures the checkpoint requires.

Architectures, attention heads, stored weight size, and MTP expert quantization
are read from the weight repository. Host paths, device IDs, ports, GPU
memory-utilization defaults, CUDA locations, container images, RDMA
interfaces, cache locations, host tuning environment, and repository revisions
belong to topology YAML or the launcher, never to a manifest.

## Qualification

A family is implemented when the Go launcher can represent its environment and
vLLM arguments without shell evaluation or untyped argument blobs. A manifest
is qualified when:

1. Its `serving` and `kernels` sections agree with `config.json` at the
   resolved commit, and any `moe_quantization` assertion matches the
   checkpoint's quantization metadata.
2. Every topology-supported TP size that divides its attention heads renders.
3. Generated arguments pass the target vLLM parser, on the controller for a
   local launch and inside the launch image for a Spark launch.
4. Multimodal flags appear only for multimodal architectures.
5. MTP expert selection resolves NVFP4, MXFP4, and BF16 to B12X and MXFP8 to
   Humming, or to the backend the entry declares.
6. CUDA graph sizes cover all resolved MTP verification batches and the mixed
   batch ladder.
7. A Hub-backed run updates every required repository in each selected rank's
   standard Hugging Face cache before launch.
8. The `fit` default TP on each discovered topology is the size the model is
   actually served at, or an override or `--tp` documents why not.

When a launcher in a vLLM working tree carries policy not represented by an
existing family, the missing behavior belongs in the typed Go builder or a
family. Copying a shell launcher into a large model-specific entry is
unsupported.

## Reference launchers

Each entry's defaults follow a reference the lab maintains: the launch
scripts at the root of the vLLM working tree (`serve-glm53.sh`,
`serve-glm53-flash-nvfp4.sh`, `serve-qwen38-flash-next-nvfp4.sh` and its TP1
variant, `serve-ds4-flash.sh`) and the qualified deployment pages in the
`rtx6kpro` repository. Capacity values are defaults for the operator to
override; the ones that encode memory fit, such as the GLM-5.3 Flash TP2
KV allocation and the DSpark sequence count, are kept under the matching
override condition.

Two settings deliberately differ from a reference and are recorded here:

- GLM-5.3 Flash serves MTP at depth 5 as its script does, while the
  `rtx6kpro` R17 page qualifies depth 3.
- The Qwen3.8 Flash Next entry enables async scheduling and full-graph
  compilation, which its scripts do not, and uses the TP2 script's BF16 KV
  cache at every TP where the TP1 script uses fp8.

MXFP8 MTP experts run on Humming, as the GLM-5.3 Flash script selects. The
Triton MXFP8 expert kernel refuses SM120 at worker start, so it is not an
alternative on this hardware.

## DeepSeek V4 Flash

The `deepseek-v4` family reproduces the SM120 PCIe policy of the
`serve-ds4-flash.sh` launcher in the vLLM working tree: the `deepseek_v4`
tokenizer and parsers, thinking enabled in the chat template, full-graph
compilation, FlashInfer autotuning, the B12X attention backend, and the
MegaMoE and multi-stream GEMM environment. Three catalog entries use it.

- `DeepSeek-V4-Flash-0731` is pinned to the release commit the shell launcher
  and the `rtx6kpro` r21 page pin. The DSpark draft head inside the checkpoint
  is the default speculator at the qualified fixed depth 5, with eight
  sequences and a 48-row graph envelope; `--speculator none` switches to the
  32-sequence target-only profile. Standard MTP is not offered on this
  checkpoint. Status: implemented, preflight-qualified against the installed
  parser, not yet served through `lil`.
- `DeepSeek-V4-Flash` is the standard checkpoint with its MTP head at depth
  2, the depth the lab's decode sweeps found best, at 64 sequences and 0.91
  utilization. Status: implemented, preflight-qualified.
- `DeepSeek-V4-Flash-Vision-Exp` carries the 0731 policy pinned to its first
  published commit. This vLLM tree registers no vision tower for the V4
  architecture, so the entry serves the text model only and its vision weights
  are outside the launcher's contract. Status: research-only.

The entries keep `max_model_len: auto` so the runtime profile sizes the KV
cache; the references fix 131072 or 1048576 tokens. The r21 page also serves
FP8 dense projections through DeepGEMM by omitting the B12X linear backend;
the entries follow the working tree's script and keep B12X.
