# Hugging Face model coverage

Model coverage is dynamic. A repository is launchable when it belongs to the
`local-inference-lab` Hugging Face account, its root `lil.yaml` has
`kind: model`, and its `config.json` and safetensors shards are readable at the
repository's head commit. `lil list` is the authoritative inventory and shows
the default tensor-parallel size on every discovered topology.

Repositories with `kind: draft` describe speculative-decoding checkpoints.
They are validated during discovery but are not listed as serving targets. A
serving manifest references a compatible draft repository by Hugging Face ID.

The executable contains no per-model inventory, no checkpoint facts, and no
model revisions. It embeds only the model families in
`configs/models/_bases.yaml`, which describe contracts shared by several
repositories: parsers, kernel selection, and environment a family needs. A
checkpoint name, model version, or one-repository convenience does not belong
in a family.

A manifest contains only what the checkpoint cannot state about itself:

- the family, when one applies;
- the served model name and API-facing serving policy;
- kernel and loader selection that differs from the launcher defaults;
- speculation methods, their depth, and the draft repository;
- launch layers and the conditions under which they apply;
- architectures the checkpoint requires.

Architectures, attention heads, stored weight size, and MTP expert quantization
are read from the repository. Host paths, device IDs, ports, GPU
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
5. MTP expert selection resolves NVFP4 to B12X and MXFP8 or BF16 to Triton.
6. CUDA graph sizes cover all resolved MTP verification batches and the mixed
   batch ladder.
7. A Hub-backed run updates every required repository in each selected rank's
   standard Hugging Face cache before launch.
8. The `fit` default TP on each discovered topology is the size the model is
   actually served at, or an override or `--tp` documents why not.

When a launcher in a vLLM working tree carries policy not represented by an
existing family, the missing behavior belongs in the typed Go builder or a
family. Copying a shell launcher into a large model-specific manifest is
unsupported.
