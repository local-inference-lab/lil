# Hugging Face model coverage

Model coverage is dynamic. A repository is launchable when it belongs to the
`local-inference-lab` Hugging Face account and its root `lil.yaml` has
`kind: model`. `lil list` is the authoritative inventory.

Repositories with `kind: draft` describe speculative-decoding checkpoints.
They are validated during discovery but are not listed as serving targets. A
serving manifest references a compatible draft repository by Hugging Face ID.

The executable contains no per-model inventory and no model revisions. It
embeds only the inheritance bases that define common launcher and family
contracts. Each repository manifest should therefore contain only facts that
differ from its family base:

An embedded base must describe a semantic contract shared by multiple model
repositories. A checkpoint name, model version, or one-repository convenience
alias belongs in that repository's `lil.yaml`, not in the executable.

- semantic family in `extends`;
- checkpoint description and exact indexed `weight_bytes`;
- architecture or backend overrides that differ from the family;
- target and MTP quantization facts needed for kernel selection;
- genuine local or Spark/RDMA policy differences;
- compatible speculative draft repository IDs.

Host paths, device IDs, ports, GPU memory-utilization defaults, CUDA locations,
container images, RDMA interfaces, cache locations, and repository revisions do
not belong in model manifests.

## Qualification

A new family contract is implemented when the Go launcher can represent its
environment and vLLM arguments without shell evaluation or untyped argument
blobs. A model manifest is qualified when:

1. Its architecture and quantization facts match `config.json` and the
   safetensors index.
2. Every topology-supported TP size that divides its attention heads renders.
3. Generated arguments pass the target vLLM parser.
4. Multimodal flags appear only for multimodal architectures.
5. MTP expert selection resolves NVFP4 to B12X and MXFP8 or BF16 to Triton.
6. CUDA graph sizes cover all resolved MTP verification batches and the mixed
   batch ladder.
7. A Hub-backed run updates every required repository in each selected rank's
   standard Hugging Face cache before launch.

When a launcher in a vLLM working tree carries policy not represented by an
existing family base, the missing behavior belongs in the typed Go builder or a
semantic family base. Copying the shell launcher into a large model-specific
manifest is unsupported.
