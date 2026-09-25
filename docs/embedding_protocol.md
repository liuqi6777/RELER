# Embedding protocol

An embedding checkpoint is defined by more than its transformer weights. RELER
stores the following representation settings in `embedding_protocol.json` next to
every saved checkpoint:

- query and document prompt templates;
- tokenizer padding side;
- terminal-token policy (`none`, `pad`, or `eos`);
- pooling method (`last`, `mean`, or `cls`);
- the maximum supported embedding length;
- tokenization protocol version and resolved token IDs;
- FP32 pooling precision.

## Tokenization

With `append_token: pad` or `append_token: eos`, RELER tokenizes content without
automatic special tokens, reserves one position, truncates the content, and then
appends exactly one attended terminal token. This keeps the terminal readout token
present at the maximum sequence length. With `append_token: none`, the tokenizer's
native special-token behavior is preserved.

The same helper is used for query and document batches. Prompt templates accept
`{query}`/`{document}` or the generic `{text}` placeholder. Query templates may
also use `{task_description}`.

## Pooling and precision

Pooling is computed in FP32 before L2 normalization, even when the transformer
forward pass uses BF16 or FP16. Similarity matrices and ranking objectives also
promote embeddings to FP32. This is required for stable ordering when candidate
scores are close.

Padding never contributes to mean or last-token pooling. Candidate padding is
masked before scoring, loss computation, and reward computation.

## Checkpoint use

The standalone MTEB adapter reads `embedding_protocol.json` automatically. Do not
override its prompt, tokenization, or pooling fields when comparing a saved
checkpoint. A model config is used only when evaluating a base model that has no
RELER protocol sidecar.

Protocol compatibility is covered by `tests/test_embedding_protocol.py` and
mixed-precision behavior by `tests/test_score_precision.py`.
