## Innovation Backlog

*Auto-maintained by the PyMasters AI Intelligence Pipeline.*
*Last updated: 2026-07-13*

### Ready to Build (scored >= 8, validated)

- **sentence-transformers/all-MiniLM-L6-v2** (score: 9, source: huggingface, added: 2026-07-13) — Canonical tutorial on sentence embeddings and semantic search; also the go-to lightweight local embedding model for Homie's RAG pipeline [link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Qwen/Qwen3-0.6B** (score: 9, source: huggingface, added: 2026-07-13) — Tutorial on running a sub-1B LLM locally; ideal ultra-light local model for Homie on low-end hardware and Android [link](https://huggingface.co/Qwen/Qwen3-0.6B)
- **Qwen/Qwen3-8B** (score: 9, source: huggingface, added: 2026-07-13) — Tutorial on running/quantizing an 8B model locally; strong default local LLM for Homie on desktop-class hardware [link](https://huggingface.co/Qwen/Qwen3-8B)
- **hexgrad/Kokoro-TTS** (score: 9, source: huggingface, added: 2026-07-13) — Kokoro is a top lightweight open TTS â€” prime candidate for Homie's local voice output; also a solid PyMasters tutorial on local TTS [link](https://huggingface.co/spaces/hexgrad/Kokoro-TTS)
- **cross-encoder/ms-marco-MiniLM-L6-v2** (score: 8, source: huggingface, added: 2026-07-13) — Tutorial on two-stage retrieval (retrieve + rerank); Homie could use it to rerank RAG results locally for better answer quality [link](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- **BAAI/bge-small-en-v1.5** (score: 8, source: huggingface, added: 2026-07-13) — Small, fast English embedding model â€” tutorial on choosing embedding models; strong candidate for Homie's on-device RAG [link](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** (score: 8, source: huggingface, added: 2026-07-13) — Multilingual embeddings tutorial; enables Homie RAG and semantic search for non-English users on-device [link](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **sentence-transformers/all-mpnet-base-v2** (score: 8, source: huggingface, added: 2026-07-13) — Best-quality general sentence embedding tutorial (vs MiniLM tradeoffs); solid default for Homie document search where compute allows [link](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- **openai/clip-vit-base-patch32** (score: 8, source: huggingface, added: 2026-07-13) — Tutorial on CLIP and zero-shot image classification; Homie could use it for local semantic photo search [link](https://huggingface.co/openai/clip-vit-base-patch32)
- **nomic-ai/nomic-embed-text-v1.5** (score: 8, source: huggingface, added: 2026-07-13) — Tutorial on Matryoshka/resizable embeddings; ONNX support makes it practical for Homie's cross-platform local RAG [link](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)

### Prototyping (scored >= 7)

- **BAAI/bge-m3** (score: 7, source: huggingface, added: 2026-07-13) — Tutorial on hybrid dense+sparse+multilingual retrieval with BGE-M3; Homie could offer it as a higher-quality multilingual RAG option [link](https://huggingface.co/BAAI/bge-m3)
- **BAAI/bge-reranker-v2-m3** (score: 7, source: huggingface, added: 2026-07-13) — Multilingual reranker for RAG quality â€” tutorial on reranking; Homie could add it as an optional RAG quality booster [link](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- **mteb/leaderboard** (score: 7, source: huggingface, added: 2026-07-13) — Guide on using the MTEB leaderboard to choose embedding models â€” directly informs Homie's embedding model selection [link](https://huggingface.co/spaces/mteb/leaderboard)

### Evaluating (scored >= 6)

*No items yet.*

### Discovered (new)

*No items yet.*
