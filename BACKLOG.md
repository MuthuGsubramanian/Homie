## Innovation Backlog

*Auto-maintained by the PyMasters AI Intelligence Pipeline.*
*Last updated: 2026-07-08*

### Ready to Build (scored >= 8, validated)

- **hexgrad/Kokoro-TTS** (score: 10, source: huggingface, added: 2026-07-08) — Kokoro is a tiny high-quality open TTS â€” a near-perfect fit for Homie's local voice output; PyMasters tutorial on building a local TTS pipeline [link](https://huggingface.co/spaces/hexgrad/Kokoro-TTS)
- **sentence-transformers/all-MiniLM-L6-v2** (score: 9, source: huggingface, added: 2026-07-08) — Tutorial on semantic search with sentence-transformers; also the default lightweight embedding model for Homie's local RAG pipeline [link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **BAAI/bge-small-en-v1.5** (score: 9, source: huggingface, added: 2026-07-08) — Small, fast English embedding model ideal for Homie's on-device RAG; PyMasters tutorial comparing bge-small vs MiniLM for retrieval [link](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **Qwen/Qwen3-0.6B** (score: 9, source: huggingface, added: 2026-07-08) — Tutorial on running tiny LLMs locally; Qwen3-0.6B is a great low-RAM fallback model for Homie on Android and older PCs [link](https://huggingface.co/Qwen/Qwen3-0.6B)
- **Qwen/Qwen3-8B** (score: 9, source: huggingface, added: 2026-07-08) — Tutorial on quantizing and serving an 8B model locally; Qwen3-8B is a prime default chat model for Homie on desktop GPUs [link](https://huggingface.co/Qwen/Qwen3-8B)
- **nomic-ai/nomic-embed-text-v1.5** (score: 9, source: huggingface, added: 2026-07-08) — Tutorial on Matryoshka embeddings and resizable dimensions; nomic-embed runs well locally (GGUF/ONNX) making it a strong Homie RAG embedder [link](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- **cross-encoder/ms-marco-MiniLM-L6-v2** (score: 8, source: huggingface, added: 2026-07-08) — Tutorial on two-stage retrieval (embed then rerank); Homie could use it as a fast local reranker to improve RAG answer quality [link](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- **sentence-transformers/all-mpnet-base-v2** (score: 8, source: huggingface, added: 2026-07-08) — Tutorial benchmark of quality-vs-speed embedding tradeoffs; strong higher-quality embedding option for Homie RAG on capable machines [link](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- **BAAI/bge-m3** (score: 8, source: huggingface, added: 2026-07-08) — Tutorial on hybrid dense+sparse+multi-vector retrieval with BGE-M3; multilingual embedding upgrade for Homie's RAG [link](https://huggingface.co/BAAI/bge-m3)
- **openai/clip-vit-base-patch32** (score: 8, source: huggingface, added: 2026-07-08) — Tutorial on zero-shot image classification with CLIP; could power a local photo search/organizer plugin in Homie [link](https://huggingface.co/openai/clip-vit-base-patch32)
- **BAAI/bge-reranker-v2-m3** (score: 8, source: huggingface, added: 2026-07-08) — Companion content for RAG reranking tutorials; multilingual reranker Homie could ship for higher-precision retrieval [link](https://huggingface.co/BAAI/bge-reranker-v2-m3)

### Prototyping (scored >= 7)

- **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** (score: 7, source: huggingface, added: 2026-07-08) — Tutorial on multilingual semantic search; lets Homie support RAG over non-English documents locally [link](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **google/gemma-4-26B-A4B-it** (score: 7, source: huggingface, added: 2026-07-08) — Tutorial on multimodal (image+text) chat with Gemma 4; the A4B MoE design could make it a viable local vision model for Homie desktops [link](https://huggingface.co/google/gemma-4-26B-A4B-it)
- **mteb/leaderboard** (score: 7, source: huggingface, added: 2026-07-08) — Tutorial on choosing embedding models via MTEB; informs which embedders Homie should bundle for RAG [link](https://huggingface.co/spaces/mteb/leaderboard)

### Evaluating (scored >= 6)

- **BAAI/bge-large-en-v1.5** (score: 6, source: huggingface, added: 2026-07-08) — Embedding quality-tier comparison content; optional high-accuracy embedding profile for Homie users with strong hardware [link](https://huggingface.co/BAAI/bge-large-en-v1.5)

### Discovered (new)

*No items yet.*
