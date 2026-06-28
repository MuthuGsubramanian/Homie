## Innovation Backlog

*Auto-maintained by the PyMasters AI Intelligence Pipeline.*
*Last updated: 2026-06-28*

### Ready to Build (scored >= 8, validated)

- **Qwen/Qwen3-0.6B** (score: 10, source: huggingface, added: 2026-06-28) — Perfect tiny local LLM for Homie's offline chat/voice on modest hardware; also a 'run an LLM locally in Python' tutorial. [link](https://huggingface.co/Qwen/Qwen3-0.6B)
- **hexgrad/Kokoro-82M** (score: 10, source: huggingface, added: 2026-06-28) — Tiny high-quality TTS â€” ideal for Homie's offline voice output; also a 'local text-to-speech in Python' tutorial. [link](https://huggingface.co/hexgrad/Kokoro-82M)
- **sentence-transformers/all-MiniLM-L6-v2** (score: 9, source: huggingface, added: 2026-06-28) — Flagship 'build a semantic search/RAG in Python' tutorial for PyMasters; the default lightweight local embedding model for Homie's RAG. [link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **BAAI/bge-small-en-v1.5** (score: 9, source: huggingface, added: 2026-06-28) — Tutorial comparing BGE vs MiniLM embeddings; small footprint makes it ideal for Homie's on-device RAG index. [link](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **BAAI/bge-m3** (score: 9, source: huggingface, added: 2026-06-28) — Tutorial on multi-functional (dense+sparse+multi-vector) multilingual retrieval; strong all-in-one embedder for Homie's privacy-first RAG. [link](https://huggingface.co/BAAI/bge-m3)
- **nomic-ai/nomic-embed-text-v1.5** (score: 9, source: huggingface, added: 2026-06-28) — Tutorial on long-context embeddings with Matryoshka dimensions; efficient local embedder for Homie's RAG with adjustable vector size. [link](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- **BAAI/bge-reranker-v2-m3** (score: 9, source: huggingface, added: 2026-06-28) — Tutorial on multilingual reranking; Homie reranker to boost local RAG precision across languages. [link](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- **Qwen/Qwen3-4B** (score: 9, source: huggingface, added: 2026-06-28) — Capable mid-size local LLM for Homie's offline assistant/voice; also a 'local 4B model in Python' tutorial. [link](https://huggingface.co/Qwen/Qwen3-4B)
- **hexgrad/Kokoro-TTS** (score: 9, source: huggingface, added: 2026-06-28) — Live demo of the Kokoro TTS that Homie can embed for offline voice; also a hands-on TTS tutorial. [link](https://huggingface.co/spaces/hexgrad/Kokoro-TTS)
- **cross-encoder/ms-marco-MiniLM-L6-v2** (score: 8, source: huggingface, added: 2026-06-28) — Tutorial on two-stage retrieval (bi-encoder + cross-encoder reranking); Homie can rerank RAG candidates for sharper local answers. [link](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- **sentence-transformers/all-mpnet-base-v2** (score: 8, source: huggingface, added: 2026-06-28) — Tutorial on higher-quality embeddings (accuracy vs speed trade-off); a quality-tier embedding option for Homie on capable hardware. [link](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- **BAAI/bge-large-en-v1.5** (score: 8, source: huggingface, added: 2026-06-28) — Tutorial on high-accuracy embeddings; quality-tier RAG embedder for Homie on higher-end devices. [link](https://huggingface.co/BAAI/bge-large-en-v1.5)

### Prototyping (scored >= 7)

- **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** (score: 7, source: huggingface, added: 2026-06-28) — Tutorial on multilingual semantic search; lets Homie support RAG across 50+ languages for non-English users. [link](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **openai/clip-vit-base-patch32** (score: 7, source: huggingface, added: 2026-06-28) — Tutorial on CLIP zero-shot image classification/search; Homie plugin for local photo search by text description. [link](https://huggingface.co/openai/clip-vit-base-patch32)
- **mteb/leaderboard** (score: 7, source: huggingface, added: 2026-06-28) — Reference for an 'how to choose an embedding model' guide; directly informs which embedder Homie bundles for RAG. [link](https://huggingface.co/spaces/mteb/leaderboard)

### Evaluating (scored >= 6)

- **timm/mobilenetv3_small_100.lamb_in1k** (score: 6, source: huggingface, added: 2026-06-28) — Beginner image-classification tutorial with timm; ultra-light model for an on-device Homie vision plugin. [link](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k)
- **laion/clap-htsat-fused** (score: 6, source: huggingface, added: 2026-06-28) — Tutorial on audio embeddings/zero-shot sound classification; Homie plugin for local sound/voice-context detection. [link](https://huggingface.co/laion/clap-htsat-fused)

### Discovered (new)

*No items yet.*
