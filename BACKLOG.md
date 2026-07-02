## Innovation Backlog

*Auto-maintained by the PyMasters AI Intelligence Pipeline.*
*Last updated: 2026-07-02*

### Ready to Build (scored >= 8, validated)

- **hexgrad/Kokoro-82M** (score: 10, source: huggingface, added: 2026-07-02) — 82M-param high-quality TTS â€” ideal local voice output engine for Homie; also a fun 'build a talking app' tutorial [link](https://huggingface.co/hexgrad/Kokoro-82M)
- **sentence-transformers/all-MiniLM-L6-v2** (score: 9, source: huggingface, added: 2026-07-02) — Tutorial on building semantic search with the most-used embedding model; also the default lightweight embedder for Homie's local RAG [link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **BAAI/bge-small-en-v1.5** (score: 9, source: huggingface, added: 2026-07-02) — Small, fast English embedding model â€” good tutorial subject and an ideal low-RAM embedding backend for Homie [link](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **Qwen/Qwen3-0.6B** (score: 9, source: huggingface, added: 2026-07-02) — Tutorial on running sub-1B LLMs locally; perfect candidate for Homie's fast on-device model tier (Android especially) [link](https://huggingface.co/Qwen/Qwen3-0.6B)
- **nomic-ai/nomic-embed-text-v1.5** (score: 9, source: huggingface, added: 2026-07-02) — Tutorial on Matryoshka/resizable embeddings; strong ONNX-ready local embedder for Homie RAG [link](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- **Qwen/Qwen3-8B** (score: 9, source: huggingface, added: 2026-07-02) — Tutorial on running/quantizing 8B models locally; a flagship local-LLM option for Homie on capable desktops [link](https://huggingface.co/Qwen/Qwen3-8B)
- **hexgrad/Kokoro-TTS** (score: 9, source: huggingface, added: 2026-07-02) — Live demo of Kokoro TTS â€” validates it as Homie's local voice engine and provides reference implementation code [link](https://huggingface.co/spaces/hexgrad/Kokoro-TTS)
- **cross-encoder/ms-marco-MiniLM-L6-v2** (score: 8, source: huggingface, added: 2026-07-02) — Tutorial on two-stage retrieval (embed + rerank); Homie could use it as a reranker to sharpen RAG answer quality [link](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- **sentence-transformers/all-mpnet-base-v2** (score: 8, source: huggingface, added: 2026-07-02) — Higher-quality embedding option â€” tutorial comparing MiniLM vs mpnet; a 'quality mode' embedder for Homie RAG [link](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- **BAAI/bge-m3** (score: 8, source: huggingface, added: 2026-07-02) — Tutorial on dense+sparse+multi-vector retrieval with BGE-M3; multilingual hybrid retrieval option for Homie [link](https://huggingface.co/BAAI/bge-m3)
- **BAAI/bge-reranker-v2-m3** (score: 8, source: huggingface, added: 2026-07-02) — Multilingual reranker for Homie's retrieval pipeline; pairs with a PyMasters piece on reranking [link](https://huggingface.co/BAAI/bge-reranker-v2-m3)

### Prototyping (scored >= 7)

- **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** (score: 7, source: huggingface, added: 2026-07-02) — Multilingual embeddings tutorial; lets Homie's RAG handle non-English documents on-device [link](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **openai/clip-vit-base-patch32** (score: 7, source: huggingface, added: 2026-07-02) — Tutorial on zero-shot image classification with CLIP; could power Homie's local image search/tagging plugin [link](https://huggingface.co/openai/clip-vit-base-patch32)
- **mteb/leaderboard** (score: 7, source: huggingface, added: 2026-07-02) — MTEB leaderboard is the reference for a 'choosing an embedding model for RAG' guide and for Homie's embedder defaults [link](https://huggingface.co/spaces/mteb/leaderboard)

### Evaluating (scored >= 6)

- **timm/mobilenetv3_small_100.lamb_in1k** (score: 6, source: huggingface, added: 2026-07-02) — Tiny image classifier suited to a Homie vision plugin on low-power devices; minor tutorial value on edge inference [link](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k)
- **BAAI/bge-large-en-v1.5** (score: 6, source: huggingface, added: 2026-07-02) — High-quality embedding benchmark subject; optional 'best quality' embedder for Homie on desktop hardware [link](https://huggingface.co/BAAI/bge-large-en-v1.5)
- **laion/clap-htsat-fused** (score: 6, source: huggingface, added: 2026-07-02) — CLAP audio-text matching could power a Homie plugin for searching/classifying local audio by description [link](https://huggingface.co/laion/clap-htsat-fused)
- **open-llm-leaderboard/open_llm_leaderboard** (score: 6, source: huggingface, added: 2026-07-02) — Source material for a 'how to pick an open LLM' guide; informs Homie's recommended-model list [link](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)

### Discovered (new)

*No items yet.*
