## Innovation Backlog

*Auto-maintained by the PyMasters AI Intelligence Pipeline.*
*Last updated: 2026-06-17*

### Ready to Build (scored >= 8, validated)

- **sentence-transformers/all-MiniLM-L6-v2** (score: 9, source: huggingface, added: 2026-06-17) — Tutorial on building semantic search with sentence embeddings; Homie's default RAG embedding model (runs locally via ONNX) [link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **BAAI/bge-small-en-v1.5** (score: 9, source: huggingface, added: 2026-06-17) — Compact, high-quality embedding ideal for Homie's privacy-first local RAG; PyMasters guide on choosing embedding models [link](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **BAAI/bge-m3** (score: 9, source: huggingface, added: 2026-06-17) — Homie's multilingual + hybrid (dense/sparse) RAG backbone; tutorial on multi-functional embeddings [link](https://huggingface.co/BAAI/bge-m3)
- **Qwen/Qwen3-0.6B** (score: 9, source: huggingface, added: 2026-06-17) — Tiny local LLM perfect for Homie on low-end devices/Android; PyMasters tutorial on running local LLMs [link](https://huggingface.co/Qwen/Qwen3-0.6B)
- **nomic-ai/nomic-embed-text-v1.5** (score: 9, source: huggingface, added: 2026-06-17) — Long-context local embedding for Homie RAG; tutorial on document-level retrieval [link](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- **hexgrad/Kokoro-82M** (score: 9, source: huggingface, added: 2026-06-17) — Lightweight local TTS to power Homie's voice interaction offline; tutorial on building a local voice assistant [link](https://huggingface.co/hexgrad/Kokoro-82M)
- **Qwen/Qwen3-4B** (score: 9, source: huggingface, added: 2026-06-17) — Capable mid-size local LLM for Homie's on-device chat/reasoning; tutorial on quantizing and serving local models [link](https://huggingface.co/Qwen/Qwen3-4B)
- **cross-encoder/ms-marco-MiniLM-L6-v2** (score: 8, source: huggingface, added: 2026-06-17) — Homie RAG plugin could add this as a reranker stage; PyMasters tutorial on two-stage retrieval (bi-encoder + cross-encoder) [link](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- **sentence-transformers/all-mpnet-base-v2** (score: 8, source: huggingface, added: 2026-06-17) — Strong general-purpose embedding for Homie RAG; PyMasters benchmark walkthrough of embedding quality [link](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- **BAAI/bge-reranker-v2-m3** (score: 8, source: huggingface, added: 2026-06-17) — Homie's multilingual reranker for higher-precision RAG; tutorial on reranking retrieved chunks [link](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- **BAAI/bge-large-en-v1.5** (score: 8, source: huggingface, added: 2026-06-17) — Higher-accuracy embedding option for Homie RAG; PyMasters guide on embedding size vs. quality tradeoffs [link](https://huggingface.co/BAAI/bge-large-en-v1.5)
- **hexgrad/Kokoro-TTS** (score: 8, source: huggingface, added: 2026-06-17) — Live demo of the TTS Homie can embed for offline voice output; PyMasters tutorial on local TTS [link](https://huggingface.co/spaces/hexgrad/Kokoro-TTS)

### Prototyping (scored >= 7)

- **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** (score: 7, source: huggingface, added: 2026-06-17) — Homie could enable multilingual document Q&A; tutorial on cross-lingual semantic search [link](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **colbert-ir/colbertv2.0** (score: 7, source: huggingface, added: 2026-06-17) — Advanced RAG tutorial on late-interaction retrieval; Homie could offer ColBERT-style precise retrieval plugin [link](https://huggingface.co/colbert-ir/colbertv2.0)

### Evaluating (scored >= 6)

- **openai/clip-vit-base-patch32** (score: 6, source: huggingface, added: 2026-06-17) — Homie image-search/RAG-over-photos plugin; tutorial on zero-shot image classification with CLIP [link](https://huggingface.co/openai/clip-vit-base-patch32)
- **laion/clap-htsat-fused** (score: 6, source: huggingface, added: 2026-06-17) — Homie audio plugin for sound/voice classification and audio RAG; niche PyMasters multimodal tutorial [link](https://huggingface.co/laion/clap-htsat-fused)
- **mteb/leaderboard** (score: 6, source: huggingface, added: 2026-06-17) — Use MTEB to pick Homie's embedding model; PyMasters guide on benchmarking embeddings for RAG [link](https://huggingface.co/spaces/mteb/leaderboard)

### Discovered (new)

*No items yet.*
