## Innovation Backlog

*Auto-maintained by the PyMasters AI Intelligence Pipeline.*
*Last updated: 2026-07-06*

### Ready to Build (scored >= 8, validated)

- **hexgrad/Kokoro-82M** (score: 10, source: huggingface, added: 2026-07-06) — 82M-param high-quality local TTS â€” ideal for Homie's voice output; also a fun 'build a talking assistant in Python' tutorial [link](https://huggingface.co/hexgrad/Kokoro-82M)
- **sentence-transformers/all-MiniLM-L6-v2** (score: 9, source: huggingface, added: 2026-07-06) — Classic tutorial on sentence embeddings and semantic search; also the default lightweight embedding model for Homie's local RAG pipeline [link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **BAAI/bge-small-en-v1.5** (score: 9, source: huggingface, added: 2026-07-06) — Small, fast English embedding model â€” ideal candidate for Homie's default on-device embeddings; tutorial comparing bge vs MiniLM [link](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **Qwen/Qwen3-0.6B** (score: 9, source: huggingface, added: 2026-07-06) — Tiny 0.6B LLM runs on almost any device â€” perfect Homie fallback model for low-end hardware and a great 'run an LLM locally' beginner tutorial [link](https://huggingface.co/Qwen/Qwen3-0.6B)
- **Qwen/Qwen3-8B** (score: 9, source: huggingface, added: 2026-07-06) — Strong 8B local model â€” flagship option for Homie's local LLM backend and a 'run Qwen3 locally with llama.cpp' tutorial [link](https://huggingface.co/Qwen/Qwen3-8B)
- **hexgrad/Kokoro-TTS** (score: 9, source: huggingface, added: 2026-07-06) — Live demo of Kokoro TTS â€” validates it as Homie's voice engine; link from a local-TTS tutorial [link](https://huggingface.co/spaces/hexgrad/Kokoro-TTS)
- **cross-encoder/ms-marco-MiniLM-L6-v2** (score: 8, source: huggingface, added: 2026-07-06) — Tutorial on two-stage retrieval (embed + rerank); Homie could use it as a fast local reranker to improve RAG answer quality [link](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- **sentence-transformers/all-mpnet-base-v2** (score: 8, source: huggingface, added: 2026-07-06) — Higher-quality embedding option to benchmark in a 'choosing an embedding model' tutorial; Homie quality-tier embedding backend [link](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- **BAAI/bge-m3** (score: 8, source: huggingface, added: 2026-07-06) — Tutorial on dense+sparse+multi-vector hybrid retrieval with bge-m3; multilingual embedding upgrade for Homie RAG [link](https://huggingface.co/BAAI/bge-m3)
- **openai/clip-vit-base-patch32** (score: 8, source: huggingface, added: 2026-07-06) — Tutorial on zero-shot image classification with CLIP; Homie could use it for local photo search/organization plugin [link](https://huggingface.co/openai/clip-vit-base-patch32)
- **nomic-ai/nomic-embed-text-v1.5** (score: 8, source: huggingface, added: 2026-07-06) — Tutorial on Matryoshka embeddings and resizable dimensions; efficient long-context embedding choice for Homie RAG [link](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- **BAAI/bge-reranker-v2-m3** (score: 8, source: huggingface, added: 2026-07-06) — Multilingual reranker to boost Homie RAG precision; pairs with bge-m3 in a two-stage retrieval tutorial [link](https://huggingface.co/BAAI/bge-reranker-v2-m3)

### Prototyping (scored >= 7)

- **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** (score: 7, source: huggingface, added: 2026-07-06) — Tutorial on multilingual semantic similarity; gives Homie multilingual RAG/search support on-device [link](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **mteb/leaderboard** (score: 7, source: huggingface, added: 2026-07-06) — Guide on using the MTEB leaderboard to choose embedding models â€” directly informs Homie's embedding model selection [link](https://huggingface.co/spaces/mteb/leaderboard)

### Evaluating (scored >= 6)

- **BAAI/bge-large-en-v1.5** (score: 6, source: huggingface, added: 2026-07-06) — Quality-tier English embeddings for desktop Homie installs; benchmark entry in an embedding comparison article [link](https://huggingface.co/BAAI/bge-large-en-v1.5)

### Discovered (new)

*No items yet.*
