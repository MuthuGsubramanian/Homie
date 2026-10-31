## Innovation Backlog

*Auto-maintained by the PyMasters AI Intelligence Pipeline.*
*Last updated: 2026-08-06*

### Ready to Build (scored >= 8, validated)

- **hexgrad/Kokoro-TTS** (score: 10, source: huggingface, added: 2026-08-06) — Best-in-class lightweight local TTS â€” Homie's voice output (plugin already built); plus an 82M-param TTS tutorial [link](https://huggingface.co/spaces/hexgrad/Kokoro-TTS)
- **sentence-transformers/all-MiniLM-L6-v2** (score: 9, source: huggingface, added: 2026-08-06) — Core embedding model for tutorials on semantic search/RAG; already powers Homie-style local pipelines â€” flagship lesson plus Homie's default embedder [link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Qwen/Qwen3-0.6B** (score: 9, source: huggingface, added: 2026-08-06) — Sub-1B local LLM â€” 'run an LLM on your laptop' lesson; ideal Homie fallback model for low-end devices [link](https://huggingface.co/Qwen/Qwen3-0.6B)
- **BAAI/bge-reranker-v2-m3** (score: 9, source: huggingface, added: 2026-08-06) — Drop-in reranker for Homie's RAG pipeline (plugin already drafted); pairs with a reranking-explained lesson [link](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- **Qwen/Qwen3-8B** (score: 9, source: huggingface, added: 2026-08-06) — Capable 8B local model â€” quantization/GGUF lesson; Homie's recommended mid-tier local LLM [link](https://huggingface.co/Qwen/Qwen3-8B)
- **cross-encoder/ms-marco-MiniLM-L6-v2** (score: 8, source: huggingface, added: 2026-08-06) — Tutorial on two-stage retrieval (bi-encoder + cross-encoder); Homie could use it to rerank RAG results locally [link](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- **BAAI/bge-small-en-v1.5** (score: 8, source: huggingface, added: 2026-08-06) — Compact English embedder â€” benchmark-vs-MiniLM tutorial; strong candidate for Homie's low-RAM RAG profile [link](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **BAAI/bge-m3** (score: 8, source: huggingface, added: 2026-08-06) — Tutorial on hybrid dense+sparse+multi-vector retrieval; Homie multilingual RAG upgrade path [link](https://huggingface.co/BAAI/bge-m3)
- **sentence-transformers/all-mpnet-base-v2** (score: 8, source: huggingface, added: 2026-08-06) — Quality-vs-speed embedder comparison lesson (mpnet vs MiniLM); Homie 'high quality' embedding tier [link](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- **openai/clip-vit-base-patch32** (score: 8, source: huggingface, added: 2026-08-06) — Zero-shot image classification tutorial; powers Homie's local photo search plugin (already prototyped) [link](https://huggingface.co/openai/clip-vit-base-patch32)
- **timm/mobilenetv3_small_100.lamb_in1k** (score: 8, source: huggingface, added: 2026-08-06) — Edge-friendly image classifier â€” timm tutorial; backs Homie's offline photo-organizer plugin [link](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k)
- **nomic-ai/nomic-embed-text-v1.5** (score: 8, source: huggingface, added: 2026-08-06) — Matryoshka embeddings tutorial (resizable dimensions); lets Homie trade RAG accuracy for memory dynamically [link](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)

### Prototyping (scored >= 7)

- **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** (score: 7, source: huggingface, added: 2026-08-06) — Multilingual embeddings lesson (ties into PyMasters' 8-language push); enables non-English semantic search in Homie [link](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **mteb/leaderboard** (score: 7, source: huggingface, added: 2026-08-06) — Guide to using MTEB to pick an embedder â€” directly informs Homie's default embedding choices [link](https://huggingface.co/spaces/mteb/leaderboard)

### Evaluating (scored >= 6)

- **TheDima/arxiv-cs-2022-2026** (score: 6, source: huggingface, added: 2026-08-06) — Build-a-paper-search-engine tutorial (embeddings + RAG over arXiv CS); Homie research-assistant plugin for local paper Q&A [link](https://huggingface.co/datasets/TheDima/arxiv-cs-2022-2026)

### Discovered (new)

*No items yet.*
