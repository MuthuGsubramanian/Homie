## Innovation Backlog

*Auto-maintained by the PyMasters AI Intelligence Pipeline.*
*Last updated: 2026-07-12*

### Ready to Build (scored >= 8, validated)

- **hexgrad/Kokoro-TTS** (score: 10, source: huggingface, added: 2026-07-12) — Kokoro is a tiny high-quality open TTS â€” a near-perfect upgrade for Homie's local voice output across desktop platforms; also a nice PyMasters 'build a talking app' tutorial [link](https://huggingface.co/spaces/hexgrad/Kokoro-TTS)
- **sentence-transformers/all-MiniLM-L6-v2** (score: 9, source: huggingface, added: 2026-07-12) — PyMasters tutorial on semantic search with sentence-transformers; the go-to lightweight local embedder for Homie's RAG pipeline (ONNX support enables fast on-device inference) [link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **BAAI/bge-small-en-v1.5** (score: 9, source: huggingface, added: 2026-07-12) — Small, strong English embedder with ONNX export â€” ideal candidate for Homie's default local embedding model; PyMasters can benchmark it vs MiniLM in a tutorial [link](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **Qwen/Qwen3-0.6B** (score: 9, source: huggingface, added: 2026-07-12) — Tiny 0.6B LLM perfect for Homie's low-resource fallback / fast intent routing; PyMasters tutorial on running sub-1B models locally [link](https://huggingface.co/Qwen/Qwen3-0.6B)
- **Qwen/Qwen3-8B** (score: 9, source: huggingface, added: 2026-07-12) — Strong 8B local chat model â€” prime candidate for Homie's default local LLM tier; PyMasters tutorial on serving it via Ollama/llama.cpp [link](https://huggingface.co/Qwen/Qwen3-8B)
- **nomic-ai/nomic-embed-text-v1.5** (score: 9, source: huggingface, added: 2026-07-12) — Tutorial on Matryoshka embeddings (resizable dimensions); Homie could use nomic-embed for memory-tunable local RAG indexes [link](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- **cross-encoder/ms-marco-MiniLM-L6-v2** (score: 8, source: huggingface, added: 2026-07-12) — Tutorial on two-stage retrieval (embed then rerank); Homie could add this cross-encoder as a rerank step to improve local RAG answer quality [link](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** (score: 8, source: huggingface, added: 2026-07-12) — Multilingual embeddings tutorial (relevant to PyMasters' 8-language content); enables Homie RAG over non-English user documents [link](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **BAAI/bge-m3** (score: 8, source: huggingface, added: 2026-07-12) — Tutorial on hybrid dense+sparse+multi-vector retrieval with BGE-M3; strong multilingual embedder option for Homie's RAG [link](https://huggingface.co/BAAI/bge-m3)
- **sentence-transformers/all-mpnet-base-v2** (score: 8, source: huggingface, added: 2026-07-12) — Higher-quality embedding option to compare against MiniLM in a PyMasters retrieval-quality tutorial; quality tier for Homie RAG when device resources allow [link](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- **BAAI/bge-reranker-v2-m3** (score: 8, source: huggingface, added: 2026-07-12) — Multilingual reranker for Homie's RAG pipeline to boost retrieval precision; pairs with a PyMasters 'build a reranked RAG' lesson [link](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- **google/gemma-4-26B-A4B-it** (score: 8, source: huggingface, added: 2026-07-12) — Gemma 4 MoE (26B total, ~4B active) with vision â€” tutorial on efficient MoE multimodal models; a strong local multimodal option for Homie on capable desktops [link](https://huggingface.co/google/gemma-4-26B-A4B-it)

### Prototyping (scored >= 7)

- **openai/clip-vit-base-patch32** (score: 7, source: huggingface, added: 2026-07-12) — Tutorial on zero-shot image classification and image-text embeddings; CLIP could power local image search over the user's photos in Homie [link](https://huggingface.co/openai/clip-vit-base-patch32)

### Evaluating (scored >= 6)

- **timm/mobilenetv3_small_100.lamb_in1k** (score: 6, source: huggingface, added: 2026-07-12) — Tutorial on efficient edge vision with timm; Homie could use MobileNetV3 for a lightweight on-device image classification plugin (fits Android too) [link](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k)
- **BAAI/bge-large-en-v1.5** (score: 6, source: huggingface, added: 2026-07-12) — Quality-tier English embedder for a small-vs-large embedding tradeoff tutorial; viable on desktop Homie installs [link](https://huggingface.co/BAAI/bge-large-en-v1.5)
- **open-llm-leaderboard/open_llm_leaderboard** (score: 6, source: huggingface, added: 2026-07-12) — Tutorial on how to read LLM benchmarks and pick a model; Homie docs/plugin could point users to it when choosing a local model [link](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
- **mteb/leaderboard** (score: 6, source: huggingface, added: 2026-07-12) — Tutorial on choosing embedding models via MTEB; informs which embedder Homie ships as default for RAG [link](https://huggingface.co/spaces/mteb/leaderboard)

### Discovered (new)

*No items yet.*
