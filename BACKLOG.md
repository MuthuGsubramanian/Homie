## Innovation Backlog

*Auto-maintained by the PyMasters AI Intelligence Pipeline.*
*Last updated: 2026-07-15*

### Ready to Build (scored >= 8, validated)

- **hexgrad/Kokoro-TTS** (score: 10, source: huggingface, added: 2026-07-15) — Kokoro is a top-tier lightweight open TTS â€” ideal upgrade for Homie's voice output on-device; also a solid 'build a talking assistant' PyMasters tutorial [link](https://huggingface.co/spaces/hexgrad/Kokoro-TTS)
- **sentence-transformers/all-MiniLM-L6-v2** (score: 9, source: huggingface, added: 2026-07-15) — The default embedding model for semantic search â€” PyMasters tutorial on building RAG from scratch; Homie already-compatible lightweight local embedder for its RAG pipeline [link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Qwen/Qwen3-0.6B** (score: 9, source: huggingface, added: 2026-07-15) — Sub-1B modern LLM â€” tutorial on running LLMs on any laptop; prime Homie candidate for low-end devices and Android [link](https://huggingface.co/Qwen/Qwen3-0.6B)
- **Qwen/Qwen3-8B** (score: 9, source: huggingface, added: 2026-07-15) — Qwen3-8B is the sweet spot for local desktop LLMs â€” tutorial on quantized local inference; strong default chat model for Homie via ollama/llama.cpp [link](https://huggingface.co/Qwen/Qwen3-8B)
- **cross-encoder/ms-marco-MiniLM-L6-v2** (score: 8, source: huggingface, added: 2026-07-15) — Cross-encoder reranking tutorial (two-stage retrieval); Homie could add reranking to improve RAG answer quality with a tiny ONNX model [link](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- **BAAI/bge-small-en-v1.5** (score: 8, source: huggingface, added: 2026-07-15) — Compact high-quality English embedder with ONNX â€” tutorial comparing embedding models; strong candidate for Homie's default local embedding backend [link](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** (score: 8, source: huggingface, added: 2026-07-15) — Multilingual sentence embeddings â€” tutorial on cross-language semantic search; lets Homie's RAG work for non-English users (aligns with PyMasters' 8-language push) [link](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **BAAI/bge-m3** (score: 8, source: huggingface, added: 2026-07-15) — BGE-M3's dense+sparse+multi-vector hybrid retrieval is a great advanced-RAG tutorial; Homie hybrid search plugin, though the model is heavier than ideal for edge devices [link](https://huggingface.co/BAAI/bge-m3)
- **sentence-transformers/all-mpnet-base-v2** (score: 8, source: huggingface, added: 2026-07-15) — The quality benchmark among sentence-transformers â€” use in a 'speed vs quality embedding tradeoffs' lesson; Homie high-quality embedding option on desktop [link](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- **openai/clip-vit-base-patch32** (score: 8, source: huggingface, added: 2026-07-15) — CLIP zero-shot classification tutorial (text-image embeddings); Homie could use it for local photo search by natural language [link](https://huggingface.co/openai/clip-vit-base-patch32)
- **nomic-ai/nomic-embed-text-v1.5** (score: 8, source: huggingface, added: 2026-07-15) — Nomic's resizable (Matryoshka) embeddings make a great tutorial on dimension/storage tradeoffs; ONNX support fits Homie's local RAG well [link](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- **intfloat/multilingual-e5-small** (score: 8, source: huggingface, added: 2026-07-15) — Small multilingual E5 with OpenVINO/ONNX â€” ideal Homie embedder for weaker hardware and non-English users; comparison entry in an embeddings lesson [link](https://huggingface.co/intfloat/multilingual-e5-small)

### Prototyping (scored >= 7)

- **BAAI/bge-reranker-v2-m3** (score: 7, source: huggingface, added: 2026-07-15) — Multilingual reranker â€” pairs with BGE-M3 for an advanced retrieval-pipeline tutorial; Homie RAG quality boost for multilingual documents [link](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- **mteb/leaderboard** (score: 7, source: huggingface, added: 2026-07-15) — MTEB leaderboard â€” companion resource for embedding tutorials ('how to choose an embedding model'); guides Homie's embedder selection [link](https://huggingface.co/spaces/mteb/leaderboard)

### Evaluating (scored >= 6)

- **timm/mobilenetv3_small_100.lamb_in1k** (score: 6, source: huggingface, added: 2026-07-15) — MobileNetV3 tutorial on efficient on-device image classification with timm; could power a lightweight Homie image-tagging plugin [link](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k)
- **black-forest-labs/FLUX.1-schnell** (score: 6, source: huggingface, added: 2026-07-15) — FLUX.1-schnell's speed makes 'fast local image generation' feasible as a tutorial and a stretch Homie plugin for GPU-equipped desktops [link](https://huggingface.co/spaces/black-forest-labs/FLUX.1-schnell)

### Discovered (new)

*No items yet.*
