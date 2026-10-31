## Innovation Backlog

*Auto-maintained by the PyMasters AI Intelligence Pipeline.*
*Last updated: 2026-07-03*

### Ready to Build (scored >= 8, validated)

- **hexgrad/Kokoro-82M** (score: 10, source: huggingface, added: 2026-07-03) — Tutorial on local TTS in Python; Kokoro-82M is small and fast â€” ideal for Homie's offline voice output [link](https://huggingface.co/hexgrad/Kokoro-82M)
- **sentence-transformers/all-MiniLM-L6-v2** (score: 9, source: huggingface, added: 2026-07-03) — Tutorial on semantic search/embeddings with sentence-transformers; ideal lightweight default embedder for Homie's local RAG [link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Qwen/Qwen3-0.6B** (score: 9, source: huggingface, added: 2026-07-03) — Tutorial on running tiny LLMs locally; Qwen3-0.6B is a great candidate for Homie's on-device fallback model on weak hardware/Android [link](https://huggingface.co/Qwen/Qwen3-0.6B)
- **Qwen/Qwen3-8B** (score: 9, source: huggingface, added: 2026-07-03) — Tutorial on serving an 8B model locally (llama.cpp/quantization); Qwen3-8B is a prime default local LLM for Homie on desktops [link](https://huggingface.co/Qwen/Qwen3-8B)
- **hexgrad/Kokoro-TTS** (score: 9, source: huggingface, added: 2026-07-03) — Walkthrough of a local TTS app; reference implementation for Homie's offline voice output plugin [link](https://huggingface.co/spaces/hexgrad/Kokoro-TTS)
- **cross-encoder/ms-marco-MiniLM-L6-v2** (score: 8, source: huggingface, added: 2026-07-03) — Tutorial on two-stage retrieval (retrieve + rerank); Homie could use it as a fast local reranker to boost RAG answer quality [link](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- **BAAI/bge-small-en-v1.5** (score: 8, source: huggingface, added: 2026-07-03) — Compare-the-embedders tutorial content; strong small English embedding option for Homie RAG on low-resource devices [link](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** (score: 8, source: huggingface, added: 2026-07-03) — Tutorial on multilingual embeddings; lets Homie offer RAG/search for non-English users with a small local model [link](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **sentence-transformers/all-mpnet-base-v2** (score: 8, source: huggingface, added: 2026-07-03) — Benchmark piece: MiniLM vs mpnet quality/speed tradeoffs; solid higher-quality embedding tier for Homie RAG [link](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- **openai/clip-vit-base-patch32** (score: 8, source: huggingface, added: 2026-07-03) — Tutorial on zero-shot image classification/CLIP embeddings; enables Homie's local image search ('find photos of my dog') [link](https://huggingface.co/openai/clip-vit-base-patch32)
- **nomic-ai/nomic-embed-text-v1.5** (score: 8, source: huggingface, added: 2026-07-03) — Tutorial on Matryoshka/resizable embeddings; ONNX support makes it a strong cross-platform embedder for Homie RAG [link](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- **mteb/leaderboard** (score: 8, source: huggingface, added: 2026-07-03) — Guide to using the MTEB leaderboard to choose embeddings; directly informs which embedder Homie should ship [link](https://huggingface.co/spaces/mteb/leaderboard)

### Prototyping (scored >= 7)

- **BAAI/bge-m3** (score: 7, source: huggingface, added: 2026-07-03) — Tutorial on hybrid dense+sparse+multi-vector retrieval with BGE-M3; multilingual long-context embedder for Homie's document RAG [link](https://huggingface.co/BAAI/bge-m3)
- **BAAI/bge-reranker-v2-m3** (score: 7, source: huggingface, added: 2026-07-03) — Advanced RAG tutorial on multilingual reranking; drop-in reranker to improve Homie retrieval precision [link](https://huggingface.co/BAAI/bge-reranker-v2-m3)

### Evaluating (scored >= 6)

- **timm/mobilenetv3_small_100.lamb_in1k** (score: 6, source: huggingface, added: 2026-07-03) — Short tutorial on edge image classification with timm; could power a lightweight local photo-tagging plugin in Homie [link](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k)
- **BAAI/bge-large-en-v1.5** (score: 6, source: huggingface, added: 2026-07-03) — Content on embedding size/quality tradeoffs; optional high-quality embedding tier for Homie users with more RAM [link](https://huggingface.co/BAAI/bge-large-en-v1.5)
- **laion/clap-htsat-fused** (score: 6, source: huggingface, added: 2026-07-03) — Tutorial on audio-text embeddings with CLAP; could power a local sound/audio search plugin in Homie [link](https://huggingface.co/laion/clap-htsat-fused)

### Discovered (new)

*No items yet.*
