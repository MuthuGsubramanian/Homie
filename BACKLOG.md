## Innovation Backlog

*Auto-maintained by the PyMasters AI Intelligence Pipeline.*
*Last updated: 2026-07-16*

### Ready to Build (scored >= 8, validated)

- **sentence-transformers/all-MiniLM-L6-v2** (score: 9, source: huggingface, added: 2026-07-16) — Canonical lightweight embedder â€” PyMasters tutorial on semantic search (already used in our runnable-code lessons); Homie's default local RAG embedding model via ONNX [link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **BAAI/bge-small-en-v1.5** (score: 9, source: huggingface, added: 2026-07-16) — Small, fast English embedder ideal for Homie's on-device RAG; PyMasters comparison lesson 'choosing an embedding model by size vs quality' [link](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **BAAI/bge-m3** (score: 9, source: huggingface, added: 2026-07-16) — BGE-M3 does dense+sparse+multi-vector in one model â€” strong Homie RAG upgrade; PyMasters lesson on hybrid retrieval [link](https://huggingface.co/BAAI/bge-m3)
- **Qwen/Qwen3-0.6B** (score: 9, source: huggingface, added: 2026-07-16) — Sub-1B LLM that runs anywhere â€” Homie's low-end-device fallback model; PyMasters lesson on local inference and memory math (already drafted) [link](https://huggingface.co/Qwen/Qwen3-0.6B)
- **Qwen/Qwen3-8B** (score: 9, source: huggingface, added: 2026-07-16) — Qwen3-8B is a top local-LLM candidate for Homie's desktop tier (Ollama/llama.cpp); PyMasters guide to quantized 8B inference on consumer hardware [link](https://huggingface.co/Qwen/Qwen3-8B)
- **nomic-ai/nomic-embed-text-v1.5** (score: 9, source: huggingface, added: 2026-07-16) — Nomic's Matryoshka embeddings let Homie trade dimension size for speed at runtime; PyMasters lesson on Matryoshka representation learning [link](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- **hexgrad/Kokoro-TTS** (score: 9, source: huggingface, added: 2026-07-16) — Kokoro's tiny high-quality TTS is a prime Homie voice-output engine (fully offline); PyMasters 'build a talking app' tutorial [link](https://huggingface.co/spaces/hexgrad/Kokoro-TTS)
- **cross-encoder/ms-marco-MiniLM-L6-v2** (score: 8, source: huggingface, added: 2026-07-16) — Tutorial on two-stage retrieval (embed then rerank); Homie could add it as a reranking stage to sharpen RAG answer quality [link](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** (score: 8, source: huggingface, added: 2026-07-16) — Multilingual embeddings tutorial (ties into PyMasters' 8-language effort); Homie cross-language semantic search over local notes [link](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **sentence-transformers/all-mpnet-base-v2** (score: 8, source: huggingface, added: 2026-07-16) — Quality benchmark embedder for tutorials comparing MiniLM vs MPNet tradeoffs; Homie high-quality embedding option when device allows [link](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- **openai/clip-vit-base-patch32** (score: 8, source: huggingface, added: 2026-07-16) — CLIP zero-shot classification tutorial is a high-engagement topic; Homie plugin for text-to-image search over the user's local photo library [link](https://huggingface.co/openai/clip-vit-base-patch32)
- **BAAI/bge-reranker-v2-m3** (score: 8, source: huggingface, added: 2026-07-16) — Multilingual reranker pairing with BGE-M3 â€” Homie RAG precision boost; PyMasters advanced-retrieval lesson section [link](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- **intfloat/multilingual-e5-small** (score: 8, source: huggingface, added: 2026-07-16) — Small multilingual embedder â€” good Homie default for non-English users; supports PyMasters multilingual search content [link](https://huggingface.co/intfloat/multilingual-e5-small)

### Prototyping (scored >= 7)

- **mteb/leaderboard** (score: 7, source: huggingface, added: 2026-07-16) — MTEB leaderboard as the backbone of a 'how to choose an embedding model' guide; informs Homie's embedder selection defaults [link](https://huggingface.co/spaces/mteb/leaderboard)

### Evaluating (scored >= 6)

- **timm/mobilenetv3_small_100.lamb_in1k** (score: 6, source: huggingface, added: 2026-07-16) — Tiny image classifier for edge devices â€” Homie could tag/organize local photos offline; PyMasters intro-to-timm lesson [link](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k)
- **BAAI/bge-large-en-v1.5** (score: 6, source: huggingface, added: 2026-07-16) — Large-tier embedder for the size/quality comparison axis in tutorials; Homie optional high-accuracy mode on capable machines [link](https://huggingface.co/BAAI/bge-large-en-v1.5)

### Discovered (new)

*No items yet.*
