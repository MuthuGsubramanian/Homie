## Innovation Backlog

*Auto-maintained by the PyMasters AI Intelligence Pipeline.*
*Last updated: 2026-07-17*

### Ready to Build (scored >= 8, validated)

- **Qwen/Qwen3-0.6B** (score: 10, source: huggingface, added: 2026-07-17) — Sub-1B local LLM lesson (already built) â€” memory math, ChatML, quantization; the natural default chat model for Homie on low-RAM devices and Android [link](https://huggingface.co/Qwen/Qwen3-0.6B)
- **hexgrad/Kokoro-TTS** (score: 10, source: huggingface, added: 2026-07-17) — Kokoro is the top local TTS pick for Homie's voice output (small, fast, CPU-friendly); pairs with a 'build a talking assistant' PyMasters lesson [link](https://huggingface.co/spaces/hexgrad/Kokoro-TTS)
- **sentence-transformers/all-MiniLM-L6-v2** (score: 9, source: huggingface, added: 2026-07-17) — Flagship lesson on sentence embeddings and semantic search; already the default local embedder pattern for Homie's RAG â€” a 'build on-device semantic search' tutorial doubles as Homie documentation [link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **BAAI/bge-small-en-v1.5** (score: 9, source: huggingface, added: 2026-07-17) — Small, fast English embedder ideal for Homie's local RAG index; PyMasters lesson comparing bge-small vs all-MiniLM for retrieval quality per MB [link](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **BAAI/bge-m3** (score: 9, source: huggingface, added: 2026-07-17) — Tutorial on hybrid retrieval (dense + sparse + multi-vector in one model); Homie could use bge-m3 as the upgrade path for multilingual RAG [link](https://huggingface.co/BAAI/bge-m3)
- **openai/clip-vit-base-patch32** (score: 9, source: huggingface, added: 2026-07-17) — Text-to-image search tutorial with CLIP; already prototyped as Homie's clip_photo_search plugin â€” ship it and write the companion lesson [link](https://huggingface.co/openai/clip-vit-base-patch32)
- **Qwen/Qwen3-8B** (score: 9, source: huggingface, added: 2026-07-17) — Lesson on running an 8B model locally (quantization, GGUF, VRAM budgeting); Homie's recommended model tier for capable desktops [link](https://huggingface.co/Qwen/Qwen3-8B)
- **nomic-ai/nomic-embed-text-v1.5** (score: 9, source: huggingface, added: 2026-07-17) — Lesson on Matryoshka embeddings (resize dimensions to fit storage); strong Homie candidate for adjustable-footprint local RAG [link](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- **cross-encoder/ms-marco-MiniLM-L6-v2** (score: 8, source: huggingface, added: 2026-07-17) — Tutorial on two-stage retrieval (bi-encoder recall + cross-encoder rerank); Homie could add this reranker to sharpen RAG answer quality with minimal CPU cost [link](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** (score: 8, source: huggingface, added: 2026-07-17) — Multilingual embeddings lesson (already drafted) that maps directly to PyMasters' 8-language goal and Homie's non-English semantic search [link](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **sentence-transformers/all-mpnet-base-v2** (score: 8, source: huggingface, added: 2026-07-17) — Quality-vs-speed embedder comparison lesson; Homie 'high-accuracy mode' option for desktop machines with more headroom [link](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- **BAAI/bge-reranker-v2-m3** (score: 8, source: huggingface, added: 2026-07-17) — Multilingual reranker lesson completing the RAG pipeline series; Homie rerank stage for higher-precision document Q&A [link](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- **intfloat/multilingual-e5-small** (score: 8, source: huggingface, added: 2026-07-17) — Small multilingual embedder well suited to Homie's RAG in non-English locales; lesson comparing e5 instruction-prefix quirks vs BGE [link](https://huggingface.co/intfloat/multilingual-e5-small)

### Prototyping (scored >= 7)

- **timm/mobilenetv3_small_100.lamb_in1k** (score: 7, source: huggingface, added: 2026-07-17) — On-device computer vision tutorial with timm; Homie plugin for fast local image tagging/classification on CPU and mobile [link](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k)

### Evaluating (scored >= 6)

*No items yet.*

### Discovered (new)

*No items yet.*
