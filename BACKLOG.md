## Innovation Backlog

*Auto-maintained by the PyMasters AI Intelligence Pipeline.*
*Last updated: 2026-07-30*

### Ready to Build (scored >= 8, validated)

- **sentence-transformers/all-MiniLM-L6-v2** (score: 9, source: huggingface, added: 2026-07-30) — Flagship lightweight embedder: PyMasters tutorial on semantic search from scratch; already proven as Homie's local RAG embedding backbone (CPU-only, ONNX) [link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **BAAI/bge-small-en-v1.5** (score: 9, source: huggingface, added: 2026-07-30) — Small, fast English embedder ideal as an alternative Homie RAG backend; PyMasters comparison lesson vs MiniLM (quality/speed tradeoffs) [link](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **Qwen/Qwen3-0.6B** (score: 9, source: huggingface, added: 2026-07-30) — Sub-1B local LLM â€” PyMasters 'run an LLM on your laptop' lesson; perfect fallback chat/summarization model for Homie on low-RAM devices [link](https://huggingface.co/Qwen/Qwen3-0.6B)
- **hexgrad/Kokoro-TTS** (score: 9, source: huggingface, added: 2026-07-30) — Kokoro already powers Homie's TTS plugin â€” deepen voice features (multi-voice, streaming); PyMasters lesson on local neural TTS [link](https://huggingface.co/spaces/hexgrad/Kokoro-TTS)
- **cross-encoder/ms-marco-MiniLM-L6-v2** (score: 8, source: huggingface, added: 2026-07-30) — Two-stage retrieval tutorial (retrieve + rerank) for PyMasters; Homie could add it as a reranking stage to sharpen RAG answer quality [link](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** (score: 8, source: huggingface, added: 2026-07-30) — Multilingual embeddings tutorial; lets Homie's RAG index and search notes in non-English languages on-device [link](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **BAAI/bge-m3** (score: 8, source: huggingface, added: 2026-07-30) — Lesson on hybrid dense+sparse+multi-vector retrieval; a heavier but higher-quality multilingual embedder option for Homie RAG [link](https://huggingface.co/BAAI/bge-m3)
- **sentence-transformers/all-mpnet-base-v2** (score: 8, source: huggingface, added: 2026-07-30) — Higher-quality embedding option â€” PyMasters benchmark exercise (MiniLM vs mpnet); optional 'quality mode' embedder in Homie RAG [link](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- **openai/clip-vit-base-patch32** (score: 8, source: huggingface, added: 2026-07-30) — Zero-shot image classification lesson (text-image embeddings); powers Homie's local photo search/tagging plugin [link](https://huggingface.co/openai/clip-vit-base-patch32)
- **BAAI/bge-reranker-v2-m3** (score: 8, source: huggingface, added: 2026-07-30) — Pairs with bge-m3 for a 'production RAG reranking' lesson; multilingual reranker plugin stage for Homie retrieval [link](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- **timm/mobilenetv3_small_100.lamb_in1k** (score: 8, source: huggingface, added: 2026-07-30) — Tiny on-device image classifier â€” PyMasters edge-AI lesson; already the basis of Homie's local image-tagger plugin [link](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k)
- **Qwen/Qwen3-8B** (score: 8, source: huggingface, added: 2026-07-30) — Lesson on quantizing and serving an 8B model locally (GGUF/llama.cpp); primary local-LLM option for Homie on capable machines [link](https://huggingface.co/Qwen/Qwen3-8B)
- **intfloat/multilingual-e5-small** (score: 8, source: huggingface, added: 2026-07-30) — Small multilingual embedder â€” good Homie RAG option for non-English users; PyMasters multilingual-retrieval comparison lesson [link](https://huggingface.co/intfloat/multilingual-e5-small)

### Prototyping (scored >= 7)

*No items yet.*

### Evaluating (scored >= 6)

*No items yet.*

### Discovered (new)

*No items yet.*
