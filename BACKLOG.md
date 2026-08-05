## Innovation Backlog

*Auto-maintained by the PyMasters AI Intelligence Pipeline.*
*Last updated: 2026-08-05*

### Ready to Build (scored >= 8, validated)

- **sentence-transformers/all-MiniLM-L6-v2** (score: 9, source: huggingface, added: 2026-08-05) — Already core to both â€” extend existing MiniLM lessons with fine-tuning/quantization content; it powers Homie's local RAG embeddings [link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Qwen/Qwen3-0.6B** (score: 9, source: huggingface, added: 2026-08-05) — Extend existing Qwen3-0.6B lesson (quantization, GGUF); ideal ultra-light local LLM for Homie on low-end devices [link](https://huggingface.co/Qwen/Qwen3-0.6B)
- **hexgrad/Kokoro-TTS** (score: 9, source: huggingface, added: 2026-08-05) — Kokoro already powers Homie's TTS plugin â€” write the PyMasters tutorial on local TTS and expand Homie voice output quality/languages [link](https://huggingface.co/spaces/hexgrad/Kokoro-TTS)
- **cross-encoder/ms-marco-MiniLM-L6-v2** (score: 8, source: huggingface, added: 2026-08-05) — Tutorial on two-stage retrieval (bi-encoder + cross-encoder rerank); lightweight reranker for Homie's RAG pipeline [link](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- **BAAI/bge-small-en-v1.5** (score: 8, source: huggingface, added: 2026-08-05) — Compact CPU-friendly embedder â€” benchmark-vs-MiniLM tutorial; strong candidate for Homie's default local embedding model [link](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** (score: 8, source: huggingface, added: 2026-08-05) — Multilingual embeddings tutorial aligning with PyMasters' 8-language support; enables non-English RAG in Homie [link](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **BAAI/bge-m3** (score: 8, source: huggingface, added: 2026-08-05) — Tutorial on hybrid dense+sparse+multi-vector retrieval; BGE-M3 could power multilingual hybrid search in Homie RAG [link](https://huggingface.co/BAAI/bge-m3)
- **openai/clip-vit-base-patch32** (score: 8, source: huggingface, added: 2026-08-05) — CLIP zero-shot classification essay pairs with the existing Homie CLIP image-search plugin â€” extend to local photo library search [link](https://huggingface.co/openai/clip-vit-base-patch32)
- **BAAI/bge-reranker-v2-m3** (score: 8, source: huggingface, added: 2026-08-05) — Already shipped as a Homie reranker plugin â€” write the companion PyMasters tutorial on reranking to close the loop [link](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- **Qwen/Qwen3-8B** (score: 8, source: huggingface, added: 2026-08-05) — Qwen3-8B as Homie's 'capable tier' local LLM (GGUF via llama.cpp); tutorial on choosing model size vs hardware [link](https://huggingface.co/Qwen/Qwen3-8B)
- **nomic-ai/nomic-embed-text-v1.5** (score: 8, source: huggingface, added: 2026-08-05) — Tutorial on Matryoshka embeddings (dimension truncation); nomic-embed is a strong long-context local embedder for Homie [link](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)

### Prototyping (scored >= 7)

- **sentence-transformers/all-mpnet-base-v2** (score: 7, source: huggingface, added: 2026-08-05) — Quality-vs-speed embedding comparison content (mpnet vs MiniLM); higher-accuracy option for Homie RAG on capable machines [link](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- **timm/mobilenetv3_small_100.lamb_in1k** (score: 7, source: huggingface, added: 2026-08-05) — Edge image-classification tutorial; already a Homie tagger plugin â€” extend with on-device batch photo organization [link](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k)

### Evaluating (scored >= 6)

*No items yet.*

### Discovered (new)

*No items yet.*
