## Innovation Backlog

*Auto-maintained by the PyMasters AI Intelligence Pipeline.*
*Last updated: 2026-10-28*

### Ready to Build (scored >= 8, validated)

- **sentence-transformers/all-MiniLM-L6-v2** (score: 9, source: huggingface, added: 2026-10-28) — Core embedding model for RAG tutorials; Homie's local RAG/doc-Q&A plugin can use this as the default lightweight encoder [link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Qwen/Qwen3-0.6B** (score: 9, source: huggingface, added: 2026-10-28) — Tiny conversational LLM that could become Homie's default low-resource local chat model [link](https://huggingface.co/Qwen/Qwen3-0.6B)
- **cross-encoder/ms-marco-MiniLM-L6-v2** (score: 8, source: huggingface, added: 2026-10-28) — Teach reranking in a two-stage RAG lesson; Homie can bolt this on to improve local search result ordering [link](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- **BAAI/bge-small-en-v1.5** (score: 8, source: huggingface, added: 2026-10-28) — Alternative compact embedding model for a RAG/embeddings comparison tutorial and as a Homie offline encoder option [link](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **BAAI/bge-m3** (score: 8, source: huggingface, added: 2026-10-28) — Multilingual, multi-functionality embedding model good for a 'next-gen embeddings' lesson and a stronger Homie RAG backbone [link](https://huggingface.co/BAAI/bge-m3)
- **sentence-transformers/all-mpnet-base-v2** (score: 8, source: huggingface, added: 2026-10-28) — Strong general-purpose embedding model; good Homie default for higher-quality semantic search when latency allows [link](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- **openai/clip-vit-base-patch32** (score: 8, source: huggingface, added: 2026-10-28) — Zero-shot multimodal classification lesson material; also the backbone for Homie's existing CLIP photo-search plugin [link](https://huggingface.co/openai/clip-vit-base-patch32)
- **timm/mobilenetv3_small_100.lamb_in1k** (score: 8, source: huggingface, added: 2026-10-28) — Lightweight on-device vision model well suited to Homie's mobile/Android image-tagging plugin [link](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k)
- **BAAI/bge-reranker-v2-m3** (score: 8, source: huggingface, added: 2026-10-28) — Multilingual reranker for an advanced RAG lesson and a stronger Homie search-quality plugin [link](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- **nomic-ai/nomic-embed-text-v1.5** (score: 8, source: huggingface, added: 2026-10-28) — Long-context open embedding model, good RAG-lesson comparison point and solid Homie local-embedding option [link](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- **hexgrad/Kokoro-TTS** (score: 8, source: huggingface, added: 2026-10-28) — Already basis of an existing untested Homie TTS plugin; finish wiring Kokoro in as Homie's offline voice output [link](https://huggingface.co/spaces/hexgrad/Kokoro-TTS)

### Prototyping (scored >= 7)

- **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** (score: 7, source: huggingface, added: 2026-10-28) — Fits PyMasters' multilingual content work; Homie could use it for cross-language semantic search [link](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **google-t5/t5-small** (score: 7, source: huggingface, added: 2026-10-28) — Small local translation model directly useful for the lesson-translation pipeline and a Homie offline-translate plugin [link](https://huggingface.co/google-t5/t5-small)

### Evaluating (scored >= 6)

- **google/vit-base-patch16-224** (score: 6, source: huggingface, added: 2026-10-28) — Standard ViT for a vision-transformer fundamentals lesson; usable as a Homie image-classification fallback [link](https://huggingface.co/google/vit-base-patch16-224)
- **black-forest-labs/FLUX.1-dev** (score: 6, source: huggingface, added: 2026-10-28) — Flagship open image-gen model worth a diffusion-models lesson; Homie could expose it as an opt-in local image-gen plugin [link](https://huggingface.co/spaces/black-forest-labs/FLUX.1-dev)
- **mteb/leaderboard** (score: 6, source: huggingface, added: 2026-10-28) — Embedding benchmark leaderboard useful for picking/justifying the embedding model in both RAG lessons and Homie's plugin choices [link](https://huggingface.co/spaces/mteb/leaderboard)

### Discovered (new)

*No items yet.*
