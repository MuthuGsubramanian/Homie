## Innovation Backlog

*Auto-maintained by the PyMasters AI Intelligence Pipeline.*
*Last updated: 2026-06-20*

### Ready to Build (scored >= 8, validated)

- **sentence-transformers/all-MiniLM-L6-v2** (score: 9, source: huggingface, added: 2026-06-20) — Cornerstone tutorial on semantic search + RAG embeddings for PyMasters; the default lightweight on-device embedding model for Homie's RAG [link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **BAAI/bge-small-en-v1.5** (score: 9, source: huggingface, added: 2026-06-20) — Lesson comparing BGE vs MiniLM embeddings; small footprint makes it ideal for Homie's local RAG index [link](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **BAAI/bge-m3** (score: 9, source: huggingface, added: 2026-06-20) — Lesson on hybrid dense/sparse/multi-vector retrieval; bge-m3 gives Homie multilingual + hybrid RAG in one model [link](https://huggingface.co/BAAI/bge-m3)
- **Qwen/Qwen3-0.6B** (score: 9, source: huggingface, added: 2026-06-20) — Tutorial on running tiny LLMs locally; sub-1B Qwen3 is ideal for Homie's low-resource/Android local inference [link](https://huggingface.co/Qwen/Qwen3-0.6B)
- **hexgrad/Kokoro-82M** (score: 9, source: huggingface, added: 2026-06-20) — Tutorial on local text-to-speech; tiny 82M Kokoro is perfect for Homie's on-device voice output [link](https://huggingface.co/hexgrad/Kokoro-82M)
- **Qwen/Qwen3-4B** (score: 9, source: huggingface, added: 2026-06-20) — Tutorial on running a capable mid-size LLM locally; Qwen3-4B is a strong default brain for Homie on desktops [link](https://huggingface.co/Qwen/Qwen3-4B)
- **cross-encoder/ms-marco-MiniLM-L6-v2** (score: 8, source: huggingface, added: 2026-06-20) — Tutorial on two-stage retrieval (retrieve + rerank); Homie can use it to rerank RAG results locally for better answers [link](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** (score: 8, source: huggingface, added: 2026-06-20) — Tutorial on multilingual semantic search; powers cross-language RAG for Homie's international users [link](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **sentence-transformers/all-mpnet-base-v2** (score: 8, source: huggingface, added: 2026-06-20) — Tutorial on high-quality general embeddings; a higher-accuracy embedding option for Homie RAG when resources allow [link](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- **openai/clip-vit-base-patch32** (score: 8, source: huggingface, added: 2026-06-20) — Tutorial on multimodal/zero-shot image classification; Homie could add a local image-understanding plugin via CLIP [link](https://huggingface.co/openai/clip-vit-base-patch32)
- **nomic-ai/nomic-embed-text-v1.5** (score: 8, source: huggingface, added: 2026-06-20) — Lesson on long-context embeddings; nomic-embed gives Homie larger-document RAG with local inference [link](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- **BAAI/bge-reranker-v2-m3** (score: 8, source: huggingface, added: 2026-06-20) — Tutorial on multilingual reranking; improves precision of Homie's local RAG retrieval [link](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- **BAAI/bge-large-en-v1.5** (score: 8, source: huggingface, added: 2026-06-20) — Tutorial on high-accuracy embeddings; the large-quality tier for Homie RAG on capable hardware [link](https://huggingface.co/BAAI/bge-large-en-v1.5)
- **hexgrad/Kokoro-TTS** (score: 8, source: huggingface, added: 2026-06-20) — Hands-on tutorial demoing Kokoro TTS; a ready reference for wiring local voice output into Homie [link](https://huggingface.co/spaces/hexgrad/Kokoro-TTS)

### Prototyping (scored >= 7)

- **laion/clap-htsat-fused** (score: 7, source: huggingface, added: 2026-06-20) — Tutorial on audio-text embeddings (CLAP); Homie could classify sounds/audio for voice-context awareness [link](https://huggingface.co/laion/clap-htsat-fused)
- **colbert-ir/colbertv2.0** (score: 7, source: huggingface, added: 2026-06-20) — Lesson on late-interaction retrieval (ColBERT); an advanced retrieval option for Homie's RAG [link](https://huggingface.co/colbert-ir/colbertv2.0)
- **mteb/leaderboard** (score: 7, source: huggingface, added: 2026-06-20) — Guide on the MTEB leaderboard to choose embedding models; directly informs Homie's RAG model selection [link](https://huggingface.co/spaces/mteb/leaderboard)

### Evaluating (scored >= 6)

- **timm/mobilenetv3_small_100.lamb_in1k** (score: 6, source: huggingface, added: 2026-06-20) — Tutorial on efficient on-device image classification; MobileNetV3 fits Homie's Android/mobile vision plugin [link](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k)
- **open-llm-leaderboard/open_llm_leaderboard** (score: 6, source: huggingface, added: 2026-06-20) — Article on how to read LLM leaderboards; helps users pick a local model to run in Homie [link](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
- **lmarena-ai/arena-leaderboard** (score: 6, source: huggingface, added: 2026-06-20) — Article on LMArena human-preference rankings to guide local model choice for Homie [link](https://huggingface.co/spaces/lmarena-ai/arena-leaderboard)

### Discovered (new)

*No items yet.*
