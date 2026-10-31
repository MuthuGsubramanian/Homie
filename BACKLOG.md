## Innovation Backlog

*Auto-maintained by the PyMasters AI Intelligence Pipeline.*
*Last updated: 2026-10-26*

### Ready to Build (scored >= 8, validated)

- **hexgrad/Kokoro-TTS** (score: 10, source: huggingface, added: 2026-10-26) — Homie kokoro_tts plugin exists but is unfinished and not wired into homie.py; complete, test, and ship it as the default offline voice. PyMasters Short/explainer scripts are drafted and await publishing. [link](https://huggingface.co/spaces/hexgrad/Kokoro-TTS)
- **sentence-transformers/all-MiniLM-L6-v2** (score: 9, source: huggingface, added: 2026-10-26) — PyMasters already has a MiniLM deep-dive set; extend with a 'quantize MiniLM to ONNX int8' lesson. Homie: default on-device embedder for RAG (already used by PyMasters semantic search). [link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **cross-encoder/ms-marco-MiniLM-L6-v2** (score: 9, source: huggingface, added: 2026-10-26) — Tutorial on two-stage retrieval (bi-encoder recall + cross-encoder rerank) with measurable precision gains. Homie: lightweight local reranker stage for RAG answers. [link](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- **Qwen/Qwen3-0.6B** (score: 9, source: huggingface, added: 2026-10-26) — PyMasters has a Qwen3-0.6B lesson; add GRPO/LoRA fine-tune on it to connect with the new GRPO trilogy. Homie: default ultra-light local LLM for Android and low-RAM devices, thinking mode toggle. [link](https://huggingface.co/Qwen/Qwen3-0.6B)
- **BAAI/bge-small-en-v1.5** (score: 8, source: huggingface, added: 2026-10-26) — Head-to-head lesson: bge-small vs MiniLM on MTEB subsets with query-instruction prefix explained. Homie: swap-in embedder option with better English retrieval at similar size. [link](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** (score: 8, source: huggingface, added: 2026-10-26) — Multilingual semantic search lesson (Tamil/Hindi queries over English docs) fits PyMasters' 8-language push. Homie: enable non-English RAG and voice queries locally. [link](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **openai/clip-vit-base-patch32** (score: 8, source: huggingface, added: 2026-10-26) — Lesson on contrastive image-text embeddings and zero-shot classification. Homie: local 'find photos by description' search over the user's files (CLIP plugin drafted, untested). [link](https://huggingface.co/openai/clip-vit-base-patch32)
- **BAAI/bge-reranker-v2-m3** (score: 8, source: huggingface, added: 2026-10-26) — Multilingual reranking lesson completing the retrieval stack (bge-m3 + bge-reranker-v2-m3). Homie: reranker plugin already drafted; integrate into RAG pipeline. [link](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- **nomic-ai/nomic-embed-text-v1.5** (score: 8, source: huggingface, added: 2026-10-26) — Lesson on Matryoshka embeddings: truncate dimensions and measure recall vs storage. Homie: adjustable embedding size to fit device memory budgets. [link](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)

### Prototyping (scored >= 7)

- **BAAI/bge-m3** (score: 7, source: huggingface, added: 2026-10-26) — Lesson on dense + sparse + ColBERT hybrid retrieval from one model (8k context, 100 languages). Homie: premium multilingual embedder tier for users with more RAM. [link](https://huggingface.co/BAAI/bge-m3)
- **sentence-transformers/all-mpnet-base-v2** (score: 7, source: huggingface, added: 2026-10-26) — Benchmark lesson: when does mpnet's accuracy justify 5x MiniLM's latency. Homie: 'quality' embedder preset for desktop users. [link](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- **timm/mobilenetv3_small_100.lamb_in1k** (score: 7, source: huggingface, added: 2026-10-26) — Homie plugin for on-device image tagging of local photo folders (prior MobileNetV3 plugin exists; finish and integrate). PyMasters: timm fine-tuning on a custom dataset lesson. [link](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k)
- **mteb/leaderboard** (score: 7, source: huggingface, added: 2026-10-26) — MTEB lesson JSON already drafted; publish it. Homie: pull MTEB scores to justify default embedder choices in docs and settings. [link](https://huggingface.co/spaces/mteb/leaderboard)
- **black-forest-labs/FLUX.1-schnell** (score: 7, source: huggingface, added: 2026-10-26) — Homie flux_image_gen plugin is drafted but untested; finish, measure VRAM, and gate on hardware. PyMasters: 'run FLUX schnell locally with diffusers' walkthrough. [link](https://huggingface.co/spaces/black-forest-labs/FLUX.1-schnell)

### Evaluating (scored >= 6)

- **open-llm-leaderboard/open_llm_leaderboard** (score: 6, source: huggingface, added: 2026-10-26) — Lesson on reading leaderboards critically (benchmark contamination, averaging). Homie: use as a data source for a 'recommended local model' picker. [link](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)

### Discovered (new)

*No items yet.*
