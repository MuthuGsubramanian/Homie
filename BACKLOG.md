## Innovation Backlog

*Auto-maintained by the PyMasters AI Intelligence Pipeline.*
*Last updated: 2026-07-27*

### Ready to Build (scored >= 8, validated)

- **sentence-transformers/all-MiniLM-L6-v2** (score: 9, source: huggingface, added: 2026-07-27) — Cornerstone embedding model â€” extend existing semantic-search lesson into a full RAG series; already powers Homie-style local embedding workflows [link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Qwen/Qwen3-0.6B** (score: 9, source: huggingface, added: 2026-07-27) — Sub-1B local LLM â€” LoRA fine-tune lesson exists, add a deployment/quantization follow-up; ideal Homie fallback model for low-end and Android devices [link](https://huggingface.co/Qwen/Qwen3-0.6B)
- **hexgrad/Kokoro-TTS** (score: 9, source: huggingface, added: 2026-07-27) — 82M CPU-friendly TTS already covered in social content â€” next step is a Homie voice-output plugin using Kokoro as the default local TTS engine [link](https://huggingface.co/spaces/hexgrad/Kokoro-TTS)
- **cross-encoder/ms-marco-MiniLM-L6-v2** (score: 8, source: huggingface, added: 2026-07-27) — Two-stage retrieval tutorial (retrieve + rerank); Homie RAG could add this as an optional reranking stage for better local doc Q&A [link](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- **BAAI/bge-small-en-v1.5** (score: 8, source: huggingface, added: 2026-07-27) — Small fast English embedder â€” lesson comparing bge-small vs MiniLM; strong candidate for Homie's default low-RAM RAG embedder [link](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **BAAI/bge-m3** (score: 8, source: huggingface, added: 2026-07-27) — BGE-M3 hybrid dense+sparse retrieval lesson (already drafted â€” extend to hybrid search demo); multilingual RAG upgrade for Homie [link](https://huggingface.co/BAAI/bge-m3)
- **openai/clip-vit-base-patch32** (score: 8, source: huggingface, added: 2026-07-27) — CLIP multimodal lesson exists â€” extend to building an image search engine; Homie zero-shot classifier plugin already shipped, could add local photo semantic search [link](https://huggingface.co/openai/clip-vit-base-patch32)
- **BAAI/bge-reranker-v2-m3** (score: 8, source: huggingface, added: 2026-07-27) — Multilingual reranker â€” pairs with BGE-M3 lesson for a complete multilingual RAG stack; drop-in rerank stage for Homie RAG [link](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- **Qwen/Qwen3-8B** (score: 8, source: huggingface, added: 2026-07-27) — Lesson on running/quantizing an 8B model locally (GGUF, llama.cpp); Homie's 'capable desktop' tier model recommendation [link](https://huggingface.co/Qwen/Qwen3-8B)
- **intfloat/multilingual-e5-small** (score: 8, source: huggingface, added: 2026-07-27) — Small multilingual embedder with ONNX/OpenVINO â€” good CPU-only RAG lesson; strong Homie multilingual embedding default [link](https://huggingface.co/intfloat/multilingual-e5-small)
- **The new rules of context engineering for Claude 5 generation models** (score: 8, source: hackernews, added: 2026-07-27) — Timely tutorial/lesson on context engineering for Claude 5-class models; principles also apply to Homie's local prompt/context management [link](https://claude.com/blog/the-new-rules-of-context-engineering-for-claude-5-generation-models)
- **Skill Router – a local-first router for large agent skill libraries** (score: 8, source: hackernews, added: 2026-07-27) — Local-first skill routing maps directly onto Homie's plugin system â€” evaluate for routing large plugin libraries; also a solid PyMasters agent-architecture lesson [link](https://github.com/torakagemusha-sudo/torafirma-skill-router)

### Prototyping (scored >= 7)

- **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** (score: 7, source: huggingface, added: 2026-07-27) — Multilingual embeddings tutorial aligning with PyMasters' 8-language push; enables non-English RAG in Homie [link](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **sentence-transformers/all-mpnet-base-v2** (score: 7, source: huggingface, added: 2026-07-27) — Quality-vs-speed embedding benchmark lesson (mpnet vs MiniLM); Homie 'high-quality' embedding tier for capable desktops [link](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- **timm/mobilenetv3_small_100.lamb_in1k** (score: 7, source: huggingface, added: 2026-07-27) — Homie photo-tagger plugin already uses it â€” extend to on-device batch photo organization; minor edge-AI lesson potential [link](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k)
- **Attestable/scam-detection-corpus** (score: 7, source: huggingface, added: 2026-07-27) — Text-classification tutorial (train a scam detector); privacy-first local scam/phishing-detection plugin is a natural Homie feature [link](https://huggingface.co/datasets/Attestable/scam-detection-corpus)
- **Open-weight AI is having its Kubernetes moment** (score: 7, source: hackernews, added: 2026-07-27) — Blog post on the open-weight ecosystem maturing; validates Homie's local-LLM strategy and guides which open models to support [link](https://tobi.knaup.me/2026-07-25-open-weight-ai-is-having-its-kubernetes-moment/)

### Evaluating (scored >= 6)

- **Wattage: A token-spend profiler and cost-regression gate for AI agents** (score: 6, source: hackernews, added: 2026-07-27) — Tutorial on profiling token spend in agent pipelines; Homie could adopt similar cost/token tracking for local-vs-cloud model calls [link](https://github.com/faizannraza/wattage)

### Discovered (new)

*No items yet.*
