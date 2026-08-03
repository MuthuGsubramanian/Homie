## Innovation Backlog

*Auto-maintained by the PyMasters AI Intelligence Pipeline.*
*Last updated: 2026-08-03*

### Ready to Build (scored >= 8, validated)

- **hexgrad/Kokoro-TTS** (score: 10, source: huggingface, added: 2026-08-03) — Kokoro is the ideal local TTS for Homie's voice output (plugin already built); PyMasters tutorial on lightweight on-device speech synthesis [link](https://huggingface.co/spaces/hexgrad/Kokoro-TTS)
- **sentence-transformers/all-MiniLM-L6-v2** (score: 9, source: huggingface, added: 2026-08-03) — Flagship lightweight embedder â€” extend existing MiniLM lesson series (semantic search, dedup, clustering) and use as Homie's default local RAG embedding model [link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Qwen/Qwen3-0.6B** (score: 9, source: huggingface, added: 2026-08-03) — Tiny local LLM tutorial (run an SLM on your laptop); strong candidate as Homie's bundled low-resource fallback model [link](https://huggingface.co/Qwen/Qwen3-0.6B)
- **openai/clip-vit-base-patch32** (score: 9, source: huggingface, added: 2026-08-03) — Zero-shot image classification tutorial; powers Homie's local photo-search plugin (already prototyped with ChromaDB indexing) [link](https://huggingface.co/openai/clip-vit-base-patch32)
- **BAAI/bge-reranker-v2-m3** (score: 9, source: huggingface, added: 2026-08-03) — Multilingual reranker for Homie's RAG pipeline (plugin already drafted); pairs with a PyMasters lesson on why reranking boosts RAG accuracy [link](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- **cross-encoder/ms-marco-MiniLM-L6-v2** (score: 8, source: huggingface, added: 2026-08-03) — Tutorial on two-stage retrieval (bi-encoder recall + cross-encoder rerank); lightweight enough to power Homie's RAG reranking on CPU [link](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- **BAAI/bge-small-en-v1.5** (score: 8, source: huggingface, added: 2026-08-03) — Compare bge-small vs MiniLM in a 'choosing your embedder' lesson; strong ONNX-friendly candidate for Homie's local RAG index [link](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **BAAI/bge-m3** (score: 8, source: huggingface, added: 2026-08-03) — Tutorial on hybrid dense+sparse+multi-vector retrieval with bge-m3; gives Homie multilingual long-context retrieval in one model [link](https://huggingface.co/BAAI/bge-m3)
- **sentence-transformers/all-mpnet-base-v2** (score: 8, source: huggingface, added: 2026-08-03) — Quality-vs-speed embedder comparison (mpnet vs MiniLM) lesson; higher-accuracy option for Homie RAG on capable machines [link](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- **timm/mobilenetv3_small_100.lamb_in1k** (score: 8, source: huggingface, added: 2026-08-03) — Efficient edge-CV lesson with timm; CPU-friendly image classifier for Homie's local photo tagging (plugin drafted) [link](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k)
- **Qwen/Qwen3-8B** (score: 8, source: huggingface, added: 2026-08-03) — Quantized Qwen3-8B as Homie's 'capable tier' local model; PyMasters tutorial on running 8B models with llama.cpp/Ollama [link](https://huggingface.co/Qwen/Qwen3-8B)
- **intfloat/multilingual-e5-small** (score: 8, source: huggingface, added: 2026-08-03) — Small multilingual embedder benchmark lesson; efficient multilingual RAG option for Homie on low-end hardware [link](https://huggingface.co/intfloat/multilingual-e5-small)
- **nomic-ai/nomic-embed-text-v1.5** (score: 8, source: huggingface, added: 2026-08-03) — Tutorial on Matryoshka embeddings and resizable dimensions; long-context local embedder option for Homie's document RAG [link](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- **mteb/leaderboard** (score: 8, source: huggingface, added: 2026-08-03) — 'Choosing an embedding model with MTEB' lesson â€” directly supports the embedder tutorials; guides Homie's RAG model defaults [link](https://huggingface.co/spaces/mteb/leaderboard)
- **Rethinking Inference-Time Scaling in Local Computer-Use Agents: Failure Modes and Compute Tradeoffs** (score: 8, source: arxiv, added: 2026-08-03) — Failure modes and compute tradeoffs of LOCAL computer-use agents map directly onto Homie's design space (what inference-time scaling is worth on consumer hardware); also a strong lesson on local-agent constraints [link](http://arxiv.org/abs/2607.28573v1)

### Prototyping (scored >= 7)

- **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** (score: 7, source: huggingface, added: 2026-08-03) — Multilingual embeddings tutorial (fits PyMasters' 8-language push); enables non-English semantic search in Homie's RAG [link](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **Sample More, Reflect Less: Self-Refine and Reflexion Lose to Repeated Sampling at Equal Token Cost, from 1.5B to 7B** (score: 7, source: arxiv, added: 2026-08-03) — Repeated sampling beating self-reflection at equal token cost on 1.5Bâ€“7B models is directly actionable: a PyMasters lesson on inference-strategy budgeting and a default strategy for Homie's local SLMs [link](http://arxiv.org/abs/2607.28576v1)

### Evaluating (scored >= 6)

- **MixFrag: Fragility-Guided Mixed-Precision Post-Training Quantization for Vision Transformers** (score: 6, source: arxiv, added: 2026-08-03) — Mixed-precision PTQ for ViTs fits a PyMasters tutorial on quantizing vision models and directly informs how Homie squeezes vision models onto consumer devices [link](http://arxiv.org/abs/2607.28589v1)
- **DualG-MRAG: Decoupling Macro-Reasoning and Micro-Matching for Multimodal Retrieval-Augmented Generation** (score: 6, source: arxiv, added: 2026-08-03) — Decoupled macro-reasoning/micro-matching is a pattern Homie's multimodal RAG over local docs could adopt, and a good advanced-RAG lesson topic [link](http://arxiv.org/abs/2607.28580v1)
- **Show HN: Mu – Tools for Agents** (score: 6, source: hackernews, added: 2026-08-03) — Evaluate Mu's agent-tool patterns for Homie's plugin API; possible interop or design borrowing for local tool-calling [link](https://github.com/micro/mu)
- **Show HN: MicroCodex Coding Agent – OpenAI/codex reimplemented in C++ <1MB binary** (score: 6, source: hackernews, added: 2026-08-03) — A <1MB C++ coding agent validates the minimal-footprint local-agent approach; study its architecture for Homie's lightweight agent core [link](https://github.com/paoloanzn/microcodex)
- **KotlinLLM** (score: 6, source: hackernews, added: 2026-08-03) — A Kotlin LLM library is directly relevant to Homie's Android build â€” evaluate it for on-device inference on the Android client [link](https://github.com/JetBrains-Research/kotlinllm-plugin)

### Discovered (new)

*No items yet.*
