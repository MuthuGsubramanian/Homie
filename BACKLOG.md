## Innovation Backlog

*Auto-maintained by the PyMasters AI Intelligence Pipeline.*
*Last updated: 2026-07-07*

### Ready to Build (scored >= 8, validated)

- **Qwen/Qwen3-0.6B** (score: 10, source: huggingface, added: 2026-07-07) — Tiny 0.6B LLM that runs anywhere â€” perfect Homie on-device model for Android and low-RAM machines; PyMasters tutorial on running sub-1B LLMs locally [link](https://huggingface.co/Qwen/Qwen3-0.6B)
- **hexgrad/Kokoro-TTS** (score: 10, source: huggingface, added: 2026-07-07) — Kokoro-TTS: small, high-quality open TTS â€” prime candidate for Homie's local voice output on all platforms; PyMasters tutorial on building a local voice assistant with it [link](https://huggingface.co/spaces/hexgrad/Kokoro-TTS)
- **sentence-transformers/all-MiniLM-L6-v2** (score: 9, source: huggingface, added: 2026-07-07) — Classic lightweight embedding model â€” ideal for a PyMasters semantic-search tutorial and as Homie's default local RAG embedder (ONNX runs fast on CPU) [link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **BAAI/bge-small-en-v1.5** (score: 9, source: huggingface, added: 2026-07-07) — Small, strong English embedding model â€” great Homie RAG candidate on low-end devices; PyMasters tutorial comparing bge vs MiniLM embeddings [link](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **Qwen/Qwen3-8B** (score: 9, source: huggingface, added: 2026-07-07) — Qwen3-8B is a top local-LLM choice â€” Homie's default 'capable' model tier (GGUF); PyMasters guide on running and quantizing 8B models [link](https://huggingface.co/Qwen/Qwen3-8B)
- **GLM 5.2 and the coming AI margin collapse** (score: 9, source: hackernews, added: 2026-07-07) — A 7 MB WASM embedding model is a near-perfect fit for Homie's lightweight local RAG (especially Android); also a fun 'embeddings in the browser' PyMasters tutorial [link](https://martinalderson.com/posts/the-upcoming-ai-margin-collapse-part-1-glm-5-2/)
- **cross-encoder/ms-marco-MiniLM-L6-v2** (score: 8, source: huggingface, added: 2026-07-07) — Cross-encoder reranker â€” tutorial on two-stage retrieval for PyMasters; Homie could use it to rerank RAG results locally for better answer quality [link](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** (score: 8, source: huggingface, added: 2026-07-07) — Multilingual embeddings â€” enables Homie RAG for non-English users; PyMasters tutorial on multilingual semantic search [link](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **sentence-transformers/all-mpnet-base-v2** (score: 8, source: huggingface, added: 2026-07-07) — Higher-quality embedding option â€” tutorial on quality-vs-speed embedding tradeoffs; Homie 'high accuracy' RAG mode with OpenVINO acceleration [link](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- **BAAI/bge-m3** (score: 8, source: huggingface, added: 2026-07-07) — Multilingual dense+sparse+ColBERT hybrid retrieval â€” advanced PyMasters retrieval tutorial; strong Homie embedder for multilingual document RAG [link](https://huggingface.co/BAAI/bge-m3)
- **openai/clip-vit-base-patch32** (score: 8, source: huggingface, added: 2026-07-07) — CLIP tutorial (zero-shot image classification, image search) for PyMasters; powers a Homie 'search my photos by text' plugin fully offline [link](https://huggingface.co/openai/clip-vit-base-patch32)
- **BAAI/bge-reranker-v2-m3** (score: 8, source: huggingface, added: 2026-07-07) — State-of-the-art multilingual reranker â€” Homie RAG quality upgrade; PyMasters tutorial on adding reranking to a retrieval pipeline [link](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- **nomic-ai/nomic-embed-text-v1.5** (score: 8, source: huggingface, added: 2026-07-07) — Nomic embed with resizable Matryoshka dims â€” Homie could trade embedding size vs quality per device; PyMasters article on Matryoshka embeddings [link](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- **google/gemma-4-26B-A4B-it** (score: 8, source: huggingface, added: 2026-07-07) — New Gemma 4 MoE vision-language model (only 4B active params) â€” PyMasters 'what's new' coverage plus tutorial; strong candidate for Homie's local multimodal (screenshot/image understanding) capability [link](https://huggingface.co/google/gemma-4-26B-A4B-it)
- **Program-as-Weights: A Programming Paradigm for Fuzzy Functions** (score: 8, source: arxiv, added: 2026-07-07) — 'Program-as-Weights' â€” replacing LLM API calls with small local fuzzy functions â€” is exactly Homie's local-first pitch (log alerting, JSON repair plugins) and a strong PyMasters tutorial [link](http://arxiv.org/abs/2607.02512v1)
- **A global workspace in language models** (score: 8, source: hackernews, added: 2026-07-07) — Context pruning directly cuts latency/memory for Homie's local RAG pipeline; also a practical 'optimize your RAG context' tutorial [link](https://www.anthropic.com/research/global-workspace)
- **AMD Ryzen AI Halo – $4k AI Dev Kit** (score: 8, source: hackernews, added: 2026-07-07) — Homie could integrate an OfficeCLI-style plugin so the assistant reads/edits Word/Excel files locally â€” high-value, privacy-preserving productivity feature [link](https://www.lttlabs.com/articles/2026/07/06/amd-ryzen-ai-halo)

### Prototyping (scored >= 7)

- **ReContext: Recursive Evidence Replay as LLM Harness for Long-Context Reasoning** (score: 7, source: arxiv, added: 2026-07-07) — Recursive evidence replay could improve Homie's RAG answers over long documents on small local models; also a solid advanced RAG article [link](http://arxiv.org/abs/2607.02509v1)
- **Companies hire more after AI adoption** (score: 7, source: hackernews, added: 2026-07-07) — 'Deterministic AI' argument maps to Homie's architecture (route simple tasks to rule-based code, save the LLM for hard ones) and makes a strong opinion/tutorial piece [link](https://ramp.com/data/heavy-ai-adopters-hire-more)

### Evaluating (scored >= 6)

- **timm/mobilenetv3_small_100.lamb_in1k** (score: 6, source: huggingface, added: 2026-07-07) — MobileNetV3 tutorial on efficient image classification; Homie could use it for a lightweight local photo-tagging plugin [link](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k)
- **BAAI/bge-large-en-v1.5** (score: 6, source: huggingface, added: 2026-07-07) — Large variant of bge â€” useful in an embedding benchmark tutorial; Homie option for desktop users with more RAM [link](https://huggingface.co/BAAI/bge-large-en-v1.5)
- **AI: The ROI Runway Could Be Long Outside the Tech Sector** (score: 6, source: hackernews, added: 2026-07-07) — AMD Ryzen AI Halo is target hardware for Homie's local LLM support â€” worth testing/optimizing for; could also fuel a 'local AI hardware buyer's guide' post [link](https://www.apollo.com/wealth/insights-news/insights/daily-spark/ai-the-roi-runway-could-be-long-outside-the-tech-sector)

### Discovered (new)

*No items yet.*
