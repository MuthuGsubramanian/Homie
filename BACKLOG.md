## Innovation Backlog

*Auto-maintained by the PyMasters AI Intelligence Pipeline.*
*Last updated: 2026-08-01*

### Ready to Build (scored >= 8, validated)

- **hexgrad/Kokoro-TTS** (score: 10, source: huggingface, added: 2026-08-01) — Kokoro TTS already powers a Homie voice plugin â€” promote/polish it, and write the companion PyMasters tutorial on local CPU TTS [link](https://huggingface.co/spaces/hexgrad/Kokoro-TTS)
- **sentence-transformers/all-MiniLM-L6-v2** (score: 9, source: huggingface, added: 2026-08-01) — Flagship lightweight embedding model â€” extend existing PyMasters semantic-search tutorials and use as Homie's default local RAG embedder (CPU-only, ONNX) [link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Qwen/Qwen3-0.6B** (score: 9, source: huggingface, added: 2026-08-01) — Sub-1B LLM already covered in PyMasters content â€” refresh the lesson, and it's an ideal low-RAM local model option for Homie [link](https://huggingface.co/Qwen/Qwen3-0.6B)
- **cross-encoder/ms-marco-MiniLM-L6-v2** (score: 8, source: huggingface, added: 2026-08-01) — Tutorial on two-stage retrieval (bi-encoder + cross-encoder), and Homie could use it as a local reranker to sharpen RAG answer quality [link](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- **BAAI/bge-small-en-v1.5** (score: 8, source: huggingface, added: 2026-08-01) — Compact English embedder â€” compare vs all-MiniLM in a PyMasters benchmark lesson; solid alternative embedding backend for Homie RAG [link](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** (score: 8, source: huggingface, added: 2026-08-01) — Multilingual sentence embeddings â€” tutorial on cross-language search, and enables Homie RAG for non-English users (aligns with PyMasters 8-language push) [link](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **BAAI/bge-m3** (score: 8, source: huggingface, added: 2026-08-01) — Tutorial on hybrid dense+sparse+multi-vector retrieval with bge-m3; strong multilingual embedding option for Homie's RAG pipeline [link](https://huggingface.co/BAAI/bge-m3)
- **sentence-transformers/all-mpnet-base-v2** (score: 8, source: huggingface, added: 2026-08-01) — Quality-vs-speed embedding comparison lesson (mpnet vs MiniLM); higher-accuracy embedding tier for Homie RAG on capable machines [link](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- **openai/clip-vit-base-patch32** (score: 8, source: huggingface, added: 2026-08-01) — Zero-shot image classification tutorial; Homie's existing CLIP plugin (image search/tagging) builds directly on this model [link](https://huggingface.co/openai/clip-vit-base-patch32)
- **BAAI/bge-reranker-v2-m3** (score: 8, source: huggingface, added: 2026-08-01) — Multilingual reranker â€” pair with bge-m3 in a retrieval-quality lesson; Homie could ship it as an optional RAG rerank stage [link](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- **Qwen/Qwen3-8B** (score: 8, source: huggingface, added: 2026-08-01) — Tutorial on running/quantizing an 8B local LLM; strong Homie local-LLM backend for users with 16GB+ RAM [link](https://huggingface.co/Qwen/Qwen3-8B)
- **intfloat/multilingual-e5-small** (score: 8, source: huggingface, added: 2026-08-01) — Small multilingual embedder â€” good subject for an e5 'query:/passage: prefix' gotcha lesson; efficient multilingual RAG option for Homie [link](https://huggingface.co/intfloat/multilingual-e5-small)
- **Rethinking Inference-Time Scaling in Local Computer-Use Agents: Failure Modes and Compute Tradeoffs** (score: 8, source: arxiv, added: 2026-08-01) — Directly about local computer-use agents under hardware constraints â€” actionable guidance for Homie's inference-time compute strategy [link](http://arxiv.org/abs/2607.28573v1)
- **Just brute force your embeddings** (score: 8, source: hackernews, added: 2026-08-01) — Brute-force embeddings validates PyMasters' recent MiniLM semantic-search tutorial (follow-up post material) and Homie's simple local RAG design without a vector DB [link](https://softwaredoug.com/blog/2026/07/29/just-brute-force-embeddings)

### Prototyping (scored >= 7)

- **timm/mobilenetv3_small_100.lamb_in1k** (score: 7, source: huggingface, added: 2026-08-01) — Tiny image classifier â€” edge-AI tutorial material, and already the basis of Homie's MobileNetV3 photo-tagger plugin [link](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k)
- **Show HN: How to build and self-host a code review agent** (score: 7, source: hackernews, added: 2026-08-01) — Self-hosted code-review agent is strong tutorial material and could become a Homie plugin for local code review [link](https://www.trytilde.ai/blog/how-to-build-code-review-agent)
- **Show HN: What should the GUI for AI agents look like?** (score: 7, source: hackernews, added: 2026-08-01) — Agent-GUI design discussion is directly useful input for Homie's assistant interface and plugin UX [link](https://marbleos.com/demo)

### Evaluating (scored >= 6)

- **Mitchins/acme-home-inbox** (score: 6, source: huggingface, added: 2026-08-01) — Synthetic home-inbox dataset â€” useful test corpus for Homie's local email/document RAG and personal-assistant plugin evaluation [link](https://huggingface.co/datasets/Mitchins/acme-home-inbox)
- **MixFrag: Fragility-Guided Mixed-Precision Post-Training Quantization for Vision Transformers** (score: 6, source: arxiv, added: 2026-08-01) — Mixed-precision quantization for edge devices: tutorial material for PyMasters and directly relevant to Homie's on-device model serving [link](http://arxiv.org/abs/2607.28589v1)
- **DualG-MRAG: Decoupling Macro-Reasoning and Micro-Matching for Multimodal Retrieval-Augmented Generation** (score: 6, source: arxiv, added: 2026-08-01) — Multimodal RAG with macro/micro matching could inform Homie's next-gen local RAG over mixed documents [link](http://arxiv.org/abs/2607.28580v1)
- **Sample More, Reflect Less: Self-Refine and Reflexion Lose to Repeated Sampling at Equal Token Cost, from 1.5B to 7B** (score: 6, source: arxiv, added: 2026-08-01) — Repeated sampling beating reflection at 1.5B-7B scale is a great lesson topic and directly applicable to how Homie budgets tokens on small local models [link](http://arxiv.org/abs/2607.28576v1)
- **Everyone is building LLM routers, we deprecated ours** (score: 6, source: hackernews, added: 2026-08-01) — Tutorial on when LLM routing helps vs. hurts; directly mirrors the multi-provider fallback chains both PyMasters' tutor and Homie use [link](https://manifest.build/blog/why-we-deprecated-our-llm-router/)

### Discovered (new)

*No items yet.*
