## Innovation Backlog

*Auto-maintained by the PyMasters AI Intelligence Pipeline.*
*Last updated: 2026-07-29*

### Ready to Build (scored >= 8, validated)

- **hexgrad/Kokoro-TTS** (score: 10, source: huggingface, added: 2026-07-29) — Kokoro TTS plugin already shipped for Homie (82M, CPU-friendly voice output) and content published â€” highest-fit item; keep iterating on voice UX [link](https://huggingface.co/spaces/hexgrad/Kokoro-TTS)
- **sentence-transformers/all-MiniLM-L6-v2** (score: 9, source: huggingface, added: 2026-07-29) — Core embedding model for semantic search tutorials (curriculum already exists â€” refresh/repromote) and Homie's default local RAG embedder [link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Qwen/Qwen3-0.6B** (score: 9, source: huggingface, added: 2026-07-29) — Sub-1B local LLM already covered (LoRA lesson) â€” Homie can ship it as the default ultra-light on-device model for low-RAM machines [link](https://huggingface.co/Qwen/Qwen3-0.6B)
- **Qwen/Qwen3-8B** (score: 9, source: huggingface, added: 2026-07-29) — Qwen3-8B as Homie's mid-tier local model (GGUF quantization guide) plus a PyMasters lesson on running 8B models locally with llama.cpp [link](https://huggingface.co/Qwen/Qwen3-8B)
- **cross-encoder/ms-marco-MiniLM-L6-v2** (score: 8, source: huggingface, added: 2026-07-29) — Two-stage retrieval (bi-encoder + cross-encoder rerank) lesson exists; Homie can use it to rerank RAG results locally on CPU [link](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- **BAAI/bge-small-en-v1.5** (score: 8, source: huggingface, added: 2026-07-29) — Compact English embedder â€” benchmark-vs-MiniLM tutorial and a lightweight alternative embedder option in Homie's RAG [link](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **BAAI/bge-m3** (score: 8, source: huggingface, added: 2026-07-29) — Lesson on hybrid dense+sparse+multi-vector retrieval exists â€” extend with a hands-on multilingual RAG project; heavier but viable Homie embedder [link](https://huggingface.co/BAAI/bge-m3)
- **openai/clip-vit-base-patch32** (score: 8, source: huggingface, added: 2026-07-29) — CLIP multimodal lesson and Homie zero-shot image classifier plugin both already shipped â€” promote and iterate rather than rebuild [link](https://huggingface.co/openai/clip-vit-base-patch32)
- **timm/mobilenetv3_small_100.lamb_in1k** (score: 8, source: huggingface, added: 2026-07-29) — Edge image classification lesson; Homie photo-tagger plugin already built on this â€” ship/iterate it [link](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k)
- **BAAI/bge-reranker-v2-m3** (score: 8, source: huggingface, added: 2026-07-29) — Multilingual reranker for the two-stage RAG lesson series; strong candidate for Homie's local rerank stage [link](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- **intfloat/multilingual-e5-small** (score: 8, source: huggingface, added: 2026-07-29) — Small multilingual E5 embedder â€” good Homie option for non-English RAG on CPU; comparison tutorial vs BGE/MiniLM [link](https://huggingface.co/intfloat/multilingual-e5-small)

### Prototyping (scored >= 7)

- **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** (score: 7, source: huggingface, added: 2026-07-29) — Multilingual semantic search tutorial (fits PyMasters' 8-language angle); Homie plugin for non-English document RAG [link](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **sentence-transformers/all-mpnet-base-v2** (score: 7, source: huggingface, added: 2026-07-29) — Quality-vs-speed embedding comparison tutorial (mpnet vs MiniLM); higher-accuracy RAG embedder tier in Homie [link](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- **A corrective agentic hybrid RAG and an operations-grounded evaluation for a scientific facility** (score: 7, source: arxiv, added: 2026-07-29) — Corrective agentic hybrid RAG patterns fit Homie's RAG pipeline directly and would make a strong advanced-RAG lesson [link](http://arxiv.org/abs/2607.24663v1)
- **Hubbele: Open-source notetaking app for you and your agents** (score: 7, source: hackernews, added: 2026-07-29) — Open-source agent-friendly notetaking app â€” Homie could integrate it (or its format) as a local notes/memory plugin [link](https://www.hubble.md/)

### Evaluating (scored >= 6)

- **Eviction as Estimation: A Fixed-Lag Smoothing View of Test-Time Memory, and When Measuring Beats Accumulating** (score: 6, source: arxiv, added: 2026-07-29) — KV-cache eviction strategies (StreamingLLM/H2O/SnapKV framing) could improve Homie's bounded-memory local LLM sessions [link](http://arxiv.org/abs/2607.24667v1)

### Discovered (new)

*No items yet.*
