## Innovation Backlog

*Auto-maintained by the PyMasters AI Intelligence Pipeline.*
*Last updated: 2026-08-02*

### Ready to Build (scored >= 8, validated)

- **hexgrad/Kokoro-TTS** (score: 10, source: huggingface, added: 2026-08-02) — Kokoro-TTS (82M, CPU-capable) â€” Homie's local voice-output plugin is already built; PyMasters content (Short, explainer, blog) shipped and can extend to a full lesson [link](https://huggingface.co/spaces/hexgrad/Kokoro-TTS)
- **sentence-transformers/all-MiniLM-L6-v2** (score: 9, source: huggingface, added: 2026-08-02) — Canonical lightweight embedding model â€” PyMasters tutorial on sentence embeddings and semantic search (already in curriculum); powers Homie's local RAG retrieval on CPU [link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Qwen/Qwen3-0.6B** (score: 9, source: huggingface, added: 2026-08-02) — Sub-1B local LLM (already featured in content) â€” 'run an LLM on your laptop' tutorial; ideal Homie fallback model for low-RAM devices and Android [link](https://huggingface.co/Qwen/Qwen3-0.6B)
- **cross-encoder/ms-marco-MiniLM-L6-v2** (score: 8, source: huggingface, added: 2026-08-02) — Tutorial on two-stage retrieval (bi-encoder recall + cross-encoder rerank); Homie can use it to rerank RAG results locally for sharper answers [link](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- **BAAI/bge-small-en-v1.5** (score: 8, source: huggingface, added: 2026-08-02) — Compact English embedding model â€” comparison lesson vs MiniLM on quality/speed; strong candidate as Homie's default RAG embedder (small, ONNX-ready) [link](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** (score: 8, source: huggingface, added: 2026-08-02) — Multilingual embeddings tutorial (cross-language search demo); enables Homie RAG for non-English users' documents [link](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **openai/clip-vit-base-patch32** (score: 8, source: huggingface, added: 2026-08-02) — Zero-shot image classification tutorial; already powering Homie's CLIP plugin â€” extend to local photo semantic search [link](https://huggingface.co/openai/clip-vit-base-patch32)
- **BAAI/bge-reranker-v2-m3** (score: 8, source: huggingface, added: 2026-08-02) — State-of-the-art multilingual reranker â€” Homie RAG quality upgrade as an optional plugin; PyMasters section in the retrieval-pipeline track [link](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- **intfloat/multilingual-e5-small** (score: 8, source: huggingface, added: 2026-08-02) — Small multilingual embedder with ONNX/OpenVINO exports â€” strong Homie candidate for multilingual RAG on modest hardware; comparison content for PyMasters [link](https://huggingface.co/intfloat/multilingual-e5-small)
- **Qwen/Qwen3-8B** (score: 8, source: huggingface, added: 2026-08-02) — Tutorial on running an 8B model locally with quantization (GGUF/llama.cpp); Homie's 'power tier' local model for machines with 16GB+ RAM [link](https://huggingface.co/Qwen/Qwen3-8B)

### Prototyping (scored >= 7)

- **BAAI/bge-m3** (score: 7, source: huggingface, added: 2026-08-02) — Tutorial on hybrid dense+sparse+multi-vector retrieval with BGE-M3; Homie could offer it as a higher-quality multilingual RAG option on capable machines [link](https://huggingface.co/BAAI/bge-m3)
- **sentence-transformers/all-mpnet-base-v2** (score: 7, source: huggingface, added: 2026-08-02) — Quality-tier embedding model â€” pairs with MiniLM in a 'speed vs quality embeddings' lesson; Homie option for users who prioritize retrieval quality [link](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- **timm/mobilenetv3_small_100.lamb_in1k** (score: 7, source: huggingface, added: 2026-08-02) — Tiny image classifier already used in Homie's image-tagger plugin; PyMasters could add an 'efficient CNNs on CPU' lesson with timm [link](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k)
- **mteb/leaderboard** (score: 7, source: huggingface, added: 2026-08-02) — MTEB leaderboard walkthrough: 'how to choose an embedding model' lesson â€” directly informs which embedders Homie ships and PyMasters teaches [link](https://huggingface.co/spaces/mteb/leaderboard)

### Evaluating (scored >= 6)

*No items yet.*

### Discovered (new)

*No items yet.*
