## Innovation Backlog

*Auto-maintained by the PyMasters AI Intelligence Pipeline.*
*Last updated: 2026-07-31*

### Ready to Build (scored >= 8, validated)

- **hexgrad/Kokoro-TTS** (score: 10, source: huggingface, added: 2026-07-31) — Kokoro TTS plugin already shipped for Homie voice output â€” promote it; PyMasters lesson on lightweight local TTS [link](https://huggingface.co/spaces/hexgrad/Kokoro-TTS)
- **sentence-transformers/all-MiniLM-L6-v2** (score: 9, source: huggingface, added: 2026-07-31) — Already powers PyMasters semantic search â€” extend the existing tutorial/curriculum; also ideal as Homie's default local embedding model for RAG [link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Qwen/Qwen3-0.6B** (score: 9, source: huggingface, added: 2026-07-31) — Already in the sub-1B curriculum â€” expand with fine-tuning content; perfect always-on local LLM for Homie on modest hardware [link](https://huggingface.co/Qwen/Qwen3-0.6B)
- **openai/clip-vit-base-patch32** (score: 9, source: huggingface, added: 2026-07-31) — CLIP plugin already shipped for Homie (local image search/tagging) â€” pair with a PyMasters zero-shot classification lesson [link](https://huggingface.co/openai/clip-vit-base-patch32)
- **Qwen/Qwen3-8B** (score: 9, source: huggingface, added: 2026-07-31) — Flagship local-LLM option for Homie on GPU machines; PyMasters tutorial on running/quantizing Qwen3-8B locally [link](https://huggingface.co/Qwen/Qwen3-8B)
- **cross-encoder/ms-marco-MiniLM-L6-v2** (score: 8, source: huggingface, added: 2026-07-31) — Tutorial on two-stage retrieval (embed then rerank); Homie could use it as a lightweight reranker in its RAG pipeline [link](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- **BAAI/bge-small-en-v1.5** (score: 8, source: huggingface, added: 2026-07-31) — Small CPU-friendly embedder â€” compare vs all-MiniLM in a lesson; strong candidate for Homie's local RAG index [link](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **BAAI/bge-m3** (score: 8, source: huggingface, added: 2026-07-31) — Tutorial on hybrid dense+sparse+multilingual retrieval; Homie could adopt it for higher-quality multilingual RAG [link](https://huggingface.co/BAAI/bge-m3)
- **BAAI/bge-reranker-v2-m3** (score: 8, source: huggingface, added: 2026-07-31) — Advanced RAG tutorial on reranking; Homie could add it as an optional quality-boost stage over bge-m3 retrieval [link](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- **timm/mobilenetv3_small_100.lamb_in1k** (score: 8, source: huggingface, added: 2026-07-31) — MobileNetV3 tagger plugin already built for Homie â€” reinforce with a PyMasters edge/efficient-CNN lesson [link](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k)
- **intfloat/multilingual-e5-small** (score: 8, source: huggingface, added: 2026-07-31) — Tiny multilingual embedder â€” good Homie default for non-English users; comparison lesson vs bge-small/MiniLM [link](https://huggingface.co/intfloat/multilingual-e5-small)
- **DenseOn with the LateOn: Fully Open Dense and Late-Interaction Models for Multilingual, Long-Context, and Code Search** (score: 8, source: arxiv, added: 2026-07-31) — Open dense + late-interaction retrieval models: a PyMasters tutorial on modern retrieval, and candidate local embedding models for Homie's RAG [link](http://arxiv.org/abs/2607.27178v1)
- **Show HN: Noisegate – a differential-privacy gateway for untrusted AI agents** (score: 8, source: hackernews, added: 2026-07-31) — A differential-privacy gateway for untrusted agents aligns directly with Homie's privacy-first design; evaluate as a built-in privacy layer or plugin [link](https://github.com/yashmahajan10/llm-differential-privacy-gateway)

### Prototyping (scored >= 7)

- **sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2** (score: 7, source: huggingface, added: 2026-07-31) — Multilingual embeddings tutorial ties into PyMasters' 8-language push; enables Homie RAG over non-English documents [link](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- **sentence-transformers/all-mpnet-base-v2** (score: 7, source: huggingface, added: 2026-07-31) — Quality-vs-speed embedding benchmark lesson (mpnet vs MiniLM); Homie 'high quality' embedding option on capable devices [link](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- **MindForge: Teaching Small Language Models Whole-Life-Cycle Software Engineering via Source-Free Program Synthesis** (score: 7, source: arxiv, added: 2026-07-31) — Small-model software engineering fits both: a PyMasters lesson on training small coding models and evidence for Homie shipping capable local SLMs [link](http://arxiv.org/abs/2607.27146v1)

### Evaluating (scored >= 6)

- **OmegaUse-OfficeVal: Benchmarking LLM Agents on Long-Horizon Office-Suite Tasks with Economic Grounding** (score: 6, source: arxiv, added: 2026-07-31) — Office-suite agent benchmark; informs Homie's roadmap for local document/office task automation and a PyMasters piece on agent cost-effectiveness [link](http://arxiv.org/abs/2607.27155v1)
- **Show HN: Distilling DeepSeek into GPT-OSS doesn't transfer censorship. Try it** (score: 6, source: hackernews, added: 2026-07-31) — Tutorial on model distillation and what transfers (capabilities vs alignment/censorship); the distilled open model is also a candidate for Homie's local LLM lineup [link](https://www.ctgt.ai/research/distillation-censorship-transfer)

### Discovered (new)

*No items yet.*
