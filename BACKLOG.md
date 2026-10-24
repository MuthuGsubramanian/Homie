## Innovation Backlog

*Auto-maintained by the PyMasters AI Intelligence Pipeline.*
*Last updated: 2026-10-25*

### Ready to Build (scored >= 8, validated)

- **hexgrad/Kokoro-TTS** (score: 10, source: huggingface, added: 2026-10-25) — Homie kokoro_tts plugin already built; finish it as the default offline voice. PyMasters Short and explainer scripted, publish them [link](https://huggingface.co/spaces/hexgrad/Kokoro-TTS)
- **CowAgent – Open-Source Super AI Assistant** (score: 9, source: hackernews, added: 2026-10-25) — Direct competitor/reference for Homie: study CowAgent's plugin and local-assistant architecture for feature parity and differentiation [link](https://cowagent.ai/en/)
- **mteb/leaderboard** (score: 8, source: huggingface, added: 2026-10-25) — Lesson on choosing embedding models via MTEB for RAG (already drafted); Homie could auto-recommend an embedding model for its RAG stack from MTEB scores [link](https://huggingface.co/spaces/mteb/leaderboard)
- **black-forest-labs/FLUX.1-schnell** (score: 8, source: huggingface, added: 2026-10-25) — Fast local text-to-image tutorial in 4 steps; Homie flux_image_gen plugin already targets schnell, ship and document it [link](https://huggingface.co/spaces/black-forest-labs/FLUX.1-schnell)
- **EpiSEM-Bench: Evaluating Safety and Evidence Recovery in LLM Agents under Missing Evidence** (score: 8, source: arxiv, added: 2026-10-25) — Directly relevant to Homie tool-using agent safety: treat missing evidence from a tool as a reason to pause, and add a benchmark-style test suite [link](http://arxiv.org/abs/2610.31650v1)
- **A practical guide to AI image generation, from diffusion to agents** (score: 8, source: hackernews, added: 2026-10-25) — Course module tracing image generation from diffusion to agentic pipelines; Homie's flux_image_gen plugin could add agentic prompt refinement [link](https://danielball.com/notes/ai_image_generation.html)
- **Dessa: Conversation Model trained only on speech** (score: 8, source: hackernews, added: 2026-10-25) — Speech-only conversation model is a strong candidate for Homie's offline voice interaction plugin; PyMasters could cover speech-to-speech model architecture [link](https://block.xyz/dessa)

### Prototyping (scored >= 7)

- **Low-Bit N-Gram Embeddings for Language Models** (score: 7, source: arxiv, added: 2026-10-25) — Low-bit n-gram embeddings shrink local model memory; Homie should track models adopting it, PyMasters can explain quantized embedding tables [link](http://arxiv.org/abs/2610.31646v1)
- **BeNice! Closing Looping's Inference-Memory Gap via KV-Sharing** (score: 7, source: arxiv, added: 2026-10-25) — Explains essay on KV-cache growth in looped transformers; directly relevant to Homie's on-device memory budget when running small recurrent-depth models [link](http://arxiv.org/abs/2610.31633v1)
- **How Ground Truth Shapes Retrieval Scores: Evidence-Aware Evaluation of a Retrieval-Augmented Generation System for Breast Lesions** (score: 7, source: arxiv, added: 2026-10-25) — Tutorial on evidence-aware RAG retrieval evaluation (annotation coverage bias); Homie's RAG plugin could adopt the same eval protocol for local document search quality [link](http://arxiv.org/abs/2610.31606v1)
- **Show HN: Corgi, a helpful {agent, terminal} multiplexer** (score: 7, source: hackernews, added: 2026-10-25) — Agent/terminal multiplexer pattern could inspire a Homie terminal-agent plugin for Linux/Mac power users [link](https://corgi.computer/)

### Evaluating (scored >= 6)

- **open-llm-leaderboard/open_llm_leaderboard** (score: 6, source: huggingface, added: 2026-10-25) — Lesson on reading LLM benchmarks critically; Homie could surface leaderboard data when helping users pick a local model [link](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
- **black-forest-labs/FLUX.1-dev** (score: 6, source: huggingface, added: 2026-10-25) — Tutorial on running FLUX.1-dev locally with diffusers and quantization; Homie image plugin could offer dev as a quality tier above schnell [link](https://huggingface.co/spaces/black-forest-labs/FLUX.1-dev)
- **facebook/MusicGen** (score: 6, source: huggingface, added: 2026-10-25) — Tutorial on audio generation with transformers; Homie plugin for offline background music or jingle generation [link](https://huggingface.co/spaces/facebook/MusicGen)
- **mrfakename/Z-Image-Turbo** (score: 6, source: huggingface, added: 2026-10-25) — Turbo image model candidate for Homie's low-VRAM image generation plugin; benchmark against FLUX schnell [link](https://huggingface.co/spaces/mrfakename/Z-Image-Turbo)
- **A whirlwind tour of systemd-nspawn containers (2025)** (score: 6, source: hackernews, added: 2026-10-25) — systemd-nspawn could sandbox Homie plugins on Linux; PyMasters could reference it in a sandboxing lesson [link](https://quantum5.ca/2025/03/22/whirlwind-tour-of-systemd-nspawn-containers/)

### Discovered (new)

*No items yet.*
