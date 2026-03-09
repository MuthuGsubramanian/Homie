from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional


class TransformersBackend:
    def __init__(self):
        self._model = None
        self._tokenizer = None

    def load(self, model_path: str | Path, **kwargs) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        path_str = str(model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(path_str, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(path_str, device_map="auto", trust_remote_code=True, **kwargs)

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7, stop: Optional[list[str]] = None) -> str:
        inputs = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], return_tensors="pt", add_generation_prompt=True,
        ).to(self._model.device)
        outputs = self._model.generate(inputs, max_new_tokens=max_tokens, temperature=temperature if temperature > 0 else None, do_sample=temperature > 0)
        new_tokens = outputs[0][inputs.shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def stream(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7, stop: Optional[list[str]] = None) -> Iterator[str]:
        from transformers import TextIteratorStreamer
        import threading
        inputs = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], return_tensors="pt", add_generation_prompt=True,
        ).to(self._model.device)
        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = {"input_ids": inputs, "max_new_tokens": max_tokens, "temperature": temperature if temperature > 0 else None, "do_sample": temperature > 0, "streamer": streamer}
        thread = threading.Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()
        yield from streamer
        thread.join()

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
