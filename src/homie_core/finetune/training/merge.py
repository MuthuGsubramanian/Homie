"""LoRA merge and GGUF quantization pipeline for Ollama deployment."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelMerger:
    """Merge LoRA adapters, quantize to GGUF, and import into Ollama."""

    def __init__(
        self,
        base_model: str,
        registry_name: str = "PyMasters/Homie",
    ):
        self.base_model = base_model
        self.registry_name = registry_name
        self._temp_dirs: list[Path] = []

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def merge_lora(
        self,
        base_model_path: str,
        adapter_path: str,
        output_path: str,
    ) -> bool:
        """Merge LoRA adapter into base model weights via peft.

        Loads the base model in float16, applies the LoRA adapter, calls
        ``merge_and_unload()``, and saves the full merged model + tokenizer
        to *output_path*.

        Returns ``True`` on success, ``False`` on failure.
        """
        output = Path(output_path)
        adapter = Path(adapter_path)

        if not adapter.exists():
            logger.error("Adapter path does not exist: %s", adapter_path)
            return False

        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            logger.error(
                "Required packages missing for LoRA merge (peft, transformers): %s",
                exc,
            )
            return False

        try:
            logger.info("Loading base model %s for merge...", base_model_path)
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )

            logger.info("Loading LoRA adapter from %s...", adapter_path)
            model = PeftModel.from_pretrained(model, adapter_path)

            logger.info("Merging adapter weights into base model...")
            model = model.merge_and_unload()

            output.mkdir(parents=True, exist_ok=True)
            logger.info("Saving merged model to %s...", output_path)
            model.save_pretrained(output_path)

            tokenizer = AutoTokenizer.from_pretrained(
                adapter_path, trust_remote_code=True
            )
            tokenizer.save_pretrained(output_path)

            logger.info("LoRA merge complete: %s", output_path)
            return True

        except torch.cuda.OutOfMemoryError:
            logger.error(
                "GPU OOM during merge. Try freeing VRAM or using CPU offload."
            )
            return False
        except Exception as exc:
            logger.error("LoRA merge failed: %s", exc, exc_info=True)
            return False
        finally:
            # Free GPU memory
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Quantize
    # ------------------------------------------------------------------

    def quantize(
        self,
        model_path: str,
        output_path: str,
        quant_type: str = "Q4_K_M",
    ) -> bool:
        """Quantize to GGUF via the ``llama-quantize`` binary."""
        result = subprocess.run(
            ["llama-quantize", model_path, output_path, quant_type],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        return result.returncode == 0

    def quantize_gguf(
        self,
        model_path: str,
        output_path: str,
        quant_type: str = "q4_k_m",
    ) -> bool:
        """Convert merged model to GGUF format.

        Tries llama.cpp ``convert_hf_to_gguf.py`` first. If the script is
        not found, falls back to Python-based conversion via the
        ``transformers`` + ``gguf`` packages.

        Returns ``True`` on success, ``False`` on failure.
        """
        model_dir = Path(model_path)
        if not model_dir.exists():
            logger.error("Model path does not exist: %s", model_path)
            return False

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Strategy 1: llama.cpp convert script
        if self._try_llama_cpp_convert(model_path, output_path, quant_type):
            return True

        # Strategy 2: Python-based conversion via gguf package
        if self._try_python_gguf_convert(model_path, output_path, quant_type):
            return True

        logger.error(
            "GGUF conversion failed. Install llama.cpp or the 'gguf' "
            "Python package: pip install gguf"
        )
        return False

    def _try_llama_cpp_convert(
        self,
        model_path: str,
        output_path: str,
        quant_type: str,
    ) -> bool:
        """Attempt conversion using llama.cpp's convert script + quantize binary."""
        # Try to find the convert script
        convert_scripts = [
            "convert_hf_to_gguf.py",
            "convert-hf-to-gguf.py",
        ]

        convert_cmd = None
        for script in convert_scripts:
            try:
                result = subprocess.run(
                    ["python", "-c", f"import importlib.util; print(importlib.util.find_spec('{script}'))"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

            # Try running it directly (may be on PATH via llama-cpp-python)
            try:
                result = subprocess.run(
                    [script, "--help"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    convert_cmd = script
                    break
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        if convert_cmd is None:
            logger.debug("llama.cpp convert script not found on PATH")
            return False

        # Step 1: Convert to f16 GGUF
        tmp_dir = Path(tempfile.mkdtemp(prefix="homie_gguf_"))
        self._temp_dirs.append(tmp_dir)
        f16_path = str(tmp_dir / "model-f16.gguf")

        try:
            logger.info("Converting to GGUF via %s...", convert_cmd)
            result = subprocess.run(
                [convert_cmd, model_path, "--outfile", f16_path, "--outtype", "f16"],
                capture_output=True,
                text=True,
                timeout=3600,
            )
            if result.returncode != 0:
                logger.warning(
                    "llama.cpp convert failed (rc=%d): %s",
                    result.returncode,
                    result.stderr[:500],
                )
                return False

            # Step 2: Quantize to target type
            logger.info("Quantizing to %s...", quant_type)
            quant_result = subprocess.run(
                ["llama-quantize", f16_path, output_path, quant_type.upper()],
                capture_output=True,
                text=True,
                timeout=3600,
            )
            if quant_result.returncode != 0:
                logger.warning(
                    "llama-quantize failed (rc=%d): %s",
                    quant_result.returncode,
                    quant_result.stderr[:500],
                )
                return False

            logger.info("GGUF quantization complete: %s", output_path)
            return True

        except subprocess.TimeoutExpired:
            logger.error("GGUF conversion timed out")
            return False
        except FileNotFoundError as exc:
            logger.debug("Binary not found: %s", exc)
            return False

    def _try_python_gguf_convert(
        self,
        model_path: str,
        output_path: str,
        quant_type: str,
    ) -> bool:
        """Attempt conversion using the Python ``gguf`` package and ``transformers``."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            logger.debug("transformers not available for Python GGUF conversion")
            return False

        try:
            # llama-cpp-python ships with a conversion utility
            from llama_cpp import llama_model_quantize_default_params  # noqa: F401

            logger.debug("llama-cpp-python available but no direct convert API")
        except ImportError:
            pass

        # Try using the gguf package's built-in conversion if available
        try:
            import gguf  # noqa: F401
        except ImportError:
            logger.debug("gguf package not installed")
            return False

        try:
            # The gguf package exposes convert_hf_to_gguf as a module entry point
            logger.info("Attempting Python-based GGUF conversion...")
            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "gguf.scripts.convert_hf_to_gguf",
                    model_path,
                    "--outfile",
                    output_path,
                    "--outtype",
                    quant_type.lower(),
                ],
                capture_output=True,
                text=True,
                timeout=3600,
            )
            if result.returncode == 0:
                logger.info("Python GGUF conversion complete: %s", output_path)
                return True

            logger.warning(
                "Python gguf module conversion failed (rc=%d): %s",
                result.returncode,
                result.stderr[:500],
            )
            return False

        except subprocess.TimeoutExpired:
            logger.error("Python GGUF conversion timed out")
            return False
        except Exception as exc:
            logger.debug("Python GGUF conversion error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def merge_and_quantize(
        self,
        base_model_path: str,
        adapter_path: str,
        gguf_output_path: str,
        quant_type: str = "q4_k_m",
        merged_model_dir: str | None = None,
    ) -> bool:
        """Full pipeline: merge LoRA adapter into base model then quantize to GGUF.

        Parameters
        ----------
        base_model_path:
            Path or HF identifier for the base (unmodified) model.
        adapter_path:
            Directory containing the LoRA adapter weights.
        gguf_output_path:
            Destination path for the final quantized ``.gguf`` file.
        quant_type:
            GGUF quantization type (default ``q4_k_m``).
        merged_model_dir:
            Optional directory for the intermediate merged model. When
            *None* a temporary directory is created and cleaned up
            automatically.

        Returns ``True`` when the full pipeline succeeds.
        """
        cleanup_merged = merged_model_dir is None
        if merged_model_dir is None:
            merged_model_dir = tempfile.mkdtemp(prefix="homie_merged_")
            self._temp_dirs.append(Path(merged_model_dir))

        try:
            # Step 1: Merge LoRA
            logger.info("Step 1/2: Merging LoRA adapter...")
            if not self.merge_lora(base_model_path, adapter_path, merged_model_dir):
                logger.error("merge_and_quantize aborted: LoRA merge failed")
                return False

            # Step 2: Convert to GGUF
            logger.info("Step 2/2: Converting to GGUF (%s)...", quant_type)
            if not self.quantize_gguf(merged_model_dir, gguf_output_path, quant_type):
                logger.error("merge_and_quantize aborted: GGUF quantization failed")
                return False

            logger.info(
                "merge_and_quantize complete: %s -> %s",
                adapter_path,
                gguf_output_path,
            )
            return True

        finally:
            if cleanup_merged:
                self._cleanup_dir(Path(merged_model_dir))

    # ------------------------------------------------------------------
    # Ollama helpers
    # ------------------------------------------------------------------

    def build_modelfile(self, gguf_path: str, system_prompt: str = "") -> str:
        """Generate Ollama Modelfile content."""
        lines = [f"FROM {gguf_path}"]
        if system_prompt:
            lines.append(f'SYSTEM """{system_prompt}"""')
        return "\n".join(lines)

    def import_to_ollama(self, modelfile_path: str) -> bool:
        """Import as staging candidate: ``ollama create registry:candidate``."""
        result = subprocess.run(
            [
                "ollama",
                "create",
                f"{self.registry_name}:candidate",
                "-f",
                modelfile_path,
            ],
            capture_output=True,
            text=True,
            timeout=600,
            encoding="utf-8",
            errors="replace",
        )
        return result.returncode == 0

    def promote_candidate(self) -> bool:
        """Atomic swap: ``ollama cp registry:candidate registry:latest``."""
        result = subprocess.run(
            [
                "ollama",
                "cp",
                f"{self.registry_name}:candidate",
                f"{self.registry_name}:latest",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            encoding="utf-8",
            errors="replace",
        )
        return result.returncode == 0

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Remove all temporary directories created during the pipeline."""
        for tmp in self._temp_dirs:
            self._cleanup_dir(tmp)
        self._temp_dirs.clear()

    @staticmethod
    def _cleanup_dir(path: Path) -> None:
        """Safely remove a directory tree."""
        try:
            if path.exists():
                shutil.rmtree(path)
                logger.debug("Cleaned up temp directory: %s", path)
        except OSError as exc:
            logger.warning("Failed to clean up %s: %s", path, exc)
