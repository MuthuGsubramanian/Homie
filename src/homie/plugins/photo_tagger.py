"""On-device photo tagging for HOMIE using MobileNetV3 (timm).

Model: ``timm/mobilenetv3_small_100.lamb_in1k`` -- an ImageNet-1k classifier
with ~2.5M parameters and a ~10 MB safetensors file. It runs comfortably on a
laptop CPU (tens of images per second), which makes it a good fit for tagging
a local photo library without sending a single byte off the machine.

What the plugin does
--------------------
* Walks configured photo folders, classifies every image and stores the top-k
  ImageNet labels (with confidence) in a local SQLite database under
  ``~/.homie/photo_tags.sqlite``.
* Skips images whose size/mtime have not changed since the last run, so
  re-tagging a big library is cheap.
* Answers "find my photos of <thing>" queries by label substring match, so
  HOMIE can hand photo paths back to the user (or to a downstream RAG step).

Privacy and safety
------------------
* No network access by default. Weights are loaded from the local Hugging
  Face cache only. Set ``plugins.photo_tagger.allow_download: true`` to let
  the plugin fetch the weights once; afterwards it works fully offline.
* Heavy dependencies (torch, timm, Pillow, huggingface_hub) are imported
  lazily. HOMIE boots without them and the commands return an install hint.
* Only reads image files; never moves, renames or writes next to your photos.

Config (homie.config.yaml)::

    plugins:
      photo_tagger:
        enabled: true
        model: timm/mobilenetv3_small_100.lamb_in1k
        allow_download: false     # opt-in one-time weight download
        device: cpu               # cpu | cuda | mps
        top_k: 5
        min_confidence: 0.10
        batch_size: 16
        db_path: ~/.homie/photo_tags.sqlite
        folders:
          - ~/Pictures

REPL integration: in ``homie/homie.py::_handle_meta_command`` add, before the
"Unknown command" fallback::

    elif lower.startswith(":photos"):
        from homie.plugins import photo_tagger
        photo_tagger.handle_meta_command(cmd)

and call ``photo_tagger.activate(cfg)`` once after ``load_config`` (plus
``photo_tagger.deactivate()`` on exit). Commands::

    :photos tag [folder] [--force]  tag configured folders (or one folder)
    :photos find <query>            list photos whose labels match the query
    :photos tags <path>             show stored tags for one image
    :photos stats                   database summary
    :photos help                    this list

Install (optional, only needed to actually tag)::

    pip install timm torch pillow huggingface_hub
"""

from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from homie.config import HomieConfig, cfg_get

LOG = logging.getLogger(__name__)

PLUGIN_NAME = "photo_tagger"
DEFAULT_MODEL = "timm/mobilenetv3_small_100.lamb_in1k"
DEFAULT_DB_PATH = Path.home() / ".homie" / "photo_tags.sqlite"
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})
INSTALL_HINT = "pip install timm torch pillow huggingface_hub"

SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
    path TEXT PRIMARY KEY,
    mtime REAL NOT NULL,
    size INTEGER NOT NULL,
    model TEXT NOT NULL,
    tagged_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tags (
    path TEXT NOT NULL,
    label TEXT NOT NULL,
    score REAL NOT NULL,
    FOREIGN KEY(path) REFERENCES photos(path) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_tags_label ON tags(label);
CREATE INDEX IF NOT EXISTS idx_tags_path ON tags(path);
"""


class PhotoTaggerError(Exception):
    """Raised when tagging cannot proceed (missing deps, weights not cached, bad path)."""


@dataclass
class Tag:
    """One predicted label with its softmax confidence."""

    label: str
    score: float


@dataclass
class TagResult:
    """Tags for a single image, or an error string if it could not be read."""

    path: Path
    tags: List[Tag] = field(default_factory=list)
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "tags": [{"label": t.label, "score": round(t.score, 4)} for t in self.tags],
            "error": self.error,
        }


@dataclass
class TaggerSettings:
    """Plugin settings resolved from ``plugins.photo_tagger`` in the HOMIE config."""

    model: str = DEFAULT_MODEL
    allow_download: bool = False
    device: str = "cpu"
    top_k: int = 5
    min_confidence: float = 0.10
    batch_size: int = 16
    db_path: Path = DEFAULT_DB_PATH
    folders: List[Path] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: Optional[HomieConfig]) -> "TaggerSettings":
        if cfg is None:
            return cls()

        def get(key: str, default: Any) -> Any:
            return cfg_get(cfg, "plugins", PLUGIN_NAME, key, default=default)

        folders = [Path(os.path.expanduser(str(p))) for p in (get("folders", []) or [])]
        return cls(
            model=str(get("model", DEFAULT_MODEL)),
            allow_download=bool(get("allow_download", False)),
            device=str(get("device", "cpu")),
            top_k=max(1, int(get("top_k", 5))),
            min_confidence=float(get("min_confidence", 0.10)),
            batch_size=max(1, int(get("batch_size", 16))),
            db_path=Path(os.path.expanduser(str(get("db_path", DEFAULT_DB_PATH)))),
            folders=folders,
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def iter_images(folder: Path, recursive: bool = True) -> Iterable[Path]:
    """Yield image files under ``folder`` (sorted for deterministic runs)."""
    if not folder.is_dir():
        raise PhotoTaggerError(f"Not a directory: {folder}")
    walker = folder.rglob("*") if recursive else folder.glob("*")
    for p in sorted(walker):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            yield p


class MobileNetV3PhotoTagger:
    """Tags local photos with ImageNet labels and persists them to SQLite.

    The model is loaded on first use (``classify`` / ``tag_folder``), never at
    import or ``activate`` time, so HOMIE's startup cost is unchanged.
    """

    def __init__(self, settings: Optional[TaggerSettings] = None) -> None:
        self.settings = settings or TaggerSettings()
        self._model: Any = None
        self._transform: Any = None
        self._labels: List[str] = []
        self._device: str = "cpu"
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------ storage
    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            self.settings.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.settings.db_path))
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(SCHEMA)
            self._conn = conn
        return self._conn

    def _is_unchanged(self, path: Path) -> bool:
        """True when the file was already tagged by this model and has not changed."""
        try:
            st = path.stat()
        except OSError:
            return False
        row = self._db().execute(
            "SELECT mtime, size, model FROM photos WHERE path = ?", (str(path),)
        ).fetchone()
        return bool(row) and row[0] == st.st_mtime and row[1] == st.st_size and row[2] == self.settings.model

    def _persist(self, results: Sequence[TagResult]) -> None:
        conn = self._db()
        now = _utc_now()
        with conn:
            for res in results:
                if res.error:
                    continue
                try:
                    st = res.path.stat()
                except OSError:
                    continue
                key = str(res.path)
                conn.execute("DELETE FROM tags WHERE path = ?", (key,))
                conn.execute(
                    "INSERT OR REPLACE INTO photos(path, mtime, size, model, tagged_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (key, st.st_mtime, st.st_size, self.settings.model, now),
                )
                conn.executemany(
                    "INSERT INTO tags(path, label, score) VALUES (?, ?, ?)",
                    [(key, t.label, t.score) for t in res.tags],
                )

    def close(self) -> None:
        """Release the model and the database handle."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        self._model = None
        self._transform = None
        self._labels = []

    # -------------------------------------------------------------------- model
    def _resolve_weights(self) -> Path:
        """Locate the weight file, from the local cache unless download is allowed."""
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise PhotoTaggerError(f"huggingface_hub missing. {INSTALL_HINT}") from exc

        repo = self.settings.model
        offline = not self.settings.allow_download
        try:
            local_dir = snapshot_download(
                repo,
                local_files_only=offline,
                allow_patterns=["*.safetensors", "*.json"],
            )
        except Exception as exc:  # noqa: BLE001 - the hub raises several error types
            if offline:
                raise PhotoTaggerError(
                    f"Weights for {repo} are not in the local Hugging Face cache and "
                    "plugins.photo_tagger.allow_download is false. Set it to true for a "
                    "one-time ~10 MB download, then switch it back off."
                ) from exc
            raise PhotoTaggerError(f"Could not download weights for {repo}: {exc}") from exc

        for name in ("model.safetensors", "pytorch_model.bin"):
            candidate = Path(local_dir) / name
            if candidate.exists():
                return candidate
        raise PhotoTaggerError(f"No weight file found in {local_dir}")

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import timm
            import torch
            from timm.data import ImageNetInfo, create_transform, resolve_data_config
        except ImportError as exc:
            raise PhotoTaggerError(f"timm/torch missing. {INSTALL_HINT}") from exc

        weights = self._resolve_weights()
        arch = self.settings.model.split("/")[-1]  # e.g. mobilenetv3_small_100.lamb_in1k
        LOG.info("photo_tagger: loading %s from %s", arch, weights)
        model = timm.create_model(
            arch,
            pretrained=True,
            pretrained_cfg_overlay={"file": str(weights)},
        )
        model.eval()

        device = self.settings.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            LOG.warning("photo_tagger: cuda requested but unavailable, using cpu")
            device = "cpu"
        if device == "mps":
            mps = getattr(torch.backends, "mps", None)
            if mps is None or not mps.is_available():
                LOG.warning("photo_tagger: mps requested but unavailable, using cpu")
                device = "cpu"
        model.to(device)

        data_cfg = resolve_data_config({}, model=model)
        self._transform = create_transform(**data_cfg, is_training=False)
        info = ImageNetInfo()
        self._labels = [info.index_to_description(i, detailed=False) for i in range(model.num_classes)]
        self._device = device
        self._model = model

    def classify(self, paths: Sequence[Path]) -> List[TagResult]:
        """Classify images and return top-k tags per image (nothing is persisted)."""
        if not paths:
            return []
        self._ensure_model()
        import torch
        from PIL import Image

        s = self.settings
        results: List[TagResult] = []
        for start in range(0, len(paths), s.batch_size):
            chunk = paths[start : start + s.batch_size]
            tensors: List[Any] = []
            kept: List[Path] = []
            for p in chunk:
                try:
                    with Image.open(p) as im:
                        tensors.append(self._transform(im.convert("RGB")))
                    kept.append(p)
                except Exception as exc:  # noqa: BLE001 - corrupt/unsupported file
                    results.append(TagResult(path=p, error=f"unreadable image: {exc}"))
            if not tensors:
                continue
            batch = torch.stack(tensors).to(self._device)
            with torch.inference_mode():
                probs = self._model(batch).softmax(dim=-1)
            k = min(s.top_k, probs.shape[-1])
            top_p, top_i = probs.topk(k, dim=-1)
            for p, scores, idxs in zip(kept, top_p.tolist(), top_i.tolist()):
                tags = [
                    Tag(self._labels[i], sc)
                    for sc, i in zip(scores, idxs)
                    if sc >= s.min_confidence
                ]
                if not tags:  # always keep the best guess so the photo stays searchable
                    tags = [Tag(self._labels[idxs[0]], scores[0])]
                results.append(TagResult(path=p, tags=tags))
        return results

    # ------------------------------------------------------------------ actions
    def tag_folder(self, folder: Path, recursive: bool = True, force: bool = False) -> Dict[str, Any]:
        """Tag every image under ``folder`` and persist results. Returns a summary."""
        folder = Path(os.path.expanduser(str(folder))).resolve()
        candidates = list(iter_images(folder, recursive=recursive))
        todo = [p for p in candidates if force or not self._is_unchanged(p)]
        results = self.classify(todo)
        self._persist(results)
        errors = [r for r in results if r.error]
        return {
            "folder": str(folder),
            "found": len(candidates),
            "skipped_unchanged": len(candidates) - len(todo),
            "tagged": len(results) - len(errors),
            "errors": [r.as_dict() for r in errors],
        }

    def tag_configured_folders(self, force: bool = False) -> List[Dict[str, Any]]:
        if not self.settings.folders:
            raise PhotoTaggerError(
                "No folders configured. Set plugins.photo_tagger.folders or pass a folder."
            )
        return [self.tag_folder(f, force=force) for f in self.settings.folders]

    def find(self, query: str, limit: int = 25) -> List[Dict[str, Any]]:
        """Return photos whose labels contain every word of ``query`` (case-insensitive)."""
        words = [w.strip().lower() for w in query.split() if w.strip()]
        if not words:
            return []
        where = " AND ".join("LOWER(label) LIKE ?" for _ in words)
        sql = (
            "SELECT path, label, MAX(score) AS score FROM tags WHERE "
            + where
            + " GROUP BY path ORDER BY score DESC LIMIT ?"
        )
        params: List[Any] = [f"%{w}%" for w in words] + [int(limit)]
        rows = self._db().execute(sql, params).fetchall()
        return [{"path": r[0], "label": r[1], "score": round(r[2], 4)} for r in rows]

    def tags_for(self, path: Path) -> List[Tag]:
        """Stored tags for one image (empty list if it was never tagged)."""
        key = str(Path(os.path.expanduser(str(path))).resolve())
        rows = self._db().execute(
            "SELECT label, score FROM tags WHERE path = ? ORDER BY score DESC", (key,)
        ).fetchall()
        return [Tag(r[0], r[1]) for r in rows]

    def stats(self) -> Dict[str, Any]:
        conn = self._db()
        photos = conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
        labels = conn.execute("SELECT COUNT(DISTINCT label) FROM tags").fetchone()[0]
        top = conn.execute(
            "SELECT label, COUNT(*) AS c FROM tags GROUP BY label ORDER BY c DESC LIMIT 10"
        ).fetchall()
        return {
            "db_path": str(self.settings.db_path),
            "model": self.settings.model,
            "model_loaded": self._model is not None,
            "photos": photos,
            "distinct_labels": labels,
            "top_labels": [{"label": lbl, "count": c} for lbl, c in top],
        }


# ---------------------------------------------------------------- plugin API
_PLUGIN: Optional[MobileNetV3PhotoTagger] = None


def activate(cfg: Optional[HomieConfig] = None) -> Optional[MobileNetV3PhotoTagger]:
    """Create the plugin instance. Cheap: no model load and no DB open yet.

    Returns ``None`` when ``plugins.photo_tagger.enabled`` is false.
    """
    global _PLUGIN
    if cfg is not None and not cfg_get(cfg, "plugins", PLUGIN_NAME, "enabled", default=True):
        LOG.info("photo_tagger disabled by config")
        return None
    if _PLUGIN is None:
        _PLUGIN = MobileNetV3PhotoTagger(TaggerSettings.from_config(cfg))
        LOG.info(
            "photo_tagger activated (model=%s, offline=%s)",
            _PLUGIN.settings.model,
            not _PLUGIN.settings.allow_download,
        )
    return _PLUGIN


def deactivate() -> None:
    """Release model + DB handle and forget the instance."""
    global _PLUGIN
    if _PLUGIN is not None:
        _PLUGIN.close()
        _PLUGIN = None


def get_plugin() -> MobileNetV3PhotoTagger:
    if _PLUGIN is None:
        raise PhotoTaggerError("photo_tagger is not active; call activate(cfg) first")
    return _PLUGIN


HELP_TEXT = (
    "Photo tagger commands:\n"
    "  :photos tag [folder] [--force]  tag configured folders or one folder\n"
    "  :photos find <query>            list photos whose labels match\n"
    "  :photos tags <path>             show stored tags for one image\n"
    "  :photos stats                   database summary\n"
    "  :photos help                    show this help"
)


def handle_meta_command(cmd: str) -> bool:
    """Handle ``:photos ...`` REPL commands. Returns True when the command was consumed."""
    parts = cmd.strip().split()
    if not parts or parts[0].lower() != ":photos":
        return False
    sub = parts[1].lower() if len(parts) > 1 else "help"
    args = parts[2:]

    try:
        plugin = get_plugin()
        if sub == "tag":
            force = "--force" in args
            folder_words = [a for a in args if not a.startswith("--")]
            if folder_words:
                summaries = [plugin.tag_folder(Path(" ".join(folder_words)), force=force)]
            else:
                summaries = plugin.tag_configured_folders(force=force)
            for s in summaries:
                print(
                    f"{s['folder']}: found={s['found']} tagged={s['tagged']} "
                    f"skipped={s['skipped_unchanged']} errors={len(s['errors'])}"
                )
                for err in s["errors"][:5]:
                    print(f"  ! {err['path']}: {err['error']}")
        elif sub == "find":
            if not args:
                print("Usage: :photos find <query>")
                return True
            hits = plugin.find(" ".join(args))
            if not hits:
                print("No tagged photos match. Run :photos tag first?")
            for h in hits:
                print(f"{h['score']:.2f}  {h['label']:<32} {h['path']}")
        elif sub == "tags":
            if not args:
                print("Usage: :photos tags <path>")
                return True
            tags = plugin.tags_for(Path(" ".join(args)))
            if not tags:
                print("No stored tags for that path.")
            for t in tags:
                print(f"{t.score:.2f}  {t.label}")
        elif sub == "stats":
            st = plugin.stats()
            print(
                f"db={st['db_path']} photos={st['photos']} labels={st['distinct_labels']} "
                f"model={st['model']} loaded={st['model_loaded']}"
            )
            for item in st["top_labels"]:
                print(f"  {item['count']:>5}  {item['label']}")
        else:
            print(HELP_TEXT)
    except PhotoTaggerError as exc:
        print(f"photo_tagger: {exc}")
    except Exception as exc:  # noqa: BLE001 - never let a plugin crash the REPL
        LOG.exception("photo_tagger command failed")
        print(f"photo_tagger: unexpected error: {exc}")
    return True


__all__ = [
    "MobileNetV3PhotoTagger",
    "PhotoTaggerError",
    "Tag",
    "TagResult",
    "TaggerSettings",
    "activate",
    "deactivate",
    "get_plugin",
    "handle_meta_command",
    "iter_images",
]
