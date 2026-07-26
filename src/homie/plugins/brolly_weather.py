"""Brolly plain-text weather plugin for HOMIE.

Brolly (https://brolly.sh) serves weather forecasts as plain text -- no API
key, no JSON, no HTML scraping. That makes it a good fit for a lightweight
HOMIE weather source: one GET request returns a human-readable forecast that
can be shown in the dashboard, spoken through the TTS pipeline, or handed to
the planner as context.

Privacy / safety:
    * The plugin performs NO network calls until it is explicitly enabled in
      ``homie.config.yaml`` (opt-in, disabled by default).
    * Responses are cached locally so repeated queries within the TTL window
      never touch the network.

Configuration (``homie.config.yaml``)::

    plugins:
      brolly_weather:
        enabled: true                                   # opt-in network gate
        forecast_url: "https://brolly.sh/forecast/RWFP2qW8"
        cache_ttl_sec: 900                              # 15 minutes
        timeout_sec: 10
        summary_lines: 6                                # lines used for TTS

Usage::

    from homie.config import load_config
    from plugins.brolly_weather import register

    plugin = register(load_config())
    if plugin is not None:
        print(plugin.get_forecast().raw_text)
        plugin.deactivate()
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import requests

from homie.config import HomieConfig, cfg_get

DEFAULT_FORECAST_URL = "https://brolly.sh/forecast/RWFP2qW8"
DEFAULT_CACHE_TTL_SEC = 900
DEFAULT_TIMEOUT_SEC = 10
DEFAULT_SUMMARY_LINES = 6

MAX_BODY_BYTES = 64 * 1024  # sanity cap; forecasts are a few KB of text


class BrollyWeatherError(Exception):
    """Raised when a forecast cannot be fetched or parsed."""


@dataclass
class Forecast:
    """A fetched plain-text forecast."""

    url: str
    fetched_at: float
    raw_text: str
    lines: List[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        return self.lines[0] if self.lines else ""

    def summary(self, max_lines: int = DEFAULT_SUMMARY_LINES) -> str:
        """Short speakable/displayable excerpt of the forecast."""
        return "\n".join(self.lines[:max_lines])


class BrollyWeatherPlugin:
    """Fetches and caches plain-text forecasts from brolly.sh.

    The plugin is inert until :meth:`activate` is called, and ``activate``
    refuses to arm itself unless ``plugins.brolly_weather.enabled`` is true
    in the loaded configuration (network access is strictly opt-in).
    """

    name = "brolly_weather"

    def __init__(self, cfg: HomieConfig) -> None:
        self.cfg = cfg
        self.enabled = bool(cfg_get(cfg, "plugins", self.name, "enabled", default=False))
        self.forecast_url = str(
            cfg_get(cfg, "plugins", self.name, "forecast_url", default=DEFAULT_FORECAST_URL)
        )
        self.cache_ttl_sec = float(
            cfg_get(cfg, "plugins", self.name, "cache_ttl_sec", default=DEFAULT_CACHE_TTL_SEC)
        )
        self.timeout_sec = float(
            cfg_get(cfg, "plugins", self.name, "timeout_sec", default=DEFAULT_TIMEOUT_SEC)
        )
        self.summary_lines = int(
            cfg_get(cfg, "plugins", self.name, "summary_lines", default=DEFAULT_SUMMARY_LINES)
        )

        self._active = False
        self._lock = threading.Lock()
        self._cached: Optional[Forecast] = None

    # -- lifecycle -----------------------------------------------------

    def activate(self) -> bool:
        """Arm the plugin. Returns True when active, False when disabled."""
        if not self.enabled:
            logging.info("brolly_weather: disabled in config, staying inert")
            return False
        if not self.forecast_url.startswith("https://brolly.sh/"):
            raise BrollyWeatherError(
                f"forecast_url must point at https://brolly.sh/, got '{self.forecast_url}'"
            )
        self._active = True
        logging.info(
            "brolly_weather: active (url=%s ttl=%ss)", self.forecast_url, self.cache_ttl_sec
        )
        return True

    def deactivate(self) -> None:
        """Disarm the plugin and drop the cache."""
        with self._lock:
            self._active = False
            self._cached = None
        logging.info("brolly_weather: deactivated")

    # -- public API ----------------------------------------------------

    def get_forecast(self, force_refresh: bool = False) -> Forecast:
        """Return the current forecast, served from cache while fresh."""
        if not self._active:
            raise BrollyWeatherError("plugin is not active; call activate() first")

        with self._lock:
            if not force_refresh and self._cached is not None:
                age = time.time() - self._cached.fetched_at
                if age < self.cache_ttl_sec:
                    return self._cached
            forecast = self._fetch()
            self._cached = forecast
            return forecast

    def speak_summary(self) -> str:
        """Summary trimmed for the TTS pipeline (Piper / kokoro_tts plugin)."""
        return self.get_forecast().summary(self.summary_lines)

    # -- internals -----------------------------------------------------

    def _fetch(self) -> Forecast:
        headers = {
            "Accept": "text/plain",
            "User-Agent": "HOMIE-brolly-weather-plugin",
        }
        try:
            resp = requests.get(self.forecast_url, headers=headers, timeout=self.timeout_sec)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise BrollyWeatherError(f"Brolly request failed: {exc}") from exc

        text = resp.text[:MAX_BODY_BYTES].replace("\r\n", "\n").strip()
        if not text:
            raise BrollyWeatherError("Brolly returned an empty forecast body")

        lines = [line.rstrip() for line in text.split("\n") if line.strip()]
        return Forecast(
            url=self.forecast_url,
            fetched_at=time.time(),
            raw_text=text,
            lines=lines,
        )


def register(cfg: HomieConfig) -> Optional[BrollyWeatherPlugin]:
    """Entry point used by the HOMIE plugin loader.

    Returns an activated plugin, or None when the plugin is disabled in
    config (the loader treats None as 'skip').
    """
    plugin = BrollyWeatherPlugin(cfg)
    if not plugin.activate():
        return None
    return plugin


__all__ = [
    "BrollyWeatherPlugin",
    "BrollyWeatherError",
    "Forecast",
    "register",
]
