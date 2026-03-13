from __future__ import annotations
from homie_core.plugins.base import HomiePlugin, PluginResult


class MusicPlugin(HomiePlugin):
    name = "music"
    description = "Now playing info and media controls"
    permissions = ["read_media", "control_media"]

    def on_activate(self, config): pass
    def on_deactivate(self): pass

    def on_context(self):
        info = self._get_now_playing()
        return {"now_playing": info} if info else {}

    def on_query(self, intent, params):
        if intent == "now_playing":
            info = self._get_now_playing()
            return PluginResult(success=True, data=info or "Nothing playing")
        return PluginResult(success=False, error=f"Unknown intent: {intent}")

    def on_action(self, action, params):
        return PluginResult(success=False, error="Media control not yet implemented")

    def _get_now_playing(self):
        try:
            import winsdk.windows.media.control as mc
            import asyncio
            async def get():
                mgr = await mc.GlobalSystemMediaTransportControlsSessionManager.request_async()
                session = mgr.get_current_session()
                if session:
                    info = await session.try_get_media_properties_async()
                    return {"title": info.title, "artist": info.artist, "album": info.album_title}
                return None
            return asyncio.run(get())
        except Exception:
            return None
