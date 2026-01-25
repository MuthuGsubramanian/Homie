from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from homie.controller.orchestrator import Orchestrator
from homie.controller.storage import Storage


def create_app(orchestrator: Orchestrator, storage: Storage) -> FastAPI:
    app = FastAPI(title="HOMIE Dashboard", docs_url=None, redoc_url=None)

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/machines")
    def machines():
        return list(storage.list_machines())

    @app.post("/api/tasks/run")
    def run_task(payload: dict):
        res = orchestrator.dispatch(payload, dry_run=payload.get("dry_run", False))
        if not res.ok:
            raise HTTPException(status_code=400, detail=res.error or "task failed")
        return JSONResponse(content=res.data)

    @app.get("/api/tasks")
    def tasks():
        return storage.recent_runs()

    @app.get("/api/ledger")
    def ledger():
        return storage.recent_ledger()

    return app


__all__ = ["create_app"]
