"""KnowledgeBuilder — coordinates conversation mining, project tracking, profiling, and knowledge graph."""

import logging
from pathlib import Path
from typing import Any, Callable, Optional

from ..observation.signals import LearningSignal
from ..storage import LearningStorage
from .behavioral_profiler import BehavioralProfiler
from .conversation_miner import ConversationMiner
from .graph.query import GraphQuery
from .graph.store import KnowledgeGraphStore
from .intake.pipeline import IntakePipeline
from .project_tracker import ProjectTracker

logger = logging.getLogger(__name__)


class KnowledgeBuilder:
    """Coordinates all knowledge-building engines including the knowledge graph."""

    def __init__(
        self,
        storage: LearningStorage,
        inference_fn: Optional[Callable[[str], str]] = None,
        graph_db_path: Optional[Path | str] = None,
    ) -> None:
        self._storage = storage
        self.miner = ConversationMiner(storage=storage, inference_fn=inference_fn)
        self.project_tracker = ProjectTracker(storage=storage)
        self.profiler = BehavioralProfiler()

        # Knowledge graph (optional — graceful if no path provided)
        self.graph_store: Optional[KnowledgeGraphStore] = None
        self.graph_query: Optional[GraphQuery] = None
        self.intake: Optional[IntakePipeline] = None

        if graph_db_path:
            self.graph_store = KnowledgeGraphStore(db_path=graph_db_path)
            self.graph_store.initialize()
            self.graph_query = GraphQuery(store=self.graph_store)
            self.intake = IntakePipeline(
                graph_store=self.graph_store,
                inference_fn=inference_fn,
            )

    def on_signal(self, signal: LearningSignal) -> None:
        """Process knowledge-related signals."""
        data = signal.data
        if "hour" in data:
            hour = data["hour"]
            for key in ("app", "activity", "topic"):
                if key in data:
                    self.profiler.record_observation(hour, key, data[key])

    def process_turn(self, user_message: str, response: str) -> list[str]:
        """Process a conversation turn for knowledge extraction."""
        return self.miner.process_turn(user_message, response)

    def ingest_source(self, source: Path | str) -> dict[str, Any]:
        """Ingest a directory or file into the knowledge graph."""
        if self.intake is None:
            return {"error": "Knowledge graph not initialized", "files_scanned": 0}
        return self.intake.ingest(source)

    def get_work_hours(self) -> list[int]:
        """Get detected work hours."""
        return self.profiler.get_work_hours()
