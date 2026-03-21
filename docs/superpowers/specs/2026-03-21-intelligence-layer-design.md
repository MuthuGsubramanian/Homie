# Intelligence Layer — Design Spec

**Date:** 2026-03-21
**Status:** Approved
**Branch:** `feat/homie-ai-v2`
**Scope:** Universal document pipeline, knowledge graph, advanced retrieval, intelligence wiring

---

## 1. Overview

Four sub-projects that transform Homie from a conversational assistant into a highly intelligent local AI that understands any document, builds a knowledge graph, retrieves with precision, and proactively uses all available context.

| Sub-project | Purpose |
|-------------|---------|
| SP1: Universal Document Pipeline | Parse any format, upgrade embeddings + chunking |
| SP2: Knowledge Graph | Entity/relationship extraction, SQLite triple store |
| SP3: Advanced Retrieval | Reranking, graph-expanded search, tiered context assembly |
| SP4: Intelligence Wiring | Wire all data sources into responses, persistent state, proactive intelligence |

---

## 2. Sub-project 1: Universal Document Pipeline

### 2.1 Format Detection

`src/homie_core/rag/format_detector.py` — detects file format via magic bytes (first 16 bytes) then extension fallback.

```python
class DocumentFormat(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    PPTX = "pptx"
    HTML = "html"
    IMAGE = "image"       # jpg, png, bmp, tiff
    CODE = "code"         # any programming language
    MARKDOWN = "markdown"
    EMAIL = "email"       # .eml, .msg
    EPUB = "epub"
    CSV = "csv"
    TEXT = "text"         # plain text fallback
    UNKNOWN = "unknown"

def detect_format(path: Path) -> DocumentFormat:
    """Detect format via magic bytes, then extension fallback."""
```

### 2.2 Parser Registry

`src/homie_core/rag/parsers/__init__.py` — maps formats to parser functions.

```python
@dataclass
class ParsedDocument:
    text_blocks: list[TextBlock]  # ordered text segments
    metadata: dict                # title, author, page_count, language, etc.
    tables: list[TableData]       # structured table data
    source_path: str

@dataclass
class TextBlock:
    content: str
    block_type: str      # "heading", "paragraph", "code", "table_cell", "caption"
    level: int           # heading level (1-6) or 0
    page: int | None     # PDF page number
    line_start: int | None
    line_end: int | None
    language: str | None # for code blocks
    parent_heading: str | None

@dataclass
class TableData:
    headers: list[str]
    rows: list[list[str]]
    caption: str | None
    source_page: int | None

PARSER_REGISTRY: dict[DocumentFormat, Callable[[Path], ParsedDocument]] = {}
```

### 2.3 Format-Specific Parsers

Each parser lives in `src/homie_core/rag/parsers/<format>.py` and returns `ParsedDocument`.

**PDF** (`parsers/pdf.py`): PyMuPDF for text+structure. Detects scanned pages (low text per page) and falls back to OCR. Extracts: text blocks by page, headings from font size changes, tables, metadata (title, author, dates).

**DOCX** (`parsers/docx.py`): python-docx. Extracts: paragraphs with heading levels, tables with headers/rows, metadata from core properties.

**XLSX/CSV** (`parsers/xlsx.py`): openpyxl for XLSX, csv stdlib for CSV. Each sheet becomes text blocks. Tables extracted with headers. Formulas resolved to values.

**PPTX** (`parsers/pptx.py`): python-pptx. Each slide becomes a text block with title + bullet points. Speaker notes included.

**HTML** (`parsers/html.py`): trafilatura for main content extraction (strips nav/ads/boilerplate). Falls back to BeautifulSoup.

**Image** (`parsers/image.py`): OCR via EasyOCR (lighter than surya, pip-installable). Returns extracted text as a single block. Optional — returns empty if easyocr not installed.

**Code** (`parsers/code.py`): tree-sitter for AST-aware parsing. Extracts: function/class definitions with signatures, imports, module-level docstrings. Falls back to regex-based extraction (current chunker logic) if tree-sitter unavailable.

**Email** (`parsers/email.py`): stdlib email module. Extracts: subject, from, to, date, body (HTML→text via BeautifulSoup or plain), attachment names.

**Epub** (`parsers/epub.py`): ebooklib. Chapters → HTML → trafilatura/BeautifulSoup → text blocks.

### 2.4 Embedding Upgrade

Replace `all-MiniLM-L6-v2` (384-dim) with `bge-base-en-v1.5` (768-dim) as default in `neural/model_manager.py`.

Config addition:
```python
class EmbeddingConfig(BaseModel):
    model_name: str = "BAAI/bge-base-en-v1.5"
    dimensions: int = 768
    batch_size: int = 32
```

Migration: on first run with new model, detect dimension mismatch in ChromaDB collection, recreate collection and re-index.

### 2.5 Semantic Chunker

`src/homie_core/rag/semantic_chunker.py` — embedding-based topic boundary detection.

```python
def semantic_chunk(text: str, embedder, threshold: float = 0.3, max_chunk_size: int = 1500) -> list[str]:
    """Split text at semantic boundaries.
    1. Split into sentences
    2. Embed each sentence
    3. Compute cosine similarity between consecutive sentences
    4. Split where similarity drops below threshold
    5. Merge small chunks up to max_chunk_size
    """
```

Used as a refinement pass after structure-aware splitting for high-value documents.

### 2.6 Pipeline Integration

Modify `src/homie_core/rag/pipeline.py`:

```python
def index_file(self, path: Path) -> int:
    fmt = detect_format(path)
    if fmt == DocumentFormat.UNKNOWN:
        return 0
    parser = PARSER_REGISTRY.get(fmt)
    if not parser:
        return 0
    doc = parser(path)
    chunks = self._chunk_parsed_document(doc)
    self._embed_and_store(chunks)
    return len(chunks)
```

The existing `_SUPPORTED_EXTENSIONS` check is replaced by `detect_format()`. All existing text file support continues working.

### 2.7 Dependencies

```toml
[project.optional-dependencies]
docs = [
    "PyMuPDF>=1.24",
    "python-docx>=1.0",
    "openpyxl>=3.1",
    "python-pptx>=0.6",
    "trafilatura>=1.8",
    "ebooklib>=0.18",
    "beautifulsoup4>=4.12",
]
ocr = ["easyocr>=1.7"]
code-ast = ["tree-sitter>=0.21"]
```

All optional — graceful degradation if not installed.

---

## 3. Sub-project 2: Knowledge Graph

### 3.1 Schema

SQLite triple store with typed entities. Two tables:

**entities table:**
```sql
CREATE TABLE entities (
    id TEXT PRIMARY KEY,          -- uuid
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,    -- person, project, concept, tool, document, task, event, location, snippet, goal
    attributes TEXT,              -- JSON dict of type-specific attributes
    confidence REAL DEFAULT 1.0,
    source TEXT,                  -- how this entity was created (extraction, user, inference)
    created_at TEXT,
    updated_at TEXT,
    last_accessed TEXT            -- for recency scoring
);
CREATE INDEX idx_entities_type ON entities(entity_type);
CREATE INDEX idx_entities_name ON entities(name);
```

**relationships table:**
```sql
CREATE TABLE relationships (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL REFERENCES entities(id),
    relation TEXT NOT NULL,       -- authored, works_on, mentions, depends_on, contains, related_to, uses, supports, child_of
    object_id TEXT NOT NULL REFERENCES entities(id),
    confidence REAL DEFAULT 1.0,
    source TEXT,                  -- provenance (document path, conversation, user statement)
    source_chunk_id TEXT,         -- link back to RAG chunk
    created_at TEXT,
    updated_at TEXT
);
CREATE INDEX idx_rel_subject ON relationships(subject_id);
CREATE INDEX idx_rel_object ON relationships(object_id);
CREATE INDEX idx_rel_relation ON relationships(relation);
```

### 3.2 Entity Types

| Type | Key Attributes | Extracted From |
|------|---------------|----------------|
| person | name, email, role, organization | Documents, emails, conversations |
| project | name, status, goal, deadline | Git repos, conversations, docs |
| concept | name, definition, domain | Documents, conversations |
| tool | name, version, purpose | Code imports, conversations |
| document | path, title, summary, format | File indexing |
| task | description, status, due_date, project | Conversations, todo middleware |
| event | name, date, participants, location | Calendar (future), conversations |
| location | name, type | Conversations, documents |
| snippet | content, source_doc, confidence | Extracted facts from documents |
| goal | description, timeframe, status | Conversations |

### 3.3 Entity Extraction

`src/homie_core/knowledge/extractor.py` — extracts entities and relationships from text.

**Two-tier approach:**

**Tier 1 — Pattern-based (fast, always available):**
- Person names: regex + capitalization heuristics
- Dates/times: dateutil parsing
- URLs, emails, file paths: regex
- Code identifiers: import statements, function calls
- Project names: from git, directory names

**Tier 2 — Model-based (better, optional):**
- spaCy NER (`en_core_web_sm`) for Person, Org, Date, Location, etc.
- Falls back to Tier 1 if spaCy not installed

```python
class EntityExtractor:
    def __init__(self, use_model: bool = True):
        self._nlp = None
        if use_model:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
            except (ImportError, OSError):
                pass  # fall back to pattern-based

    def extract(self, text: str, source: str) -> tuple[list[Entity], list[Relationship]]:
        """Extract entities and relationships from text."""
```

### 3.4 Knowledge Graph Manager

`src/homie_core/knowledge/graph.py`:

```python
class KnowledgeGraph:
    def __init__(self, db_path: str | Path):
        self._db = sqlite3.connect(str(db_path))
        self._ensure_schema()

    # Entity CRUD
    def add_entity(self, entity: Entity) -> str: ...
    def get_entity(self, entity_id: str) -> Entity | None: ...
    def find_entities(self, name: str = None, entity_type: str = None) -> list[Entity]: ...
    def merge_entity(self, entity: Entity) -> str:
        """Add or merge with existing entity of same name+type."""

    # Relationship CRUD
    def add_relationship(self, rel: Relationship) -> str: ...
    def get_relationships(self, entity_id: str, relation: str = None, direction: str = "both") -> list[Relationship]: ...

    # Graph traversal
    def neighbors(self, entity_id: str, max_hops: int = 2) -> list[Entity]: ...
    def path_between(self, from_id: str, to_id: str, max_hops: int = 3) -> list[Relationship] | None: ...

    # Query support
    def entities_mentioned_in(self, text: str) -> list[Entity]:
        """Find entities whose names appear in the text."""

    def context_for_entity(self, entity_id: str) -> str:
        """Generate a natural language summary of an entity and its relationships."""

    # Maintenance
    def decay_scores(self, half_life_days: int = 30): ...
    def prune(self, min_confidence: float = 0.1): ...
```

### 3.5 Integration with Document Pipeline

After chunks are embedded and stored (SP1), the extraction pipeline runs:

```python
# In rag/pipeline.py after embedding
extractor = EntityExtractor()
for chunk in chunks:
    entities, relationships = extractor.extract(chunk.content, source=chunk.source)
    for entity in entities:
        graph.merge_entity(entity)
    for rel in relationships:
        graph.add_relationship(rel)
```

### 3.6 Integration with Learning Pipeline

Modify `memory/learning_pipeline.py` to also feed extracted facts into the knowledge graph:

```python
# When a fact is learned:
entity = Entity(name=subject, entity_type=infer_type(subject))
graph.merge_entity(entity)
snippet = Entity(name=fact_text, entity_type="snippet", attributes={"confidence": confidence})
graph.add_relationship(Relationship(subject=entity.id, relation="has_fact", object=snippet.id))
```

---

## 4. Sub-project 3: Advanced Retrieval

### 4.1 Cross-Encoder Reranker

`src/homie_core/rag/reranker.py`:

```python
class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # Load via ONNX Runtime or sentence-transformers
        ...

    def rerank(self, query: str, documents: list[str], top_k: int = 10) -> list[tuple[int, float]]:
        """Score (query, doc) pairs jointly. Returns (index, score) sorted by score."""
```

Integrated into `hybrid_search.py` as an optional second stage:
1. Vector + BM25 → top-50 candidates via RRF
2. Cross-encoder reranks top-50 → final top-K

### 4.2 Graph-Expanded Retrieval

After hybrid search returns top-K chunks, expand via knowledge graph:

```python
def graph_expand(query: str, chunks: list[Chunk], graph: KnowledgeGraph, budget: int = 3) -> list[Chunk]:
    """Expand retrieval results using knowledge graph.
    1. Find entities mentioned in query
    2. For each entity, get 1-hop neighbors
    3. Find chunks that mention those neighbors
    4. Add to results (up to budget additional chunks)
    """
```

### 4.3 Tiered Context Assembly

`src/homie_core/rag/context_assembler.py`:

Replace the simple "concatenate top-K chunks" approach with intelligent tiered assembly:

```python
class ContextAssembler:
    def __init__(self, config: HomieConfig, graph: KnowledgeGraph):
        self._config = config
        self._graph = graph

    def assemble(self, query: str, chunks: list[Chunk], conversation: list[dict],
                 user_profile: dict, token_budget: int) -> str:
        """Assemble context within token budget using tiered priority:
        Tier 1: Top reranked chunks (highest relevance)
        Tier 2: Graph context for mentioned entities
        Tier 3: Recent conversation turns
        Tier 4: User profile + persistent facts
        Deduplicate near-identical content (cosine > 0.95).
        Place highest relevance at start and end (lost-in-middle mitigation).
        """
```

### 4.4 Modified Cognitive Pipeline

The RETRIEVE stage in `cognitive_arch.py` is upgraded:

```python
# Old: simple fact + episode + RAG retrieval
# New: hybrid search → rerank → graph expand → context assembly

chunks = self._rag.search(query, top_k=50)           # hybrid search
chunks = self._reranker.rerank(query, chunks, top_k=10)  # cross-encoder
chunks = graph_expand(query, chunks, self._graph)      # graph expansion
context = self._assembler.assemble(query, chunks, ...)  # tiered assembly
```

---

## 5. Sub-project 4: Intelligence Wiring

### 5.1 Wire Email into Retrieval

Email data is already synced to `cache.db`. Add email chunks to the RAG index:

```python
class EmailIndexer:
    """Indexes synced emails into the RAG pipeline and knowledge graph."""
    def index_emails(self, since_days: int = 30):
        for email in self._get_recent_emails(since_days):
            # Parse email → TextBlocks
            # Chunk and embed
            # Extract entities (sender → Person, recipients → Person, subject → Topic)
            # Add to knowledge graph
```

### 5.2 Wire Social/Behavioral into Context

Create a `ContextEnricher` middleware that injects non-document context:

```python
class ContextEnricherMiddleware(HomieMiddleware):
    name = "context_enricher"
    order = 25

    def modify_prompt(self, system_prompt: str) -> str:
        context_blocks = []
        # Recent email summary (last 24h)
        if self._email_service:
            context_blocks.append(self._email_summary())
        # Active project context (from git + knowledge graph)
        if self._graph:
            context_blocks.append(self._active_project_context())
        # Behavioral state (current activity, mood, flow)
        if self._behavioral:
            context_blocks.append(self._behavioral_summary())
        if context_blocks:
            return system_prompt + "\n[LIVE CONTEXT]\n" + "\n".join(context_blocks) + "\n"
        return system_prompt
```

### 5.3 Persistent Cross-Session State

Save working memory state to disk at session end, restore on next start:

```python
class SessionPersistence:
    def save_session(self, working_memory: WorkingMemory, path: Path):
        """Serialize current activity, behavioral state, recent topics to JSON."""

    def restore_session(self, working_memory: WorkingMemory, path: Path):
        """Restore last session's activity context (not conversation — that's in episodic memory)."""
```

### 5.4 Proactive Intelligence

Upgrade `ProactiveEngine` to use knowledge graph:

- **Deadline awareness**: query graph for tasks/events with upcoming due_dates
- **Follow-up suggestions**: when user discusses topic X, check graph for related unresolved tasks
- **Knowledge gaps**: detect when user asks about something with low-confidence graph entries
- **Habit reinforcement**: combine behavioral observers with graph-based goal tracking

---

## 6. File Inventory

### Sub-project 1: Universal Document Pipeline
| File | Purpose |
|------|---------|
| `rag/format_detector.py` | Format detection via magic bytes + extension |
| `rag/parsers/__init__.py` | Parser registry, ParsedDocument, TextBlock, TableData |
| `rag/parsers/pdf.py` | PDF parser (PyMuPDF) |
| `rag/parsers/docx.py` | Word parser (python-docx) |
| `rag/parsers/xlsx.py` | Excel/CSV parser (openpyxl) |
| `rag/parsers/pptx.py` | PowerPoint parser (python-pptx) |
| `rag/parsers/html.py` | HTML/web parser (trafilatura) |
| `rag/parsers/image.py` | OCR parser (easyocr) |
| `rag/parsers/code.py` | Code AST parser (tree-sitter) |
| `rag/parsers/email_parser.py` | Email parser (stdlib) |
| `rag/parsers/epub.py` | Epub parser (ebooklib) |
| `rag/semantic_chunker.py` | Embedding-based boundary detection |
| Modify: `rag/pipeline.py` | Use format detection + parser registry |
| Modify: `neural/model_manager.py` | Upgrade to bge-base-en-v1.5 |
| Modify: `config.py` | Add EmbeddingConfig |
| Modify: `pyproject.toml` | Add docs, ocr, code-ast optional deps |

### Sub-project 2: Knowledge Graph
| File | Purpose |
|------|---------|
| `knowledge/__init__.py` | Public API |
| `knowledge/models.py` | Entity, Relationship dataclasses |
| `knowledge/graph.py` | KnowledgeGraph SQLite triple store |
| `knowledge/extractor.py` | EntityExtractor (pattern + spaCy) |
| Modify: `rag/pipeline.py` | Feed extracted entities to graph |
| Modify: `memory/learning_pipeline.py` | Feed learned facts to graph |

### Sub-project 3: Advanced Retrieval
| File | Purpose |
|------|---------|
| `rag/reranker.py` | CrossEncoderReranker |
| `rag/graph_retrieval.py` | Graph-expanded retrieval |
| `rag/context_assembler.py` | Tiered context assembly |
| Modify: `rag/hybrid_search.py` | Integrate reranker |
| Modify: `brain/cognitive_arch.py` | Upgraded RETRIEVE stage |

### Sub-project 4: Intelligence Wiring
| File | Purpose |
|------|---------|
| `middleware/context_enricher.py` | ContextEnricherMiddleware |
| `knowledge/email_indexer.py` | Index emails into RAG + graph |
| `knowledge/session_persistence.py` | Save/restore working memory state |
| Modify: `intelligence/proactive_engine.py` | Graph-aware proactive suggestions |

---

## 7. Dependency Additions

```toml
[project.optional-dependencies]
docs = [
    "PyMuPDF>=1.24",
    "python-docx>=1.0",
    "openpyxl>=3.1",
    "python-pptx>=0.6",
    "trafilatura>=1.8",
    "ebooklib>=0.18",
    "beautifulsoup4>=4.12",
]
ocr = ["easyocr>=1.7"]
code-ast = ["tree-sitter>=0.21"]
nlp = ["spacy>=3.7"]
reranker = ["sentence-transformers>=2.5"]
```

All optional — each sub-system gracefully degrades without its deps.

---

## 8. Migration & Testing

- Embedding model change triggers automatic re-indexing on first run
- Knowledge graph starts empty, builds incrementally
- All new features are additive — existing functionality unchanged
- Each sub-project has independent tests
- Integration test: index a sample PDF → verify entities in graph → query retrieves relevant context
