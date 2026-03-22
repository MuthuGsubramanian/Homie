"""Surface extractor — quick AST/regex extraction without LLM."""

import ast
import re
from pathlib import Path
from typing import Any


class SurfaceExtractor:
    """Extracts surface-level knowledge from files using AST and regex."""

    def extract(self, file_path: Path, file_type: str) -> dict[str, Any]:
        """Extract surface knowledge from a file."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, PermissionError):
            return {"error": "unreadable"}

        base = {
            "file": str(file_path),
            "file_type": file_type,
            "line_count": content.count("\n") + 1,
            "size_bytes": len(content.encode("utf-8")),
        }

        if file_type == "python":
            base.update(self._extract_python(content))
        elif file_type == "markdown":
            base.update(self._extract_markdown(content))
        elif file_type in ("javascript", "typescript"):
            base.update(self._extract_js(content))
        else:
            base.update(self._extract_generic(content))

        # Generate entity list from extracted names
        entities = []
        for cls in base.get("classes", []):
            entities.append({"name": cls, "type": "class", "source_file": str(file_path)})
        for fn in base.get("functions", []):
            entities.append({"name": fn, "type": "function", "source_file": str(file_path)})
        base["entities"] = entities

        return base

    def _extract_python(self, content: str) -> dict:
        """Extract from Python using AST."""
        classes, functions, imports = [], [], []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    # Skip methods (inside classes)
                    if not any(isinstance(p, ast.ClassDef) for p in ast.walk(tree) if hasattr(p, 'body') and node in getattr(p, 'body', [])):
                        functions.append(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module.split(".")[0])
        except SyntaxError:
            # Fallback to regex
            classes = re.findall(r"^class\s+(\w+)", content, re.MULTILINE)
            functions = re.findall(r"^def\s+(\w+)", content, re.MULTILINE)
            imports = re.findall(r"^(?:import|from)\s+(\w+)", content, re.MULTILINE)

        return {
            "classes": classes,
            "functions": functions,
            "imports": list(set(imports)),
        }

    def _extract_markdown(self, content: str) -> dict:
        """Extract headings from markdown."""
        headings = re.findall(r"^#+\s+(.+)$", content, re.MULTILINE)
        return {"headings": headings}

    def _extract_js(self, content: str) -> dict:
        """Extract from JavaScript/TypeScript using regex."""
        classes = re.findall(r"(?:class|interface)\s+(\w+)", content)
        functions = re.findall(r"(?:function|const|let|var)\s+(\w+)\s*(?:=\s*(?:async\s*)?\(|[\(])", content)
        imports = re.findall(r"(?:import|require)\s*\(?['\"]([^'\"]+)", content)
        return {"classes": classes, "functions": functions, "imports": imports}

    def _extract_generic(self, content: str) -> dict:
        """Minimal extraction for unknown file types."""
        return {}
