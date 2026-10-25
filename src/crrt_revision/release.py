"""Inspect the public source tree for private artifacts and credentials."""
from pathlib import Path
import re


def check(root):
    root = Path(root).resolve()
    required = ["README.md", "CITATION.cff", "NOTICE.md", "pyproject.toml",
                "requirements.lock", "configs/main.yaml", "docs/methods.md", "docs/reproduction.md"]
    missing = [name for name in required if not (root / name).is_file()]
    forbidden, secrets = [], []
    for file in root.rglob("*"):
        if not file.is_file():
            continue
        relative = file.relative_to(root)
        if any(p in {".git", "__pycache__", ".pytest_cache", ".env", "sources", "predictions_private", "bootstrap_private",
                     "_checkpoints", "_checkpoint_archive", "cate_private", "publication_outputs", "publication_archive",
                     "build", "dist", "fixtures"} or p.endswith(".egg-info") for p in relative.parts):
            forbidden.append(str(relative))
        if file.suffix.lower() in {".parquet", ".pkl", ".pickle", ".pdf", ".tiff", ".tif", ".pyc", ".csv", ".tsv",
                                   ".xlsx", ".png", ".jpg", ".svg", ".html", ".css", ".js", ".tex", ".log", ".zip", ".json"}:
            forbidden.append(str(relative))
        if file.suffix.lower() in {".py", ".md", ".yaml", ".toml", ".txt", ".sql", ".json", ".cff", ".r"}:
            content = file.read_text(encoding="utf-8")
            if re.search(r"postgres(?:ql)?://[^\s]+:[^\s]+@", content):
                secrets.append(str(relative))
    result = {"files": sum(f.is_file() for f in root.rglob("*")), "missing_required": missing,
              "forbidden_artifacts": sorted(set(forbidden)), "credential_findings": secrets,
              "status": "pass" if not missing + forbidden + secrets else "fail"}
    if result["status"] != "pass":
        raise ValueError(str(result))
    return result
