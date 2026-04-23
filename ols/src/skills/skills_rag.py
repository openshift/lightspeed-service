"""Hybrid Skills RAG implementation for skill selection."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import frontmatter
from langchain_core.tools.structured import StructuredTool

from ols.src.rag.hybrid_rag import HybridRAGBase

logger = logging.getLogger(__name__)

_SKILL_MD = "skill.md"


@dataclass(frozen=True, slots=True)
class SkillLoadResult:
    """Outcome of reading the primary skill document for progressive disclosure."""

    content: str | None
    has_support_files: bool
    ok: bool


@dataclass(frozen=True, slots=True)
class Skill:
    """A loaded skill artifact with parsed metadata and directory path."""

    name: str
    description: str
    source_path: str
    support_files: list[tuple[str, str]] = field(default_factory=list)

    def load_skill(self) -> SkillLoadResult:
        """Read the skill body and optionally append a support-file manifest.

        On read or parse failure, returns ``SkillLoadResult`` with ``ok`` false
        and ``content`` set to ``None`` instead of raising.

        Returns:
            ``SkillLoadResult`` with prompt text (or ``None``), manifest flag,
            and success flag.
        """
        skill_dir = Path(self.source_path)
        try:
            raw = (skill_dir / _SKILL_MD).read_text(encoding="utf-8").strip()
            body = frontmatter.loads(raw).content.strip()
        except (OSError, UnicodeDecodeError, ValueError):
            logger.exception(
                "Failed to read or parse skill.md for skill '%s'",
                self.name,
            )
            return SkillLoadResult(content=None, has_support_files=False, ok=False)

        if not self.support_files:
            return SkillLoadResult(content=body, has_support_files=False, ok=True)

        manifest_lines = [
            "",
            "---",
            "Available support files (use load_skill_support_files to retrieve):",
        ]
        manifest_lines.extend(f"- {name} ({path})" for name, path in self.support_files)
        text = body + "\n".join(manifest_lines)
        return SkillLoadResult(content=text, has_support_files=True, ok=True)

    def load_support_files(self, files: list[str]) -> str:
        """Retrieve content of requested support files by relative path.

        Args:
            files: List of relative paths within the skill directory.

        Returns:
            Combined content of requested files, each with a header.
        """
        skill_dir = Path(self.source_path)
        parts: list[str] = []
        skill_root = skill_dir.resolve()

        for rel_path in files:
            target = skill_dir / rel_path
            resolved = target.resolve()
            if not resolved.is_relative_to(skill_root):
                parts.append(f"## {rel_path}\n\nERROR: path outside skill directory")
                continue
            if not resolved.is_file():
                parts.append(f"## {rel_path}\n\nERROR: file not found in skill")
                continue
            try:
                raw = resolved.read_text(encoding="utf-8").strip()
                parts.append(f"## {rel_path}\n\n{raw}")
            except (OSError, UnicodeDecodeError, ValueError):
                parts.append(f"## {rel_path}\n\nERROR: could not read file")

        return "\n\n".join(parts)

    def load_content(self) -> str:
        """Read all files in the skill directory tree and concatenate on demand.

        The main skill file body (everything after frontmatter) is returned
        first, followed by the contents of any additional text files found
        anywhere in the skill directory tree, each separated by a header
        showing the relative path.

        Returns:
            Combined content of all files in the skill directory tree.

        Raises:
            OSError: If the skill directory or its files cannot be read.
        """
        skill_dir = Path(self.source_path)
        parts: list[str] = []

        for entry in sorted(skill_dir.rglob("*")):
            if not entry.is_file():
                continue
            try:
                raw = entry.read_text(encoding="utf-8").strip()
            except (UnicodeDecodeError, ValueError):
                continue
            if entry.name.lower() == _SKILL_MD:
                parts.insert(0, frontmatter.loads(raw).content.strip())
            else:
                rel = entry.relative_to(skill_dir)
                parts.append(f"## {rel}\n\n{raw}")

        return "\n\n".join(parts)


def create_skill_support_tool(skill: Skill) -> StructuredTool:
    """Create a tool that retrieves additional support files for a skill.

    Args:
        skill: Loaded skill bound to one skill directory.

    Returns:
        A StructuredTool the LLM can invoke to fetch support-file contents.
    """

    async def _load_skill_support_files(files: list[str]) -> str:
        """Load additional support files for the current skill.

        Args:
            files: List of relative paths within the skill directory.

        Returns:
            Combined content of the requested support files.
        """
        return skill.load_support_files(files)

    return StructuredTool.from_function(
        coroutine=_load_skill_support_files,
        name="load_skill_support_files",
        description=(
            "Retrieve additional support files for the matched skill. "
            "Pass a list of relative file paths from the manifest."
        ),
    )


def _discover_support_files(skill_dir: Path) -> list[tuple[str, str]]:
    """Scan a skill directory for text files other than skill.md.

    Args:
        skill_dir: Root directory of the skill.

    Returns:
        Sorted list of (filename, relative_path) tuples for each support file.
    """
    support: list[tuple[str, str]] = []
    for entry in sorted(skill_dir.rglob("*")):
        if not entry.is_file():
            continue
        if entry.name.lower() == _SKILL_MD:
            continue
        try:
            entry.read_text(encoding="utf-8", errors="strict")
        except (UnicodeDecodeError, ValueError):
            continue
        rel = str(entry.relative_to(skill_dir))
        support.append((entry.name, rel))
    return support


def _find_skill_file(directory: Path) -> Path | None:
    """Find the skill definition file in a directory (case-insensitive)."""
    for child in directory.iterdir():
        if child.is_file() and child.name.lower() == _SKILL_MD:
            return child
    return None


def load_skills_from_directory(skills_dir: str | Path) -> list[Skill]:
    """Load all skill definitions from a directory of skill subdirectories.

    Each immediate subdirectory of ``skills_dir`` is treated as a skill.
    The subdirectory must contain a ``skill.md`` or ``SKILL.md`` file with
    YAML frontmatter.
    The subdirectory path is stored as ``source_path`` so that ``load_content``
    can read all files in it on demand.

    Args:
        skills_dir: Root directory containing skill subdirectories.

    Returns:
        List of parsed Skill objects.
    """
    skills_path = Path(skills_dir)
    if not skills_path.is_dir():
        logger.warning("Skills directory does not exist: %s", skills_dir)
        return []

    skills: list[Skill] = []
    for child in sorted(skills_path.iterdir()):
        if not child.is_dir():
            continue
        skill_file = _find_skill_file(child)
        if skill_file is None:
            logger.debug("Skipping directory without skill.md: %s", child)
            continue
        skill = _parse_skill_directory(child, skill_file)
        if skill is not None:
            skills.append(skill)

    logger.info("Loaded %d skills from %s", len(skills), skills_dir)
    return skills


def _parse_skill_directory(skill_dir: Path, skill_file: Path) -> Skill | None:
    """Parse frontmatter from a skill directory's skill definition file.

    Args:
        skill_dir: Path to the skill directory (stored as source_path).
        skill_file: Path to the skill definition file within the directory.

    Returns:
        Parsed Skill with source_path pointing to the directory,
        or None if the file is malformed.
    """
    try:
        post = frontmatter.load(str(skill_file))
    except Exception:
        logger.warning("Cannot read or parse skill file: %s", skill_file)
        return None

    name = post.metadata.get("name")
    description = post.metadata.get("description", "")
    if not name:
        logger.warning("Skill file missing 'name' in frontmatter: %s", skill_file)
        return None

    support_files = _discover_support_files(skill_dir)

    return Skill(
        name=name,
        description=description,
        source_path=str(skill_dir),
        support_files=support_files,
    )


class SkillsRAG(HybridRAGBase):
    """Hybrid RAG system for skill selection using dense and sparse retrieval."""

    _COLLECTION = "skills"

    _MAX_TOP_K = 20

    def __init__(
        self,
        encode_fn: Callable[[str], list[float]],
        alpha: float = 0.8,
        threshold: float = 0.01,
    ) -> None:
        """Initialize the SkillsRAG system.

        Args:
            encode_fn: Function that encodes text into an embedding vector.
            alpha: Weight for dense vs sparse (1.0 = full dense, 0.0 = full sparse).
            threshold: Minimum similarity score to accept a skill match.
        """
        super().__init__(
            collection=self._COLLECTION,
            encode_fn=encode_fn,
            alpha=alpha,
            top_k=self._MAX_TOP_K,
            threshold=threshold,
        )
        self._skills: dict[str, Skill] = {}

    def populate_skills(self, skills: list[Skill]) -> None:
        """Index skills for hybrid retrieval.

        Args:
            skills: List of Skill objects to index.
        """
        ids: list[str] = []
        docs: list[str] = []
        vectors: list[list[float]] = []

        for skill in skills:
            text = f"{skill.name} {skill.description}"
            ids.append(skill.source_path)
            docs.append(text)
            vectors.append(self._encode(text))
            self._skills[skill.source_path] = skill

        self._index_documents(ids, docs, vectors)
        self.top_k = min(len(self._skills), self._MAX_TOP_K)
        logger.info("Indexed %d skills for retrieval", len(skills))

    def retrieve_skill(self, query: str) -> tuple[Skill | None, float]:
        """Retrieve the best matching skill for a query.

        Args:
            query: User query to match against indexed skills.

        Returns:
            Tuple of (best matching Skill, confidence score). Skill is None
            when no skill exceeds the threshold.
        """
        if not self._skills:
            return None, 0.0

        q_vec = self._encode(query)
        dense, _, _ = self._dense_scores(q_vec, self.top_k)
        sparse, _ = self._sparse_scores(query)
        fused = self._fuse_scores(dense, sparse, self.alpha, self.top_k)

        if not fused:
            return None, 0.0

        best_id = next(iter(fused))
        best_score = fused[best_id]

        if best_score < self.threshold:
            logger.debug(
                "Best skill '%s' scored %.3f, below threshold %.3f",
                best_id,
                best_score,
                self.threshold,
            )
            return None, best_score

        skill = self._skills.get(best_id)
        if skill is None:
            return None, 0.0

        logger.debug(
            "Selected skill '%s' with score %.3f for query: %s",
            skill.name,
            best_score,
            query[:80],
        )
        return skill, best_score
