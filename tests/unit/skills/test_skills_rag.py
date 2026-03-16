"""Unit tests for skills_rag module."""

from pathlib import Path

import pytest

from ols.src.skills.skills_rag import Skill, SkillsRAG, load_skills_from_directory

DIMENSION = 8


def _fake_encode(text: str) -> list[float]:
    """Deterministic encode: sum of char ordinals spread across DIMENSION dims."""
    total = sum(ord(c) for c in text)
    return [(total + i) / 1000.0 for i in range(DIMENSION)]


def _make_skill(name: str, description: str, source_path: str = "") -> Skill:
    """Create a Skill with minimal fields for testing."""
    return Skill(
        name=name,
        description=description,
        source_path=source_path or f"skills/{name}",
    )


def _make_rag(**kwargs: object) -> SkillsRAG:
    """Create a SkillsRAG with fake encode and sensible test defaults."""
    defaults: dict = {
        "encode_fn": _fake_encode,
        "alpha": 0.5,
        "threshold": 0.0,
    }
    defaults.update(kwargs)
    return SkillsRAG(**defaults)


def _sample_skills() -> list[Skill]:
    """Return a fixed set of skills for testing."""
    return [
        _make_skill(
            "pod-failure-diagnosis",
            "Troubleshoot CrashLoopBackOff, ImagePullBackOff, Pending, Error, or OOMKilled "
            "status. Use when a workload keeps restarting, fails to start, or is crash-looping.",
        ),
        _make_skill(
            "degraded-operator-recovery",
            "Troubleshoot ClusterOperator in Degraded, Unavailable, or not Progressing state. "
            "Use when operator status shows error conditions, reconciliation failures, or "
            "degraded health checks.",
        ),
        _make_skill(
            "node-not-ready",
            "Troubleshoot NotReady or SchedulingDisabled node status. Use when a node is down, "
            "unschedulable, or needs to be drained and restored.",
        ),
        _make_skill(
            "route-ingress-troubleshooting",
            "Troubleshoot Route or Ingress connectivity failures. Use when traffic returns "
            "502, 503, connection refused, or the endpoint is not reachable externally.",
        ),
        _make_skill(
            "namespace-troubleshooting",
            "Troubleshoot namespace stuck in Terminating state, ResourceQuota exhaustion, or "
            "RBAC permission denied errors. Use when resources cannot be created or forbidden "
            "errors occur.",
        ),
    ]


class TestSkillDataclass:
    """Tests for the Skill dataclass."""

    def test_skill_is_frozen(self) -> None:
        """Verify Skill instances are immutable."""
        skill = _make_skill("test", "desc")
        with pytest.raises(AttributeError):
            skill.name = "changed"  # type: ignore[misc]

    def test_skill_fields(self) -> None:
        """Verify all fields are accessible."""
        skill = _make_skill("test-skill", "A test skill")
        assert skill.name == "test-skill"
        assert skill.description == "A test skill"
        assert skill.source_path == "skills/test-skill"


class TestLoadSkillsFromDirectory:
    """Tests for load_skills_from_directory."""

    def test_loads_skills_from_valid_directory(self, tmp_path: Path) -> None:
        """Verify skills are loaded from a directory with valid skill files."""
        skill_dir = tmp_path / "pod-diagnosis"
        skill_dir.mkdir()
        (skill_dir / "skill.md").write_text(
            "---\nname: pod-diagnosis\ndescription: Diagnose pods\n"
            "---\n\n# Pod Diagnosis\n\nWorkflow here.",
            encoding="utf-8",
        )
        skills = load_skills_from_directory(tmp_path)
        assert len(skills) == 1
        assert skills[0].name == "pod-diagnosis"
        assert skills[0].description == "Diagnose pods"
        assert skills[0].source_path == str(skill_dir)

    def test_loads_skill_with_uppercase_filename(self, tmp_path: Path) -> None:
        """Verify SKILL.md (uppercase) is recognised as a valid skill file."""
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: Upper case\n"
            "---\n\n# My Skill\n\nContent.",
            encoding="utf-8",
        )
        skills = load_skills_from_directory(tmp_path)
        assert len(skills) == 1
        assert skills[0].name == "my-skill"

    def test_returns_empty_for_nonexistent_directory(self) -> None:
        """Verify empty list for missing directory."""
        skills = load_skills_from_directory("/nonexistent/path")
        assert skills == []

    def test_skips_dirs_without_skill_md(self, tmp_path: Path) -> None:
        """Verify directories without skill.md are skipped."""
        empty_dir = tmp_path / "no-skill-file"
        empty_dir.mkdir()
        (empty_dir / "readme.md").write_text("Not a skill.", encoding="utf-8")
        skills = load_skills_from_directory(tmp_path)
        assert skills == []

    def test_skips_files_without_frontmatter(self, tmp_path: Path) -> None:
        """Verify skill.md without YAML frontmatter is skipped."""
        skill_dir = tmp_path / "bad-skill"
        skill_dir.mkdir()
        (skill_dir / "skill.md").write_text(
            "# No Frontmatter\n\nJust markdown.",
            encoding="utf-8",
        )
        skills = load_skills_from_directory(tmp_path)
        assert skills == []

    def test_skips_files_with_missing_name(self, tmp_path: Path) -> None:
        """Verify skill.md without 'name' in frontmatter is skipped."""
        skill_dir = tmp_path / "no-name"
        skill_dir.mkdir()
        (skill_dir / "skill.md").write_text(
            "---\ndescription: Missing name field\n---\n\n# Content",
            encoding="utf-8",
        )
        skills = load_skills_from_directory(tmp_path)
        assert skills == []

    def test_skips_invalid_yaml(self, tmp_path: Path) -> None:
        """Verify skill.md with broken YAML frontmatter is skipped."""
        skill_dir = tmp_path / "bad-yaml"
        skill_dir.mkdir()
        (skill_dir / "skill.md").write_text(
            "---\n: [invalid yaml\n---\n\n# Content",
            encoding="utf-8",
        )
        skills = load_skills_from_directory(tmp_path)
        assert skills == []

    def test_loads_multiple_skills_sorted(self, tmp_path: Path) -> None:
        """Verify multiple skills are loaded and sorted by path."""
        for name in ["beta-skill", "alpha-skill"]:
            d = tmp_path / name
            d.mkdir()
            (d / "skill.md").write_text(
                f"---\nname: {name}\ndescription: Skill {name}\n---\n\n# {name}",
                encoding="utf-8",
            )
        skills = load_skills_from_directory(tmp_path)
        assert len(skills) == 2
        assert skills[0].name == "alpha-skill"
        assert skills[1].name == "beta-skill"

    def test_description_defaults_to_empty(self, tmp_path: Path) -> None:
        """Verify missing description defaults to empty string."""
        skill_dir = tmp_path / "no-desc"
        skill_dir.mkdir()
        (skill_dir / "skill.md").write_text(
            "---\nname: no-desc\n---\n\n# No Description Skill",
            encoding="utf-8",
        )
        skills = load_skills_from_directory(tmp_path)
        assert len(skills) == 1
        assert skills[0].description == ""

    def test_load_content_reads_skill_md_body(self, tmp_path: Path) -> None:
        """Verify load_content reads the skill.md body on demand."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "skill.md").write_text(
            "---\nname: test-skill\ndescription: Test\n"
            "---\n\n# Test Skill\n\nBody content here.",
            encoding="utf-8",
        )
        skills = load_skills_from_directory(tmp_path)
        assert len(skills) == 1
        content = skills[0].load_content()
        assert "Body content here." in content
        assert "# Test Skill" in content

    def test_load_content_includes_extra_files(self, tmp_path: Path) -> None:
        """Verify load_content concatenates additional files from the directory."""
        skill_dir = tmp_path / "multi-file-skill"
        skill_dir.mkdir()
        (skill_dir / "skill.md").write_text(
            "---\nname: multi-file\ndescription: Test\n" "---\n\n# Main Skill Body",
            encoding="utf-8",
        )
        (skill_dir / "checklist.md").write_text(
            "- Step 1\n- Step 2",
            encoding="utf-8",
        )
        (skill_dir / "examples.yaml").write_text(
            "example: value",
            encoding="utf-8",
        )
        skills = load_skills_from_directory(tmp_path)
        content = skills[0].load_content()
        assert "# Main Skill Body" in content
        assert "## checklist.md" in content
        assert "- Step 1" in content
        assert "## examples.yaml" in content
        assert "example: value" in content

    def test_load_content_ignores_non_text_files(self, tmp_path: Path) -> None:
        """Verify load_content skips files with unsupported extensions."""
        skill_dir = tmp_path / "skip-binary"
        skill_dir.mkdir()
        (skill_dir / "skill.md").write_text(
            "---\nname: skip-binary\ndescription: Test\n" "---\n\n# Skill Body",
            encoding="utf-8",
        )
        (skill_dir / "image.png").write_bytes(b"\x89PNG")
        skills = load_skills_from_directory(tmp_path)
        content = skills[0].load_content()
        assert "image.png" not in content

    def test_load_content_recurses_into_subdirectories(self, tmp_path: Path) -> None:
        """Verify load_content includes files from subdirectories."""
        skill_dir = tmp_path / "full-skill"
        skill_dir.mkdir()
        (skill_dir / "skill.md").write_text(
            "---\nname: full-skill\ndescription: Test\n---\n\n# Main Skill Body",
            encoding="utf-8",
        )
        scripts_dir = skill_dir / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "deploy.sh").write_text(
            "#!/bin/bash\necho deploy",
            encoding="utf-8",
        )
        refs_dir = skill_dir / "references"
        refs_dir.mkdir()
        (refs_dir / "guide.md").write_text(
            "# Reference Guide\n\nDetailed steps.",
            encoding="utf-8",
        )
        (refs_dir / "data.bin").write_bytes(b"\x80\x81\x82\xff\xfe")
        skills = load_skills_from_directory(tmp_path)
        content = skills[0].load_content()
        assert "# Main Skill Body" in content
        assert "scripts/deploy.sh" in content
        assert "echo deploy" in content
        assert "references/guide.md" in content
        assert "Reference Guide" in content
        assert "data.bin" not in content


class TestSkillsRAGPopulate:
    """Tests for SkillsRAG.populate_skills."""

    def test_populate_stores_skills(self) -> None:
        """Verify populate_skills stores all skills in the index."""
        rag = _make_rag()
        skills = _sample_skills()
        rag.populate_skills(skills)

        data = rag.store.get_all()
        assert len(data["ids"]) == 5

    def test_populate_keeps_skill_references(self) -> None:
        """Verify populated skills are retrievable by source_path."""
        rag = _make_rag()
        skills = _sample_skills()
        rag.populate_skills(skills)
        assert skills[0].source_path in rag._skills
        assert rag._skills[skills[0].source_path].name == skills[0].name


class TestSkillsRAGRetrieve:
    """Tests for SkillsRAG.retrieve_skill."""

    def _populated_rag(self, **kwargs: object) -> SkillsRAG:
        """Create and populate a SkillsRAG with sample skills."""
        rag = _make_rag(**kwargs)
        rag.populate_skills(_sample_skills())
        return rag

    def test_retrieve_returns_skill_and_score(self) -> None:
        """Verify retrieve returns a (Skill, score) tuple."""
        rag = self._populated_rag()
        skill, score = rag.retrieve_skill("my pod is crashing")
        assert isinstance(skill, Skill)
        assert skill.name
        assert 0.0 <= score <= 1.0

    def test_retrieve_returns_none_when_empty(self) -> None:
        """Verify None returned when no skills are populated."""
        rag = _make_rag()
        skill, score = rag.retrieve_skill("anything")
        assert skill is None
        assert score == 0.0

    def test_retrieve_returns_none_below_threshold(self) -> None:
        """Verify None returned when best score is below threshold."""
        rag = self._populated_rag(threshold=0.99)
        skill, _ = rag.retrieve_skill("completely unrelated query xyz abc")
        assert skill is None

    def test_retrieve_returns_one_of_indexed_skills(self) -> None:
        """Verify retrieval returns a skill that was indexed."""
        rag = self._populated_rag()
        skill, _ = rag.retrieve_skill("operator is degraded and unavailable")
        assert skill is not None
        indexed_names = {s.name for s in _sample_skills()}
        assert skill.name in indexed_names

    def test_retrieve_skill_has_source_path(self) -> None:
        """Verify returned skill has a source_path for on-demand content loading."""
        rag = self._populated_rag()
        skill, _ = rag.retrieve_skill("node not ready")
        assert skill is not None
        assert skill.source_path
