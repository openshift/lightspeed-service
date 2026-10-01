#!/usr/bin/env python3
"""Policy-driven dependency resolver for Hermeto/Cachi2 hermetic builds.

Enforces: RHOAI wheel > PyPI sdist > PyPI wheel (last resort).
Usage: python3 scripts/konflux_resolve.py --profile cpu|cuda [--verbose | --quiet]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
import urllib.parse
import urllib.request
from collections import deque
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger("konflux_resolve")

# ---------------------------------------------------------------------------
# Task 2 — Version parsing and constraint matching (PEP 440)
# ---------------------------------------------------------------------------


def parse_version(version_str: str) -> Version:
    """Parse a PEP 440 version string."""
    return Version(version_str)


def version_satisfies(version: str, constraint: str) -> bool:
    """Check whether *version* satisfies a comma-separated PEP 440 constraint."""
    constraint = constraint.strip()
    if not constraint:
        return True

    try:
        return Version(version) in SpecifierSet(constraint)
    except (InvalidVersion, InvalidSpecifier):
        return False


def merge_constraints(existing: str | None, new: str) -> str:
    """Merge two constraint strings by comma-joining."""
    if not existing:
        return new
    return f"{existing},{new}"


# ---------------------------------------------------------------------------
# Task 3 — Package name normalization and pyproject.toml parsing
# ---------------------------------------------------------------------------

_NORMALIZE_RE = re.compile(r"[-_.]+")


def normalize_name(name: str) -> str:
    """PEP 503 normalization: lowercase, replace runs of ``-``, ``.``, ``_`` with ``-``."""
    return _NORMALIZE_RE.sub("-", name).lower()


def _parse_dep_string(dep: str) -> tuple[str, str, str]:
    """Parse ``name[extras]>=1.0; marker`` into ``(normalized_name, version_spec, marker)``."""
    marker = ""
    if ";" in dep:
        dep, marker = dep.split(";", 1)
        marker = marker.strip()

    dep = dep.strip()
    # Strip extras: name[extras]>=... → name>=...
    dep = re.sub(r"\[.*?\]", "", dep)

    match = re.match(r"^([A-Za-z0-9][-A-Za-z0-9_.]*)", dep)
    if match is None:
        raise ValueError(f"Cannot parse dependency: {dep!r}")

    name = normalize_name(match.group(1))
    version_spec = dep[match.end() :].strip()

    return name, version_spec, marker


def parse_direct_deps(pyproject_path: str) -> list[tuple[str, str]]:
    """Parse ``[project].dependencies`` from a TOML file.

    Returns ``[(normalized_name, version_spec), ...]``.
    """
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    raw_deps: list[str] = data.get("project", {}).get("dependencies", [])
    result: list[tuple[str, str]] = []
    for dep_str in raw_deps:
        name, spec, _marker = _parse_dep_string(dep_str)
        result.append((name, spec))
    return result


# ---------------------------------------------------------------------------
# Task 4 — PEP 503 simple index parser
# ---------------------------------------------------------------------------


class _LinkCollector(HTMLParser):
    """Collect ``href`` attributes from ``<a>`` tags."""

    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []
        self._texts: list[str] = []
        self._in_a = False
        self.link_texts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            for attr_name, attr_val in attrs:
                if attr_name == "href" and attr_val is not None:
                    self.hrefs.append(attr_val)
            self._in_a = True
            self._texts = []

    def handle_data(self, data: str) -> None:
        if self._in_a:
            self._texts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_a:
            self.link_texts.append("".join(self._texts).strip())
            self._in_a = False


_WHEEL_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][-A-Za-z0-9_.]*?)"
    r"-(?P<version>\d[A-Za-z0-9_.+]*?)"
    r"(?:-(?P<build>\d[A-Za-z0-9_.]*)?)?"
    r"-(?P<python>[A-Za-z0-9_.]+)"
    r"-(?P<abi>[A-Za-z0-9_.]+)"
    r"-(?P<platform>[A-Za-z0-9_.]+)"
    r"\.whl$"
)

_SDIST_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][-A-Za-z0-9_.]*?)"
    r"-(?P<version>\d[A-Za-z0-9_.]*)"
    r"(?:\.tar\.gz|\.zip)$"
)


class SimpleIndexParser:
    """Parse PEP 503 Simple Repository API HTML pages."""

    @staticmethod
    def parse_root(html: str) -> list[str]:
        """Return list of package names from root index page."""
        collector = _LinkCollector()
        collector.feed(html)
        return collector.link_texts

    @staticmethod
    def parse_package_page(html: str) -> list[dict[str, Any]]:
        """Return list of entry dicts from a per-package page.

        Each dict has keys: ``filename``, ``sha256``, ``version``, ``is_wheel``,
        and for wheels: ``python_tag``, ``abi_tag``, ``platform_tag``.
        """
        collector = _LinkCollector()
        collector.feed(html)
        entries: list[dict[str, Any]] = []

        for href, link_text in zip(collector.hrefs, collector.link_texts, strict=False):
            filename = link_text.strip()
            if not filename:
                filename = href.rsplit("/", 1)[-1].split("#")[0]

            sha256 = ""
            if "#sha256=" in href:
                sha256 = href.split("#sha256=", 1)[1]

            whl_m = _WHEEL_RE.match(filename)
            if whl_m:
                entries.append(
                    {
                        "filename": filename,
                        "sha256": sha256,
                        "version": whl_m.group("version"),
                        "is_wheel": True,
                        "python_tag": whl_m.group("python"),
                        "abi_tag": whl_m.group("abi"),
                        "platform_tag": whl_m.group("platform"),
                    }
                )
                continue

            sdist_m = _SDIST_RE.match(filename)
            if sdist_m:
                entries.append(
                    {
                        "filename": filename,
                        "sha256": sha256,
                        "version": sdist_m.group("version"),
                        "is_wheel": False,
                    }
                )

        return entries


# ---------------------------------------------------------------------------
# Task 5 — Wheel compatibility checker
# ---------------------------------------------------------------------------


def _abi3_compatible(python_tag: str, target_ver: tuple[int, int]) -> bool:
    """Check if an abi3 wheel's cpXY tag is compatible with *target_ver*."""
    for sub in python_tag.split("."):
        if sub.startswith("cp") and len(sub) >= 3:
            digits = sub[2:]
            try:
                tag_major = int(digits[0])
                tag_minor = int(digits[1:]) if len(digits) > 1 else 0
                if (tag_major, tag_minor) <= target_ver:
                    return True
            except ValueError:
                pass
    return False


def is_wheel_compatible(
    python_tag: str,
    platform_tag: str,
    target_python: str,
    target_platforms: Sequence[str],
    abi_tag: str = "",
) -> bool:
    """Check if a wheel's tags match the target environment.

    *target_python* is e.g. ``"3.12"``; *target_platforms* is e.g.
    ``["linux_x86_64", "linux_aarch64"]``.
    """
    major, minor = target_python.split(".")
    target_ver = (int(major), int(minor))
    compatible_py = {
        f"cp{major}{minor}",
        f"cp{major}",
        f"py{major}",
        f"py{major}{minor}",
    }

    py_ok = any(sub in compatible_py for sub in python_tag.split("."))
    if not py_ok and abi_tag and "abi3" in abi_tag.split("."):
        py_ok = _abi3_compatible(python_tag, target_ver)
    if not py_ok:
        return False

    # Platform matching: "any" and "none" always match.
    if platform_tag.lower() in ("any", "none"):
        return True

    # A compound platform tag like "manylinux_2_17_x86_64.manylinux2014_x86_64"
    # can contain multiple sub-tags separated by ".". Check each sub-tag and
    # each target platform for a suffix match (the arch part).
    sub_tags = platform_tag.split(".")
    for target in target_platforms:
        # Extract the arch from the target, e.g. "linux_x86_64" → "x86_64"
        arch = target.split("_", 1)[1] if "_" in target else target
        for sub in sub_tags:
            if sub == target or sub.endswith(f"_{arch}"):
                return True

    return False


# ---------------------------------------------------------------------------
# Task 6 — RHOAI index loader
# ---------------------------------------------------------------------------


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Reject redirects from the trusted package index."""

    def redirect_request(
        self,
        _req: Any,
        _fp: Any,
        _code: int,
        _msg: str,
        _headers: Any,
        _newurl: str,
    ) -> None:
        """Stop a redirect before it can reach another host."""
        return None


class RhoaiIndex:
    """RHOAI simple index with lazy per-package fetching.

    The root page is downloaded eagerly (to learn which packages exist),
    but individual package pages are fetched on-demand and cached.
    """

    def __init__(
        self,
        index_url: str,
        python_version: str,
        platforms: Sequence[str],
    ) -> None:
        """Initialize with the RHOAI simple index URL, target Python version, and platforms."""
        parsed = urllib.parse.urlsplit(index_url.rstrip("/") + "/")
        if (
            parsed.scheme != "https"
            or parsed.netloc != "packages.redhat.com"
            or not re.fullmatch(
                r"/api/pypi/public-rhai/rhoai/\d+\.\d+/(?:cpu|cuda)-ubi9/simple/",
                parsed.path,
            )
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(f"Untrusted RHOAI index URL: {index_url}")
        self.index_url = urllib.parse.urlunsplit(
            ("https", "packages.redhat.com", parsed.path, "", "")
        )
        self._opener = urllib.request.build_opener(_NoRedirect())
        self.python_version = python_version
        self.platforms = list(platforms)
        self._parser = SimpleIndexParser()
        self._known_packages: set[str] = set()
        self._packages: dict[str, dict[str, dict[str, tuple[str, str]]]] = {}

    def _fetch_url(self, url: str) -> str:
        """Fetch *url* with retry (3 attempts, exponential backoff)."""
        if url != self.index_url:
            suffix = url.removeprefix(self.index_url)
            if not url.startswith(self.index_url) or not re.fullmatch(
                r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?/", suffix
            ):
                raise ValueError(f"Untrusted RHOAI package URL: {url}")
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                with self._opener.open(url, timeout=30) as resp:
                    return str(resp.read().decode())
            except Exception as exc:
                last_exc = exc
                logger.debug("Fetch %s attempt %d failed: %s", url, attempt + 1, exc)
                time.sleep(2**attempt)
        raise RuntimeError(f"Failed to fetch {url} after 3 attempts") from last_exc

    def load(self) -> None:
        """Download root page to learn which packages exist on RHOAI."""
        root_html = self._fetch_url(self.index_url)
        package_names = self._parser.parse_root(root_html)
        self._known_packages = {normalize_name(n) for n in package_names}
        logger.info("RHOAI index: %d packages available", len(self._known_packages))

    def _ensure_loaded(self, name: str) -> None:
        """Fetch and cache a package page if not already loaded."""
        norm = normalize_name(name)
        if norm in self._packages or norm not in self._known_packages:
            return

        target_platforms = [f"linux_{p}" for p in self.platforms]
        page_url = f"{self.index_url}{norm}/"
        try:
            page_html = self._fetch_url(page_url)
        except Exception as exc:
            logger.warning("Failed to fetch RHOAI page for %s: %s", norm, exc)
            return

        entries = self._parser.parse_package_page(page_html)
        versions: dict[str, dict[str, tuple[str, str]]] = {}
        for entry in entries:
            if not entry["is_wheel"]:
                continue
            if not is_wheel_compatible(
                entry["python_tag"],
                entry["platform_tag"],
                self.python_version,
                target_platforms,
                abi_tag=entry.get("abi_tag", ""),
            ):
                continue

            ver = entry["version"]
            if ver not in versions:
                versions[ver] = {}

            plat = entry["platform_tag"]
            matched_arch = self._match_arch(plat, target_platforms)
            if matched_arch:
                versions[ver][matched_arch] = (entry["filename"], entry["sha256"])

        if versions:
            self._packages[norm] = versions

    def _match_arch(self, platform_tag: str, target_platforms: list[str]) -> str | None:
        """Determine which target arch a platform tag matches."""
        if platform_tag.lower() in ("any", "none"):
            return "any"
        sub_tags = platform_tag.split(".")
        for target in target_platforms:
            arch = target.split("_", 1)[1] if "_" in target else target
            for sub in sub_tags:
                if sub == target or sub.endswith(f"_{arch}"):
                    return target
        return None

    def has_package(self, name: str) -> bool:
        """Return whether the RHOAI index lists this package name."""
        return normalize_name(name) in self._known_packages

    def find_best(self, name: str, constraint: str) -> dict[str, Any] | None:
        """Find latest version satisfying *constraint*.

        Returns ``{"version": str, "platforms": {arch: (filename, sha256)}}``
        or ``None``.
        """
        norm = normalize_name(name)
        self._ensure_loaded(norm)
        versions = self._packages.get(norm)
        if not versions:
            return None

        candidates = [v for v in versions if version_satisfies(v, constraint)]
        if not candidates:
            return None

        best = max(candidates, key=parse_version)
        return {"version": best, "platforms": versions[best]}


# ---------------------------------------------------------------------------
# Task 7 — PEP 508 marker evaluation & PyPI client
# ---------------------------------------------------------------------------

_MARKER_ENV_KEYS = {
    "sys_platform",
    "os_name",
    "platform_system",
    "implementation_name",
    "python_version",
    "platform_machine",
    "extra",
}

_MARKER_COMPARE_RE = re.compile(
    r"""^
    \s*(?P<left>[A-Za-z_][A-Za-z0-9_.]*|'[^']*'|"[^"]*")
    \s*(?P<op>~=|===|==|!=|>=|<=|>|<|not\s+in|in)
    \s*(?P<right>[A-Za-z_][A-Za-z0-9_.]*|'[^']*'|"[^"]*")
    \s*$
    """,
    re.VERBOSE,
)


def _eval_marker(marker: str, python_version: str) -> bool:
    """Simplified PEP 508 marker evaluation for a Linux CPython target."""
    env = {
        "sys_platform": "linux",
        "os_name": "posix",
        "platform_system": "Linux",
        "implementation_name": "cpython",
        "python_version": python_version,
    }

    marker = marker.strip()
    if not marker:
        return True

    or_parts = re.split(r"\s+or\s+", marker)
    for or_part in or_parts:
        and_parts = re.split(r"\s+and\s+", or_part)
        all_true = True
        for expr in and_parts:
            if not _eval_single_marker(expr.strip(), env):
                all_true = False
                break
        if all_true:
            return True
    return False


_MARKER_CMP_OPS: dict[str, Any] = {
    "==": lambda lv, rv: lv == rv,
    "!=": lambda lv, rv: lv != rv,
    ">=": lambda lv, rv: lv >= rv,
    "<=": lambda lv, rv: lv <= rv,
    ">": lambda lv, rv: lv > rv,
    "<": lambda lv, rv: lv < rv,
    "in": lambda lv, rv: lv in rv,
    "not in": lambda lv, rv: lv not in rv,
}


def _eval_single_marker(expr: str, env: dict[str, str]) -> bool:
    """Evaluate a single marker comparison like ``sys_platform == 'linux'``."""
    m = _MARKER_COMPARE_RE.match(expr)
    if m is None:
        return True

    left_raw = m.group("left").strip("'\"")
    right_raw = m.group("right").strip("'\"")
    op = re.sub(r"\s+", " ", m.group("op"))

    if left_raw in env:
        lval, rval = env[left_raw], right_raw
    elif right_raw in env:
        lval, rval = left_raw, env[right_raw]
    else:
        return True

    check = _MARKER_CMP_OPS.get(op)
    return bool(check(lval, rval)) if check else True


class PypiClient:
    """Lazy, per-package PyPI client with caching."""

    def __init__(self, python_version: str, platforms: Sequence[str]) -> None:
        """Initialize with the target Python version and platforms."""
        self.python_version = python_version
        self.platforms = list(platforms)
        self._parser = SimpleIndexParser()
        self._info_cache: dict[str, dict[str, Any]] = {}
        self._requires_cache: dict[str, list[tuple[str, str]]] = {}

    def _fetch_url(self, url: str) -> str:
        """Fetch *url* with retry (3 attempts, exponential backoff)."""
        parsed = urllib.parse.urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "pypi.org"
            or not re.fullmatch(
                r"/(?:simple/[a-z0-9][a-z0-9-]*/|pypi/[a-z0-9][a-z0-9-]*/[^/]+/json)",
                parsed.path,
            )
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(f"Untrusted PyPI URL: {url}")
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                opener = urllib.request.build_opener(_NoRedirect())
                with opener.open(url, timeout=30) as resp:
                    return str(resp.read().decode())
            except Exception as exc:
                last_exc = exc
                logger.debug("Fetch %s attempt %d failed: %s", url, attempt + 1, exc)
                time.sleep(2**attempt)
        raise RuntimeError(f"Failed to fetch {url} after 3 attempts") from last_exc

    def get_package_info(self, name: str) -> dict[str, dict[str, Any]]:
        """Fetch and cache the simple index page for *name*.

        Returns ``{version: {"has_sdist": bool, "sdist_hashes": [...],
        "wheel_hashes": [...], "wheel_files": [...]}}``.
        """
        norm = normalize_name(name)
        if norm in self._info_cache:
            return self._info_cache[norm]

        url = f"https://pypi.org/simple/{norm}/"
        html = self._fetch_url(url)
        entries = self._parser.parse_package_page(html)

        info: dict[str, dict[str, Any]] = {}
        for entry in entries:
            ver = entry["version"]
            if ver not in info:
                info[ver] = {
                    "has_sdist": False,
                    "sdist_hashes": [],
                    "wheel_hashes": [],
                    "wheel_files": [],
                }
            if entry["is_wheel"]:
                if entry["sha256"]:
                    info[ver]["wheel_hashes"].append(entry["sha256"])
                info[ver]["wheel_files"].append(entry["filename"])
            else:
                info[ver]["has_sdist"] = True
                if entry["sha256"]:
                    info[ver]["sdist_hashes"].append(entry["sha256"])

        self._info_cache[norm] = info
        return info

    def get_requires_dist(self, name: str, version: str) -> list[tuple[str, str]]:
        """Fetch ``Requires-Dist`` from PyPI JSON API.

        Returns ``[(dep_name, spec), ...]``, filtering out extras and markers
        that don't match the target environment.
        """
        cache_key = f"{normalize_name(name)}=={version}"
        if cache_key in self._requires_cache:
            return self._requires_cache[cache_key]

        url = f"https://pypi.org/pypi/{name}/{version}/json"
        text = self._fetch_url(url)
        data = json.loads(text)

        requires_dist: list[str] = data.get("info", {}).get("requires_dist") or []
        result: list[tuple[str, str]] = []

        for dep_str in requires_dist:
            dep_name, spec, marker = _parse_dep_string(dep_str)

            if marker:
                stripped = marker.replace(" ", "")
                if "extra==" in stripped or "extra ==" in marker:
                    continue

            if marker and not _eval_marker(marker, self.python_version):
                continue

            result.append((dep_name, spec))

        self._requires_cache[cache_key] = result
        return result

    def find_best(self, name: str, constraint: str) -> dict[str, Any] | None:
        """Find latest version on PyPI satisfying *constraint*.

        Returns ``{"version": str, "has_sdist": bool, "sdist_hashes": [...],
        "wheel_hashes": [...], "wheel_files": [...]}`` or ``None``.
        """
        info = self.get_package_info(name)
        candidates = [v for v in info if version_satisfies(v, constraint)]
        if not candidates:
            return None

        best = max(candidates, key=parse_version)
        return {"version": best, **info[best]}


# ---------------------------------------------------------------------------
# Task 8 — Dependency resolver (BFS graph walk)
# ---------------------------------------------------------------------------


class Resolver:
    """BFS dependency resolver enforcing RHOAI-first policy."""

    def __init__(
        self,
        rhoai: RhoaiIndex,
        pypi: PypiClient,
        wheel_only_packages: set[str] | None = None,
    ) -> None:
        """Initialize with RHOAI and PyPI clients and the wheel-only package set."""
        self.rhoai = rhoai
        self.pypi = pypi
        self.wheel_only = {normalize_name(p) for p in (wheel_only_packages or set())}
        self.fallback_reasons: dict[str, str] = {}

    # pylint: disable-next=too-many-branches
    def resolve(self, direct_deps: list[tuple[str, str]]) -> dict[str, dict[str, Any]]:
        """Resolve all transitive dependencies via BFS.

        Returns ``{name: {"version": str, "source": "rhoai"|"pypi", ...}}``.
        """
        resolved: dict[str, dict[str, Any]] = {}
        constraints: dict[str, str] = {}
        queue: deque[tuple[str, str]] = deque()

        for name, spec in direct_deps:
            norm = normalize_name(name)
            constraints[norm] = (
                merge_constraints(constraints.get(norm), spec)
                if spec
                else constraints.get(norm, "")
            )
            queue.append((norm, constraints[norm]))

        visited_queue: set[str] = set()

        while queue:
            name, _constraint_at_enqueue = queue.popleft()
            norm = normalize_name(name)

            if norm in resolved:
                current_ver = resolved[norm]["version"]
                if version_satisfies(current_ver, constraints.get(norm, "")):
                    continue
                if resolved[norm]["source"] == "rhoai":
                    logger.info(
                        "Constraint conflict for %s (RHOAI %s); falling back to PyPI",
                        norm,
                        current_ver,
                    )
                    del resolved[norm]
                    self.fallback_reasons[norm] = (
                        f"RHOAI version {current_ver} conflicts with "
                        f"constraint {constraints.get(norm, '')}"
                    )
                else:
                    raise RuntimeError(
                        f"Constraint conflict for {norm}: resolved {current_ver} "
                        f"does not satisfy {constraints.get(norm, '')}"
                    )

            constraint = constraints.get(norm, "")

            rhoai_result = self.rhoai.find_best(norm, constraint)
            if rhoai_result is not None:
                resolved[norm] = {
                    "version": rhoai_result["version"],
                    "source": "rhoai",
                    "platforms": rhoai_result["platforms"],
                }
            else:
                pypi_result = self.pypi.find_best(norm, constraint)
                if pypi_result is None:
                    raise RuntimeError(
                        f"Cannot resolve {norm} with constraint "
                        f"{constraint!r}: not found on RHOAI or PyPI"
                    )
                if norm not in self.fallback_reasons:
                    if self.rhoai.has_package(norm):
                        self.fallback_reasons[norm] = (
                            f"RHOAI has {norm} but no version satisfies {constraint!r}"
                        )
                    else:
                        self.fallback_reasons[norm] = "not in RHOAI index"
                resolved[norm] = {
                    "version": pypi_result["version"],
                    "source": "pypi",
                    "has_sdist": pypi_result["has_sdist"],
                    "sdist_hashes": pypi_result["sdist_hashes"],
                    "wheel_hashes": pypi_result["wheel_hashes"],
                    "wheel_files": pypi_result["wheel_files"],
                }

            pinned_version = resolved[norm]["version"]
            visit_key = f"{norm}=={pinned_version}"
            if visit_key in visited_queue:
                continue
            visited_queue.add(visit_key)

            try:
                trans_deps = self.pypi.get_requires_dist(norm, pinned_version)
            except Exception as exc:
                logger.warning(
                    "Could not fetch deps for %s==%s: %s",
                    norm,
                    pinned_version,
                    exc,
                )
                continue

            for dep_name, dep_spec in trans_deps:
                dep_norm = normalize_name(dep_name)
                if dep_spec:
                    constraints[dep_norm] = merge_constraints(
                        constraints.get(dep_norm),
                        dep_spec,
                    )
                elif dep_norm not in constraints:
                    constraints[dep_norm] = ""
                queue.append((dep_norm, constraints[dep_norm]))

        return resolved


# ---------------------------------------------------------------------------
# Task 9 — Classifier
# ---------------------------------------------------------------------------


def classify_packages(
    resolved: dict[str, dict[str, Any]],
    wheel_only: set[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Classify resolved packages into output buckets.

    Returns ``{"rhoai_wheel": {...}, "pypi_sdist": {...}, "pypi_wheel": {...}}``.
    """
    wheel_only_norm = {normalize_name(p) for p in wheel_only}

    buckets: dict[str, dict[str, dict[str, Any]]] = {
        "rhoai_wheel": {},
        "pypi_sdist": {},
        "pypi_wheel": {},
    }

    for name, info in resolved.items():
        norm = normalize_name(name)
        if info["source"] == "rhoai":
            buckets["rhoai_wheel"][norm] = info
        elif norm in wheel_only_norm:
            buckets["pypi_wheel"][norm] = info
        elif info.get("has_sdist", False):
            buckets["pypi_sdist"][norm] = info
        else:
            logger.warning(
                "Package %s==%s has no sdist on PyPI; auto-promoting to "
                "pypi_wheel. Consider adding it to pypi_wheel_only.txt.",
                norm,
                info["version"],
            )
            buckets["pypi_wheel"][norm] = info

    return buckets


# ---------------------------------------------------------------------------
# Task 10 — Output writer: hashed requirements files
# ---------------------------------------------------------------------------


def write_hashed_requirements(
    packages: dict[str, dict[str, Any]],
    output_path: str,
    index_url: str,
) -> None:
    """Write a pip-compatible hashed requirements file.

    *packages* maps ``{name: info}`` where *info* has the fields produced by
    the resolver (``version``, and either ``platforms`` for RHOAI or
    ``sdist_hashes`` / ``wheel_hashes`` / ``wheel_files`` for PyPI).
    """
    lines: list[str] = [f"--index-url {index_url}\n"]

    for name in sorted(packages):
        info = packages[name]
        version = info["version"]

        hashes: set[str] = set()

        # RHOAI packages store hashes per platform
        if "platforms" in info:
            for _, sha in info["platforms"].values():
                if sha:
                    hashes.add(sha)

        # PyPI packages store hashes in flat lists
        for key in ("sdist_hashes", "wheel_hashes"):
            for sha in info.get(key, []):
                if sha:
                    hashes.add(sha)
        # wheel_files entries are filenames, not hashes — but the info dict
        # may carry per-file hashes via wheel_hashes already.  Nothing extra
        # to extract here.

        sorted_hashes = sorted(hashes)
        if not sorted_hashes:
            raise RuntimeError(
                f"No hashes collected for {name}=={version} while writing {output_path}"
            )
        lines.append(f"{name}=={version} \\\n")
        hash_lines = [f"    --hash=sha256:{h}" for h in sorted_hashes]
        lines.append(" \\\n".join(hash_lines) + "\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


# ---------------------------------------------------------------------------
# Task 11 — Tekton YAML patching
# ---------------------------------------------------------------------------


def patch_tekton_packages(yaml_path: str, package_names: list[str]) -> None:
    """Replace the ``"packages": "..."`` value in a Tekton pipeline YAML."""
    with open(yaml_path, encoding="utf-8") as f:
        content = f.read()

    sorted_names = sorted(package_names)
    replacement = f'"packages": "{",".join(sorted_names)}"'
    content = re.sub(r'"packages":\s*"[^"]*"', replacement, content)

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Task 12 — Config loading
# ---------------------------------------------------------------------------

KONFLUX_DIR = ".konflux"


def load_config(profiles_path: str, profile_name: str) -> dict[str, Any]:
    """Load and merge ``[common]`` + ``[profiles.<name>]`` from a TOML file.

    Returns a dict with keys: ``python_version``, ``platforms``,
    ``bootstrap_packages``, ``extras``, ``rhoai_index_url``, ``output_suffix``,
    ``tekton_files``.
    """
    with open(profiles_path, "rb") as f:
        data = tomllib.load(f)

    common = dict(data.get("common", {}))
    profiles = data.get("profiles", {})

    if profile_name not in profiles:
        raise KeyError(
            f"Profile {profile_name!r} not found in {profiles_path}. "
            f"Available: {', '.join(profiles)}"
        )

    merged = {**common, **profiles[profile_name]}
    return merged


def load_wheel_only(path: str) -> set[str]:
    """Load ``.konflux/pypi_wheel_only.txt`` — one package name per line.

    Skips blank lines and ``#`` comments.  Returns normalized names.
    """
    names: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            names.add(normalize_name(line))
    return names


# ---------------------------------------------------------------------------
# Hybrid resolution: uv pip compile + RHOAI reclassification
# ---------------------------------------------------------------------------

_UV_COMPILED_RE = re.compile(r"^([a-zA-Z0-9][a-zA-Z0-9._-]*)([=<>!~].*)?$")

UV_IMAGE = "quay.io/syedriko/uv:prefer-index"


def _container_runtime() -> str:
    """Find an installed container runtime for the dependency resolver."""
    if shutil.which("podman"):
        return "podman"
    if shutil.which("docker"):
        return "docker"
    raise RuntimeError("Neither podman nor docker is installed")


def uv_resolve(
    python_version: str,
    rhoai_index_url: str,
    suffix: str,
    extras: Sequence[str] = (),
) -> dict[str, dict[str, Any]]:
    """Run ``uv pip compile --index-strategy prefer-index`` to resolve deps.

    Returns ``{normalized_name: {"version": str, "index": str}}``
    where *index* is the URL of the index the package was resolved from.
    """
    overrides_name = "requirements.overrides.txt"
    if suffix:
        overrides_name = f"requirements.overrides{suffix}.txt"
    overrides_file = os.path.join(KONFLUX_DIR, overrides_name)
    cmd = [
        _container_runtime(),
        "run",
        "--rm",
        "--volume",
        f"{os.getcwd()}:/io:ro,Z",
        "--workdir",
        "/io",
        UV_IMAGE,
        "pip",
        "compile",
        "pyproject.toml",
        "--python-platform",
        "x86_64-manylinux_2_28",
        "--python-version",
        python_version,
        "--refresh",
        "--index",
        rhoai_index_url,
        "--default-index",
        "https://pypi.org/simple/",
        "--index-strategy",
        "prefer-index",
        "--emit-index-annotation",
        "--no-sources",
    ]
    if os.path.exists(overrides_file):
        cmd += ["--override", overrides_file]
    for extra in extras:
        cmd += ["--extra", extra]

    logger.debug("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Failed:")
        print(e.stderr)
        sys.exit(1)

    resolved: dict[str, dict[str, Any]] = {}
    current_package: str | None = None

    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("-"):
            continue

        m = _UV_COMPILED_RE.match(line)
        if m and not line.startswith("#"):
            name = normalize_name(m.group(1))
            version_spec = (m.group(2) or "").strip()
            if version_spec.startswith("=="):
                version = version_spec[2:]
            else:
                version = version_spec.lstrip("=")
            if version:
                current_package = name
                resolved[name] = {"version": version, "index": ""}
        elif "# from " in line and current_package:
            index_url = line.split("# from ", 1)[1].strip()
            resolved[current_package]["index"] = index_url

    logger.info("uv resolved %d packages", len(resolved))
    return resolved


def reclassify_with_rhoai(
    uv_resolved: dict[str, str],
    rhoai: RhoaiIndex,
) -> dict[str, dict[str, Any]]:
    """Reclassify uv-resolved packages using RHOAI-first policy.

    For each package, if RHOAI has a compatible wheel at the resolved version,
    classify as ``source=rhoai``; otherwise ``source=pypi``.
    """
    result: dict[str, dict[str, Any]] = {}
    rhoai_count = 0
    pypi_count = 0

    for name, version in sorted(uv_resolved.items()):
        rhoai_match = rhoai.find_best(name, f"=={version}")
        if rhoai_match and rhoai_match["version"] == version:
            result[name] = {
                "version": version,
                "source": "rhoai",
                "platforms": rhoai_match["platforms"],
            }
            rhoai_count += 1
            logger.debug("RHOAI: %s==%s", name, version)
        else:
            result[name] = {
                "version": version,
                "source": "pypi",
                "has_sdist": True,
                "sdist_hashes": [],
                "wheel_hashes": [],
                "wheel_files": [],
            }
            pypi_count += 1
            if rhoai.has_package(name):
                logger.info(
                    "PyPI: %s==%s (RHOAI has package but not version %s)",
                    name,
                    version,
                    version,
                )
            else:
                logger.debug("PyPI: %s==%s (not in RHOAI)", name, version)

    logger.info("Reclassified: %d RHOAI, %d PyPI", rhoai_count, pypi_count)
    return result


def _fetch_hashes_for_pypi_packages(
    resolved: dict[str, dict[str, Any]],
    pypi: PypiClient,
) -> None:
    """Populate sdist/wheel hashes for PyPI-sourced packages in-place."""
    for name, info in resolved.items():
        if info["source"] != "pypi":
            continue
        version = info["version"]
        try:
            pkg_info = pypi.get_package_info(name)
        except Exception as exc:
            logger.warning("Could not fetch PyPI info for %s: %s", name, exc)
            continue
        ver_info = pkg_info.get(version, {})
        info["has_sdist"] = ver_info.get("has_sdist", False)
        info["sdist_hashes"] = ver_info.get("sdist_hashes", [])
        info["wheel_hashes"] = ver_info.get("wheel_hashes", [])
        info["wheel_files"] = ver_info.get("wheel_files", [])


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def _strip_rhoai_duplicates_from_build_deps(
    build_file: str,
    rhoai_names: set[str],
) -> None:
    """Remove packages from build deps file that are already provided elsewhere.

    Strips packages that already exist as RHOAI wheels or bootstrap packages
    to prevent hermeto from fetching PyPI wheels for them, which would cause
    EC policy violations (binary=true on PyPI-sourced packages).
    """
    with open(build_file, encoding="utf-8") as f:
        lines = f.readlines()

    output: list[str] = []
    skip = False
    for line in lines:
        if line and line[0].isalpha():
            pkg_name = normalize_name(line.split("==")[0].strip())
            skip = pkg_name in rhoai_names
        elif line.startswith("#") and not skip:
            skip = False

        if not skip:
            output.append(line)

    with open(build_file, "w", encoding="utf-8") as f:
        f.writelines(output)


def _generate_hermetic_requirements(
    rhoai: RhoaiIndex,
    bootstrap_packages: Sequence[str],
    platforms: Sequence[str],
    rhoai_index_url: str,
    suffix: str,
) -> None:
    """Generate pinned, hashed build tools from the RHOAI index."""
    hermetic_file = os.path.join(KONFLUX_DIR, f"requirements.hermetic{suffix}.txt")
    packages: dict[str, dict[str, Any]] = {}

    for package in bootstrap_packages:
        name = normalize_name(package)
        match = rhoai.find_best(name, ">=0")
        if match is None:
            raise RuntimeError(f"No RHOAI wheel for bootstrap package {name}")
        version = match["version"]
        available = match["platforms"]
        missing = [
            platform
            for platform in platforms
            if "any" not in available and f"linux_{platform}" not in available
        ]
        if missing:
            raise RuntimeError(
                f"No RHOAI wheel for {name}=={version} on {', '.join(missing)}; "
                f"update {hermetic_file}"
            )
        packages[name] = match

    write_hashed_requirements(packages, hermetic_file, rhoai_index_url)
    logger.info("Wrote %s (%d bootstrap packages)", hermetic_file, len(packages))


# pylint: disable-next=too-many-branches,too-many-statements
def main() -> None:
    """Resolve dependencies with RHOAI-first policy and write Hermeto output files."""
    parser = argparse.ArgumentParser(
        description="Policy-driven dependency resolver for Hermeto/Cachi2 builds."
    )
    parser.add_argument("--profile", required=True, help="Build profile (cpu|cuda)")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", action="store_true", help="Verbose logging")
    verbosity.add_argument("--quiet", action="store_true", help="Errors only")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif args.quiet:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)

    profiles_path = os.path.join(KONFLUX_DIR, "profiles.toml")
    config = load_config(profiles_path, args.profile)

    wheel_only_path = os.path.join(KONFLUX_DIR, "pypi_wheel_only.txt")
    wheel_only = load_wheel_only(wheel_only_path)

    python_version = config["python_version"]
    platforms = config["platforms"]
    rhoai_index_url = config["rhoai_index_url"]
    suffix = config.get("output_suffix", "")
    tekton_files = config.get("tekton_files", [])
    bootstrap_packages = config.get("bootstrap_packages", [])

    # Step 1: Resolve via uv with prefer-index strategy
    logger.info("Running uv pip compile --index-strategy prefer-index …")
    uv_resolved = uv_resolve(
        python_version,
        rhoai_index_url,
        suffix,
        config.get("extras", []),
    )

    # Step 2: Build resolved dict from uv output + index annotations
    resolved: dict[str, dict[str, Any]] = {}
    for name, info in uv_resolved.items():
        index = info["index"]
        is_rhoai = "packages.redhat.com" in index if index else False
        if is_rhoai:
            resolved[name] = {
                "version": info["version"],
                "source": "rhoai",
                "platforms": {},
            }
        else:
            resolved[name] = {
                "version": info["version"],
                "source": "pypi",
                "has_sdist": True,
                "sdist_hashes": [],
                "wheel_hashes": [],
                "wheel_files": [],
            }

    rhoai_count = sum(1 for v in resolved.values() if v["source"] == "rhoai")
    pypi_count = len(resolved) - rhoai_count
    logger.info("Classified: %d RHOAI, %d PyPI", rhoai_count, pypi_count)

    # Step 3: Fetch hashes — RHOAI from the RHOAI index, PyPI from PyPI
    logger.info("Fetching hashes …")
    rhoai = RhoaiIndex(rhoai_index_url, python_version, platforms)
    rhoai.load()
    _generate_hermetic_requirements(
        rhoai,
        bootstrap_packages,
        platforms,
        rhoai_index_url,
        suffix,
    )
    for name, info in resolved.items():
        if info["source"] == "rhoai":
            match = rhoai.find_best(name, f"=={info['version']}")
            if match:
                info["platforms"] = match["platforms"]

    pypi = PypiClient(python_version, platforms)
    _fetch_hashes_for_pypi_packages(resolved, pypi)

    # Step 5: Classify into buckets
    buckets = classify_packages(resolved, wheel_only)

    # Step 6: Write hashed requirements files
    write_hashed_requirements(
        buckets["rhoai_wheel"],
        os.path.join(KONFLUX_DIR, f"requirements.hashes.wheel{suffix}.txt"),
        rhoai_index_url,
    )
    write_hashed_requirements(
        buckets["pypi_sdist"],
        os.path.join(KONFLUX_DIR, f"requirements.hashes.source{suffix}.txt"),
        "https://pypi.org/simple",
    )
    write_hashed_requirements(
        buckets["pypi_wheel"],
        os.path.join(KONFLUX_DIR, f"requirements.hashes.wheel.pypi{suffix}.txt"),
        "https://pypi.org/simple",
    )

    # Step 7: Build dependencies via pybuild-deps
    sdist_names = list(buckets["pypi_sdist"].keys())
    build_output = os.path.join(KONFLUX_DIR, f"requirements-build{suffix}.txt")
    if sdist_names:
        tmp_sdist_file = os.path.join(KONFLUX_DIR, f"_tmp_sdist_list{suffix}.txt")
        try:
            with open(tmp_sdist_file, "w", encoding="utf-8") as f:
                for name in sorted(sdist_names):
                    info = buckets["pypi_sdist"][name]
                    f.write(f"{name}=={info['version']}\n")
            build_deps_executable = shutil.which("pybuild-deps")
            if build_deps_executable is None:
                raise RuntimeError("pybuild-deps is not installed")
            try:
                subprocess.run(
                    [
                        build_deps_executable,
                        "compile",
                        f"--output-file={build_output}",
                        tmp_sdist_file,
                    ],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                print("Failed:")
                print(e.stderr)
                sys.exit(1)
        finally:
            if os.path.exists(tmp_sdist_file):
                os.remove(tmp_sdist_file)

        # Strip build deps that duplicate RHOAI wheel packages or bootstrap
        # packages to avoid hermeto fetching them as binary from PyPI
        # (EC policy violation).
        rhoai_names = set(buckets["rhoai_wheel"].keys())
        bootstrap_names = {normalize_name(p) for p in bootstrap_packages}
        _strip_rhoai_duplicates_from_build_deps(
            build_output,
            rhoai_names | bootstrap_names,
        )
    else:
        with open(build_output, "w", encoding="utf-8") as f:
            f.write("# No sdist packages — no build dependencies needed.\n")

    # Step 9: Patch Tekton pipelines
    wheel_package_names = sorted(
        set(buckets["rhoai_wheel"])
        | set(buckets["pypi_wheel"])
        | {normalize_name(p) for p in bootstrap_packages}
    )
    for tekton_file in tekton_files:
        if os.path.exists(tekton_file):
            patch_tekton_packages(tekton_file, wheel_package_names)
            logger.info("Patched %s", tekton_file)
        else:
            logger.warning("Tekton file not found: %s", tekton_file)

    # Summary
    total = len(resolved)
    print(f"\n{'=' * 60}")
    print(f"Resolution complete ({args.profile} profile)")
    print(f"{'=' * 60}")
    print(f"  RHOAI wheels:          {len(buckets['rhoai_wheel']):>4} packages")
    print(f"  PyPI sdist:            {len(buckets['pypi_sdist']):>4} packages")
    print(f"  PyPI wheel (last resort): {len(buckets['pypi_wheel']):>4} packages")
    print(f"  Total:                 {total:>4} packages")
    print()
    print(f"  Hashed wheel (RHOAI):  .konflux/requirements.hashes.wheel{suffix}.txt")
    print(f"  Hashed source (PyPI):  .konflux/requirements.hashes.source{suffix}.txt")
    pypi_wheel_file = f".konflux/requirements.hashes.wheel.pypi{suffix}.txt"
    print(f"  Hashed wheel (PyPI):   {pypi_wheel_file}")
    print(f"  Build deps:            {build_output}")
    print()
    print("Remember to commit output files and push the changes.")


if __name__ == "__main__":
    main()
