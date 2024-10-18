#!/usr/bin/python

"""Utility script for syncing up upstream repository with downstream one."""

import argparse
import logging
import os
import subprocess
from datetime import datetime

"""Usage:
    usage: sync_repos.py [-h] -s SOURCE_REPOSITORY -t TARGET_REPOSITORY [-e SINCE]

    options:
      -h, --help            show this help message and exit
      -s SOURCE_REPOSITORY, --source-repository SOURCE_REPOSITORY
                            Path to the source repository
      -t TARGET_REPOSITORY, --target-repository TARGET_REPOSITORY
                            Path to the target repository
      -e SINCE, --since SINCE
                            Synchronize commits since given date
"""

logger = logging.getLogger("Repository sync")


def command_output_as_lines(stdout: bytes) -> list[str]:
    """Transform output of external command into lines."""
    text = stdout.decode("utf-8")
    return text.strip().split("\n")


def read_shas(source_repository: str, since: str) -> list[str]:
    """Read commit SHAs from Git output."""
    logger.info("read SHAs of last commits")
    result = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "git",
            "-C",
            source_repository,
            "log",
            "--oneline",
            "--no-merges",
            f"--since={since}",
        ],
        capture_output=True,
    )

    # check if the command finished with ok status
    if result.returncode != 0:
        message = "retrieving changes failed"
        logger.error(message)
        raise Exception(message)

    # transform stdout
    lines = command_output_as_lines(result.stdout)
    return [line.split(" ")[0] for line in lines]


def export_patches(
    source_repository: str, workdir: str, patches: list[str]
) -> list[str]:
    """Export all found patches that are not merge commits."""
    logger.info("exporting %d patches", patches)
    result = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "git",
            "-C",
            source_repository,
            "format-patch",
            f"-{patches}",
            "-o",
            workdir,
        ],
        capture_output=True,
    )

    # check if the command finished with ok status
    if result.returncode != 0:
        message = "export patches failed"
        logger.error(message)
        raise Exception(message)

    # transform stdout
    return command_output_as_lines(result.stdout)


def check_changes(target_repository: str, patch_file: str) -> None:
    """Check the changes in the patch."""
    result = subprocess.run(  # noqa: S603
        ["git", "-C", target_repository, "apply", "--stat", patch_file],  # noqa: S607
        capture_output=True,
    )

    # check if the command finished with ok status
    if result.returncode != 0:
        error = result.stdout.decode("utf-8")
        message = f"cannot apply patch {patch_file} because of {error}"
        logger.error(message)
        raise Exception(message)

    logger.info("checking the changes: ok")


def check_if_applicable(target_repository: str, patch_file: str) -> None:
    """Check that the patch can be applied."""
    result = subprocess.run(  # noqa: S603
        ["git", "-C", target_repository, "apply", "--check", patch_file],  # noqa: S607
        capture_output=True,
    )

    # check if the command finished with ok status
    if result.returncode != 0:
        error = result.stderr.decode("utf-8")
        message = f"cannot apply patch {patch_file} (dry run) because of {error}"
        logger.error(message)
        raise Exception(message)

    logger.info("checking that the patch can be applied: ok")


def apply_patch(target_repository: str, patch_file: str) -> None:
    """Apply the patch to target repository."""
    result = subprocess.run(  # noqa: S603
        ["git", "-C", target_repository, "am", "--signoff", patch_file],  # noqa: S607
        capture_output=True,
    )

    # check if the command finished with ok status
    if result.returncode != 0:
        error = result.stderr.decode("utf-8")
        message = f"cannot apply patch {patch_file} because of {error}"
        logger.error(message)
        raise Exception(message)

    logger.info("apply patch: ok")


def apply_patches(target_repository: str, patch_files: list[str]) -> None:
    """Apply patches to destination repository."""
    logger.info("applying patches to %s", target_repository)

    # try to apply all patches, one by one
    for patch_file in patch_files:
        logger.info("patch file %s", patch_file)
        check_changes(target_repository, patch_file)
        check_if_applicable(target_repository, patch_file)
        apply_patch(target_repository, patch_file)


def sync_repositories(
    source_repository: str, target_repository: str, since: str
) -> None:
    """Synchronize source repository with target one."""
    workdir = os.getcwd()

    logger.info(
        "syncing %s with %s since %s", source_repository, target_repository, since
    )
    logger.info("working directory for patches %s", workdir)

    shas = read_shas(source_repository, since)
    patches = len(shas)
    if patches == 0:
        message = "nothing to sync"
        logger.error(message)
        raise Exception(message)

    logger.info("SHAs for commits to be synced: %s", ",".join(shas))

    patch_files = export_patches(source_repository, workdir, patches)
    logger.info("patches generated: %s", patch_files)

    apply_patches(target_repository, patch_files)


def default_since_value() -> str:
    """Return the today's time in a format expected by Git."""
    today = datetime.today().strftime("%Y-%m-%d")
    return today + " 00:00"


def main() -> None:
    """Entry point to the script to synchronize repositories."""
    parser = argparse.ArgumentParser(
        description="Utility script for syncing up upstream repository with downstream one"
    )
    parser.add_argument(
        "-s",
        "--source-repository",
        required=True,
        help="Path to the source repository",
        default=None,
    )
    parser.add_argument(
        "-t",
        "--target-repository",
        required=True,
        help="Path to the target repository",
        default=None,
    )
    parser.add_argument(
        "-e",
        "--since",
        required=False,
        help="Synchronize commits since given date",
        default=default_since_value(),
    )

    args = parser.parse_args()
    sync_repositories(args.source_repository, args.target_repository, args.since)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
