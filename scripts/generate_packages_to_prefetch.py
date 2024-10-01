#!/usr/bin/python

"""Generate list of packages to be prefetched in Cachi2 and used in Konflux for hermetic build.

usage: generate_packages_to_prefetch.py [-h] [-p]

options:
  -h, --help            show this help message and exit
  -p, --process-special-packages
                        Enable or disable processing special packages like torch etc.
  -c, --cleanup         Enable or disable work directory cleanup
  -w WORK_DIRECTORY, --work-directory WORK_DIRECTORY
                        Work directory to store files generated during different stages
                        of processing



This script performs several steps:
    1. removes torch+cpu dependency from project file
    2. generates requirements.txt file from pyproject.toml + pdm.lock
    3. removes all torch dependencies (including CUDA/Nvidia packages)
    4. downloads torch+cpu wheel
    5. computes hashes for this wheel
    6. adds the URL to wheel + hash to resulting requirements.txt file
    7. downloads script pip_find_builddeps from the Cachito project
    8. generated requirements-build.in file
    9. compiles requirements-build.in file into requirements-build.txt file

Please note that this script depends on tool that is downloaded from repository containing
Cachito system. This tool is run locally w/o any additional security checks etc. so some
care is needed (run this script from within containerized environment etc.).
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from os.path import join
from urllib.request import urlretrieve

import requests
from packaging import tags

# just these files are needed as project stub, no other configs and/or sources are needed
PROJECT_FILES = ("pyproject.toml", "pdm.lock", "LICENSE", "README.md")

# registry with Torch wheels (CPU variant)
TORCH_REGISTRY = "https://download.pytorch.org/whl/cpu"
TORCH_VERSION = "2.2.2"
TORCH_WHEEL = f"torch-{TORCH_VERSION}%2Bcpu-cp311-cp311-linux_x86_64.whl"

# URL to static content of repository containing Cachito system
CACHITO_URL = "https://raw.githubusercontent.com/containerbuildsystem/cachito"

# path to helper script to generate list of packages that will need to be build
# This script is part of Cachito system (predecedor of Cachi2) and have to be user
BUILDDEPS_SCRIPT_PATH = "master/bin"  # wokeignore:rule=master

# name of path to helper script to generate list of packages that will need to be build
BUILDDEPS_SCRIPT_NAME = "pip_find_builddeps.py"

# name of standard pip requirement file
REQUIREMENTS_FILE = "requirements.txt"
FILTERED_REQUIREMENTS_FILE = "requirements_filtered.txt"


def shell(command, directory):
    """Run command via shell inside specified directory."""
    return subprocess.check_output(command, cwd=directory, shell=True)  # noqa: S602


def copy_project_stub(directory):
    """Copy all files that represent project stub into specified directory."""
    for project_file in PROJECT_FILES:
        shutil.copy(project_file, directory)


def remove_torch_dependency(directory):
    """Remove torch (specifically torch+cpu) dependency from the project.toml file."""
    shell("pdm remove torch", directory)


def generate_requirements_file(work_directory):
    """Generate file requirements.txt that contains hashes for all packages."""
    shell("pip-compile -vv pyproject.toml --generate-hashes", work_directory)


def remove_package(directory, source, target, package_prefix):
    """Remove package or packages with specified prefix from the requirements file."""
    package_block = False

    with open(join(directory, source)) as fin:
        with open(join(directory, target), "w") as fout:
            for line in fin:
                if line.startswith(package_prefix):
                    print(line)
                    package_block = True
                elif package_block:
                    # the whole block with hashes needs to be filtered out
                    if not line.startswith("    "):
                        # end of package block detected
                        package_block = False
                if not package_block:
                    fout.write(line)


def remove_unwanted_dependencies(directory):
    """Remove all unwanted dependencies from requirements file, creating in-between files."""
    # the torch itself
    remove_package(directory, REQUIREMENTS_FILE, "step1.txt", "torch")

    # all CUDA-related packages (torch depends on them)
    remove_package(directory, "step1.txt", "step2.txt", "nvidia")


def wheel_url(registry, wheel):
    """Construct full URL to wheel."""
    return f"{registry}/{wheel}"


def download_wheel(directory, registry, wheel):
    """Download selected wheel from registry."""
    url = wheel_url(registry, wheel)
    into = join(directory, wheel)
    urlretrieve(url, into)  # noqa: S310


def generate_hash(directory, registry, wheel, target):
    """Generate hash entry for given wheel."""
    output = shell(f"pip hash {wheel}", directory)
    hash_line = output.decode("ascii").splitlines()[1]
    with open(join(directory, target), "w") as fout:
        url = wheel_url(registry, wheel)
        fout.write(f"torch @ {url} \\\n")
        fout.write(f"    {hash_line}\n")


def retrieve_supported_tags():
    """Retrieve all supported tags for the current environment."""
    supported_tags = {str(tag) for tag in tags.sys_tags()}

    # Print supported tags for the current environment
    print("Supported tags for current environment:")
    for tag in supported_tags:
        print(tag)

    return supported_tags


def filter_hashes(hashes, package_line, supported_tags):
    """Filter hashes based on platform and Python version compatibility."""
    package_name, package_version = package_line.split("==")
    package_version = (
        package_version.strip()
    )  # Ensure there's no extra space or backslash
    for hash_line in hashes:
        # Extract the hash from the hash line
        hash_value = re.search(r"sha256:(\w+)", hash_line).group(1)

        # Fetch wheel metadata from PyPI
        file_info = get_package_file_info(package_name, package_version, hash_value)
        if file_info:
            # Check if the file is a wheel or source distribution
            if file_info["filename"].endswith(".whl"):
                # Extract the tags directly from the filename
                # (e.g., "cp310-cp310-manylinux_2_17_x86_64")
                wheel_tags = extract_tags_from_filename(file_info["filename"])

                # Print tags found in file_info
                print(f"\nTags for {file_info['filename']} from PyPI:")
                for tag in wheel_tags:
                    print(tag)

                # Find the best matching tag based on priority
                common_tags = wheel_tags & supported_tags
                if common_tags:
                    best_tag = select_best_tag(common_tags)
                    print(f"Best tag selected for {file_info['filename']}: {best_tag}")
                    return hash_value
            else:
                # For source distributions (.tar.gz or .zip),
                # assume compatibility with all platforms
                print(f"\nSource distribution {file_info['filename']} is compatible.")
                return hash_value  # Skip tag matching for source distributions

    return None


def extract_tags_from_filename(filename):
    """Extract tags from a wheel filename, handling multiple platform tags.

    Example filename:
    aiohttp-3.10.2-cp311-cp311-manylinux_2_5_i686.manylinux1_i686.manylinux_2_17_i686.manylinux2014_i686.whl
    """
    # Remove the .whl extension
    filename = filename.replace(".whl", "")

    # Split the filename into parts, ignoring the first two (package name and version)
    parts = filename.split("-")

    if len(parts) < 3:
        print(f"Unexpected filename format: {filename}")
        return set()

    # Extract Python version and ABI tags (e.g., "cp38-cp38")
    python_tag = parts[2]
    abi_tag = parts[3] if len(parts) > 3 else "none"

    # Extract platform tags, which may have multiple parts separated by dots
    platform_tags = parts[4:]  # Everything after Python and ABI tags
    platform_tag_list = ".".join(platform_tags).split(
        "."
    )  # Split on dots to handle multiple platform tags

    # Create a set of tags by combining the Python tag, ABI tag, and each platform tag
    tags_set = {
        f"{python_tag}-{abi_tag}-{platform_tag}" for platform_tag in platform_tag_list
    }
    return tags_set


def select_best_tag(common_tags):
    """Generalized function to select the 'best' tag based on versioning and architecture."""

    def tag_priority(tag):
        """Assign a priority to the tag based on manylinux version."""

        def extract_manylinux_version(tag):
            """Extract the manylinux version from the tag."""
            match = re.search(r"manylinux_(\d+)_(\d+)", tag)
            if match:
                return int(match.group(1)), int(match.group(2))
            # Normalize manylinux2014 to manylinux_2_17 and manylinux1 to manylinux_2_5
            elif "manylinux2014" in tag:
                return 2, 17
            elif "manylinux1" in tag:
                return 2, 5
            return 0, 0  # Default to lowest version if no match

        # Extract version from the tag
        manylinux_version = extract_manylinux_version(tag)
        return manylinux_version

    # Sort the common tags based on the priority function
    return max(common_tags, key=tag_priority)


def get_package_file_info(package_name, package_version, hash_value):
    """Fetch package file information from PyPI for a specific hash."""
    url = f"https://pypi.org/pypi/{package_name}/{package_version}/json"
    response = requests.get(url, timeout=60)
    if response.status_code == requests.codes.ok:
        package_data = response.json()
        for file_info in package_data.get("urls", []):
            if file_info["digests"]["sha256"] == hash_value:
                # Return the filename to extract the tags
                return {"filename": file_info["filename"]}
    return None


def append_package(outfile, hashes, package_line, supported_tags):
    """Append package with filtered hashes into the output file."""
    if package_line and hashes:
        filtered_hash = filter_hashes(hashes, package_line, supported_tags)
        if filtered_hash:
            outfile.write(f"{package_line} \\\n")
            outfile.write(f"    --hash=sha256:{filtered_hash}\n")
        else:
            # If no valid hash was found for this package, issue a warning
            print(f"WARNING: No valid hash found for {package_line}, skipping.")


def filter_packages_for_platform(input_file: str, output_file: str):
    """Filter packages for given platform."""
    supported_tags = retrieve_supported_tags()
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        package_line = None
        hashes = []
        for line in infile:
            # Check if we are at a package definition line
            if "==" in line:
                # If we have a previous package, filter its hashes and write it to the file
                append_package(outfile, hashes, package_line, supported_tags)

                # Start a new package (clean the trailing continuation character and quotes)
                package_line = line.strip().rstrip("\\").strip()
                hashes = []
            # Collect hashes for the current package
            elif "--hash=sha256:" in line:
                hashes.append(line.strip())

        # Handle the last package in the file
        append_package(outfile, hashes, package_line, supported_tags)


def generate_list_of_packages(
    work_directory: str, process_special_packages: bool, filter_packages: bool
):
    """Generate list of packages, take care of unwanted packages and wheel with Torch package."""
    copy_project_stub(work_directory)
    if process_special_packages:
        remove_torch_dependency(work_directory)
    generate_requirements_file(work_directory)

    if process_special_packages:
        remove_unwanted_dependencies(work_directory)
        download_wheel(work_directory, TORCH_REGISTRY, TORCH_WHEEL)
        shutil.copy(join(work_directory, "step2.txt"), REQUIREMENTS_FILE)
        generate_hash(work_directory, TORCH_REGISTRY, TORCH_WHEEL, "hash.txt")
        shell("cat step2.txt hash.txt > " + REQUIREMENTS_FILE, work_directory)

    if filter_packages:
        filter_packages_for_platform(
            join(work_directory, REQUIREMENTS_FILE),
            join(work_directory, FILTERED_REQUIREMENTS_FILE),
        )
        # copy the newly generated requirements file back to the project
        shutil.copy(join(work_directory, FILTERED_REQUIREMENTS_FILE), REQUIREMENTS_FILE)
    else:
        # copy the newly generated requirements file back to the project
        shutil.copy(join(work_directory, REQUIREMENTS_FILE), REQUIREMENTS_FILE)


def generate_packages_to_be_build(work_directory):
    """Generate list of packages that will need to be build."""
    # download helper script to generate list of packages
    url = f"{CACHITO_URL}/{BUILDDEPS_SCRIPT_PATH}/{BUILDDEPS_SCRIPT_NAME}"
    into = join(work_directory, BUILDDEPS_SCRIPT_NAME)
    urlretrieve(url, into)  # noqa: S310

    infile = "requirements-build.in"
    outfile = "requirements-build.txt"

    # generate file requirements-build.in
    command = (
        "python pip_find_builddeps.py requirements.txt --append "
        + f"--only-write-on-update --ignore-errors --allow-binary -o {infile}"
    )
    shell(command, work_directory)

    # generate requirements-build.txt file
    command = f"pip-compile {infile} --allow-unsafe --generate-hashes -o {outfile}"
    shell(command, work_directory)

    # copy everything back to project
    shutil.copy(join(work_directory, infile), infile)
    shutil.copy(join(work_directory, outfile), outfile)


def args_parser(args: list[str]) -> argparse.Namespace:
    """Command line arguments parser."""
    parser = argparse.ArgumentParser()

    # flag that enables processing special packages ('pytorch' and 'nvidia-' at this moment)
    parser.add_argument(
        "-p",
        "--process-special-packages",
        default=False,
        action="store_true",
        help="Enable or disable processing special packages like torch etc.",
    )

    # work directory
    parser.add_argument(
        "-w",
        "--work-directory",
        default=tempfile.mkdtemp(),
        type=str,
        help="Work directory to store files generated during different stages of processing",
    )

    # flag that enables cleaning up work directory
    parser.add_argument(
        "-c",
        "--cleanup",
        default=False,
        action="store_true",
        help="Enable or disable work directory cleanup",
    )

    # flag that enables filtering package SHAs that are not compatible with specified platform
    parser.add_argument(
        "-f",
        "--filter-packages",
        default=False,
        action="store_true",
        help="Enable or disable filtering packages not compatible with specified platform",
    )

    # execute parser
    return parser.parse_args()


def main() -> None:
    """Generate packages to prefetch."""
    args = args_parser(sys.argv[1:])

    # sanitize work directory
    work_directory = os.path.normpath("/" + args.work_directory)
    if work_directory.startswith("/"):
        work_directory = work_directory[1:]
    if work_directory == "":
        work_directory = "."

    print(f"Work directory {work_directory}")
    generate_list_of_packages(
        work_directory, args.process_special_packages, args.filter_packages
    )
    generate_packages_to_be_build(work_directory)

    # optional cleanup step
    # (for debugging purposes it might be better to see 'steps' files to check
    # if everything's ok)
    if args.cleanup:
        shutil.rmtree(work_directory)


if __name__ == "__main__":
    main()
