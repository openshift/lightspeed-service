#!/usr/bin/python

"""Filter the SHAs in the requirements.txt file.

This script processes the requirements.txt file and filters
out hashes that are not compatible with the current environment.
It retains only the matching hashes based on the platform and Python version.
"""

import re

import requests
from packaging import tags


def filter_requirements(input_file, output_file):
    """Get all supported tags for the current environment."""
    supported_tags = {str(tag) for tag in tags.sys_tags()}

    # Print supported tags for the current environment
    print("Supported tags for current environment:")
    for tag in supported_tags:
        print(tag)

    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        package_line = None
        hashes = []
        for line in infile:
            # Check if we are at a package definition line
            if "==" in line:
                # If we have a previous package, filter its hashes and write it to the file
                if package_line and hashes:
                    filtered_hash = filter_hashes(hashes, package_line, supported_tags)
                    if filtered_hash:
                        outfile.write(f"{package_line} \\\n")
                        outfile.write(f"    --hash=sha256:{filtered_hash}\n")
                    else:
                        # If no valid hash was found for this package, issue a warning
                        print(
                            f"WARNING: No valid hash found for {package_line}, skipping."
                        )

                # Start a new package (clean the trailing continuation character and quotes)
                package_line = line.strip().rstrip("\\").strip()
                hashes = []
            # Collect hashes for the current package
            elif "--hash=sha256:" in line:
                hashes.append(line.strip())

        # Handle the last package in the file
        if package_line and hashes:
            filtered_hash = filter_hashes(hashes, package_line, supported_tags)
            if filtered_hash:
                outfile.write(f"{package_line} \\\n")
                outfile.write(f"    --hash=sha256:{filtered_hash}\n")
            else:
                # Warning for the last package if no valid hash is found
                print(f"WARNING: No valid hash found for {package_line}, skipping.")


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
    if response.status_code == 200:
        package_data = response.json()
        for file_info in package_data.get("urls", []):
            if file_info["digests"]["sha256"] == hash_value:
                # Return the filename to extract the tags
                return {"filename": file_info["filename"]}
    return None


# Example usage
filter_requirements("requirements.txt", "filtered-requirements.txt")
