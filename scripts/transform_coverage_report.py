"""Utility script to convert a Python to GO coverage report."""

import json
import sys


def write_go_coverage_format(
    file_path: str, file_data: dict, output_file_name: str
) -> None:
    """Write coverage information in a format similar to GO test report to a file.

    Args:
        file_path (str): Path of the file.
        file_data (dict): Coverage information for the file.
        output_file_name (str): Path of the output file.
    """
    executed_lines = file_data.get("executed_lines", [])
    missing_lines = file_data.get("missing_lines", [])
    with open(output_file_name, "a", encoding="utf-8") as f:
        f.writelines(
            f"{file_path}:{line}.0,{line + 1}.0 1 1\n" for line in executed_lines
        )

        f.writelines(
            f"{file_path}:{line}.0,{line + 1}.0 1 0\n" for line in missing_lines
        )


def parse_coverage_json(json_content: str, output_file_name: str) -> None:
    """Parse the content of a Python coverage report in JSON format and write to a file.

    Args:
        json_content (str): The JSON content of the coverage report.
        output_file_name (str): Path of the output file.

    Returns:
        None: Writes information to the output file.

    Raises:
        json.JSONDecodeError: If there is an error decoding the JSON content.
    """
    try:
        coverage_data = json.loads(json_content)
        files_info = coverage_data.get("files", {})
        for file_path, file_data in files_info.items():
            write_go_coverage_format(
                f"github.com/openshift/lightspeed-service/{file_path}",
                file_data,
                output_file_name,
            )

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <json_file> <output_file>")
        sys.exit(1)

    json_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    try:
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write("mode: set\n")

        with open(json_file_path, "r", encoding="utf-8") as json_file:
            json_file_content = json_file.read()
            parse_coverage_json(json_file_content, output_file_path)

    except FileNotFoundError:
        print(f"File not found: {json_file_path}")
    except Exception as e:
        print(f"Error processing files: {e}")
