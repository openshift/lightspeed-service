#!/usr/bin/python

"""Display all Enterprise Contract failures found in EC job logs."""

import json
import sys
from pathlib import Path

PREFIX = "[report-json]"

json_raw_data = ""

if len(sys.argv) < 2:
    print("Usage: print_ec_failures.py log_file_name.log")
    sys.exit(1)

log_file_name = sys.argv[1]

# read the log file that contains several sections
# and where each line has some prefix
with Path(log_file_name).open("r") as log_file:
    for line in log_file:
        # retrieve just those lines with JSON data
        # and append them
        if line.startswith(PREFIX):
            stripped = line[len(PREFIX) :].strip()
            json_raw_data += stripped

# deserialize JSON
json_data = json.loads(json_raw_data)

# serialize into pretty-printed format
with Path("logs.json").open("w") as json_file:
    json.dump(json_data, json_file, indent=4)

# print summary
for component in json_data["components"]:
    print(f"Component: {component['name']}")
    violations = component["violations"]
    print(f"    violations: {len(violations)}")
    for i, violation in enumerate(violations):
        print(f"        #{i+1}: {violation['msg']}")
