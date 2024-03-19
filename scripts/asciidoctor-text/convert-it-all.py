#!/usr/bin/python

"""Utility script to convert OCP docs from adoc to plain text."""

import argparse
import os
import subprocess
import sys

import yaml


def process_node(node, dir="", file_list=[]):
    """Process YAML node from the topic map."""
    currentdir = dir
    if "Topics" in node:
        currentdir = os.path.join(currentdir, node["Dir"])
        for subnode in node["Topics"]:
            file_list = process_node(subnode, dir=currentdir, file_list=file_list)
    else:
        file_list.append(os.path.join(currentdir, node["File"]))
    return file_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This command converts the openshift-docs assemblies to plain text.",
        usage="convert-it-all [options]",
    )

    parser.add_argument(
        "--input-dir",
        "-i",
        required=True,
        help="The input directory for the openshift-docs repo",
    )
    parser.add_argument("--topic-map", "-t", required=True, help="The topic map file")
    parser.add_argument(
        "--output-dir", "-o", required=True, help="The output directory for text"
    )
    parser.add_argument(
        "--attributes", "-a", help="An optional file containing attributes"
    )

    args = parser.parse_args(sys.argv[1:])

    attribute_list = []
    if args.attributes is not None:
        with open(args.attributes, "r") as fin:
            attributes = yaml.safe_load(fin)
        for key, value in attributes.items():
            attribute_list = [*attribute_list, "-a", key + '="%s"' % value]

    with open(args.topic_map, "r") as fin:
        topic_map = yaml.safe_load_all(fin)
        mega_file_list = []
        for map in topic_map:
            file_list = []
            file_list = process_node(map, file_list=file_list)
            mega_file_list = mega_file_list + file_list

    output_dir = os.path.normpath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    input_dir = os.path.normpath(args.input_dir)
    script_dir = os.path.dirname(os.path.realpath(__file__))

    for filename in mega_file_list:
        output_file = os.path.join(output_dir, filename + ".txt")
        os.makedirs(os.path.dirname(os.path.realpath(output_file)), exist_ok=True)
        input_file = os.path.join(input_dir, filename + ".adoc")
        converter_file = os.path.join(script_dir, "text-converter.rb")
        print("Processing: " + input_file)
        command = ["asciidoctor"]
        command = command + attribute_list
        command = [
            *command,
            "-r",
            converter_file,
            "-b",
            "text",
            "-o",
            output_file,
            "--trace",
            "--quiet",
            input_file,
        ]
        result = subprocess.run(command, check=False)  # noqa: S603
        if result.returncode != 0:
            print(result)
            print(result.stdout)
