#!/usr/bin/env python3

"""Script to download feedbacks from Ceph bucket and create CSV file with consolidated report.

usage: user_data_aggregator.py [-h] [-e ENDPOINT] [-b BUCKET]
                               [--access-key ACCESS_KEY] [--secret-access-key SECRET_ACCESS_KEY]
                               [-r REGION] [-p] [-k KEEP]
                               [-s] [-d] [-l] [-o OUTPUT] [-w WORK_DIRECTORY] [-t] [-v]


Typical use cases:
~~~~~~~~~~~~~~~~~~
    Test if Ceph bucket is accessible:
        ./user_data_aggregator.py -e URL --access-key=KEY --secret-access-key=KEY --bucket=BUCKET -p
    List all objects stored in Ceph bucket:
        ./user_data_aggregator.py -e URL --access-key=KEY --secret-access-key=KEY --bucket=BUCKET -l
    Download tarballs, aggregate feedback, and cleanup tarballs:
        ./user_data_aggregator.py -e URL --access-key=KEY --secret-access-key=KEY --bucket=BUCKET
    Download tarballs, aggregate feedback, without cleanup:
        ./user_data_aggregator.py -e URL --access-key=KEY --secret-access-key=KEY --bucket=BUCKET -k
    Download tarballs only:
        ./user_data_aggregator.py -e URL --access-key=KEY --secret-access-key=KEY --bucket=BUCKET -d


All CLI options:
~~~~~~~~~~~~~~~~

options:
  -h, --help            show this help message and exit
  -e ENDPOINT, --endpoint ENDPOINT
                        Ceph storage endpoint URL
  -b BUCKET, --bucket BUCKET
                        Bucket name
  --access-key ACCESS_KEY
                        Ceph access key ID
  --secret-access-key SECRET_ACCESS_KEY
                        Ceph secret access key ID
  -r REGION, --region REGION
                        Ceph region
  -p, --ping            Perform check if Ceph bucket is accessible
  -i IGNORE, --ignore IGNORE
                        Cluster IDs to ignore during processing
  -k, --keep            Keep downloaded files on disk
  -s, --skip-downloading
                        Skip downloading, generate CSV from existing files
  -d, --download-only   Just download tarballs
  -l, --list-objects    Just list objects in bucket
  -o OUTPUT, --output OUTPUT
                        Output file name
  -co CONVERSATION_HISTORY_OUTPUT, --conversation-history-output CONVERSATION_HISTORY_OUTPUT
                        Output file name for conversation history
  -w WORK_DIRECTORY, --work-directory WORK_DIRECTORY
                        Directory to store downloaded tarballs
  -c, --conversation-history
                        Include the whole conversation history in the generated output
  -rd, --referenced-documents
                        Include referenced documents in the generated output
  -t, --statistic       Print aggregation statistic at the end
  -v, --verbose         Verbose operations
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import tarfile
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import boto3
import pytest

# magic file that should be stored in a tarball
TOPLEVEL_MAGIC_FILE = "openshift_lightspeed.json"

# directory within tarball with user feedback files
FEEDBACK_DIRECTORY = "feedback/"

# directory within tarball with conversation history
HISTORY_DIRECTORY = "transcripts/"

logger = logging.getLogger("Feedback aggregator")


@dataclass
class Statistic:
    """Statistic printed at the end of processing."""

    downloaded_tarballs: int = 0
    tarballs_not_downloaded: int = 0
    tarballs_with_incorrect_key: int = 0
    tarballs_without_feedback: int = 0
    feedback_read_error: int = 0
    feedbacks_read: int = 0
    feedbacks_aggregated: int = 0
    conversation_history_included: int = 0


statistic = Statistic()


def args_parser(args: list[str]) -> argparse.Namespace:
    """Command line arguments parser."""
    parser = argparse.ArgumentParser(description="Feedback aggregator")
    parser.add_argument(
        "-e",
        "--endpoint",
        type=str,
        default="",
        help="Ceph storage endpoint URL",
    )
    parser.add_argument(
        "-b",
        "--bucket",
        type=str,
        default="",
        help="Bucket name",
    )
    parser.add_argument(
        "--access-key",
        type=str,
        default="",
        help="Ceph access key ID",
    )
    parser.add_argument(
        "--secret-access-key",
        type=str,
        default="",
        help="Ceph secret access key ID",
    )
    parser.add_argument(
        "-r",
        "--region",
        type=str,
        default="us-east-1",
        help="Ceph region",
    )
    parser.add_argument(
        "-p",
        "--ping",
        default=False,
        action="store_true",
        help="Perform check if Ceph bucket is accessible",
    )
    parser.add_argument(
        "-i",
        "--ignore",
        # cluster UUID used by CCX monitoring infrastructure
        default="00000000-1111-0000-1111-000000000001",
        help="Cluster IDs to ignore during processing splited by comma: "
        + "00000000-0000-0000-0000-000000000001, 00000000-0000-0000-0000-000000000002",
    )
    parser.add_argument(
        "-k",
        "--keep",
        default=False,
        action="store_true",
        help="Keep downloaded files on disk",
    )
    parser.add_argument(
        "-s",
        "--skip-downloading",
        default=False,
        action="store_true",
        help="Skip downloading, generate CSV from existing files",
    )
    parser.add_argument(
        "-d",
        "--download-only",
        default=False,
        action="store_true",
        help="Just download tarballs",
    )
    parser.add_argument(
        "-l",
        "--list-objects",
        default=False,
        action="store_true",
        help="Just list objects in bucket",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="feedbacks.csv",
        help="Output file name for user feedback",
    )
    parser.add_argument(
        "-co",
        "--conversation-history-output",
        type=str,
        default="conversation_history.csv",
        help="Output file name for conversation history",
    )
    parser.add_argument(
        "-w",
        "--work-directory",
        type=str,
        default=".",
        help="Directory to store downloaded tarballs",
    )
    parser.add_argument(
        "-c",
        "--conversation-history",
        default=False,
        action="store_true",
        help="Include the whole conversation history in the generated output",
    )
    parser.add_argument(
        "-rd",
        "--referenced-documents",
        default=False,
        action="store_true",
        help="Include referenced documents in the generated output",
    )
    parser.add_argument(
        "-t",
        "--statistic",
        default=False,
        action="store_true",
        help="Print aggregation statistic at the end",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="Verbose operations",
    )
    return parser.parse_args(args)


def connect(args: argparse.Namespace):
    """Connect to Ceph."""
    client = boto3.client(
        "s3",
        endpoint_url=args.endpoint,
        aws_access_key_id=args.access_key,
        aws_secret_access_key=args.secret_access_key,
        region_name=args.region,
    )
    return client


def ping_ceph(args: argparse.Namespace) -> None:
    """Ping Ceph and check if bucket is accessible."""
    client = connect(args)
    try:
        client.head_bucket(Bucket=args.bucket)
        logger.info("Bucket is accessible")
    except Exception as e:
        logger.error(f"Bucket is not accessible: {e}")


def list_objects(args: argparse.Namespace) -> None:
    """List all objects in specified bucket."""
    client = connect(args)
    bucket_name = args.bucket
    try:
        response = client.list_objects_v2(Bucket=bucket_name)
        logger.debug(response)
        if "Contents" in response:
            logger.info(f"Contents of bucket '{bucket_name}':")
            for obj in response["Contents"]:
                logger.info(obj["Key"])
        else:
            logger.warning(f"Bucket '{bucket_name}' is empty.")
    except Exception as e:
        logger.error(f"Failed to list contents of bucket '{bucket_name}':", e)


def key_match(key: str) -> Optional[re.Match]:
    """Match the string with regexp with expected key format."""
    # build the regular expression for checking the key format
    key_prefix = r"archives\/compressed"
    pre_selector = "[0-9A-Fa-f]{2}"
    cluster_uuid = (
        "[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
    )
    timestamp = r"(\d{6})\/(\d{2})\/(\d{6})"
    extension = ".tar.gz"
    pattern = rf"{key_prefix}\/{pre_selector}\/({cluster_uuid})\/{timestamp}{extension}"
    return re.match(pattern, key)


def check_key(key: str) -> bool:
    """Check if the object key has correct format."""
    return key_match(key) is not None


def construct_filename(key: str) -> str:
    """Construct filename from key."""
    groups = key_match(key)
    if groups is None:
        raise Exception(f"Can not construct filename from key {key}")

    cluster_id = groups[1].lower()
    year = groups[2][:4]
    month = groups[2][4:]
    day = groups[3]
    hour = groups[4][:2]
    minute = groups[4][2:4]
    second = groups[4][4:]
    return f"{cluster_id}_{year}_{month}_{day}_{hour}_{minute}_{second}.tar.gz"


def get_cluster_name(key: str) -> str:
    """Retrieve cluster name from object key."""
    groups = key_match(key)
    if groups is None:
        raise Exception(f"Can not construct filename from key {key}")

    return groups[1].lower()


def download_tarball(client, bucket_name: str, obj) -> None:
    """Download one tarball from Ceph."""
    try:
        key = obj["Key"]
        if check_key(key):
            filename = construct_filename(key)
            logger.info(f"Downloading {key} into {filename}")
            data = client.get_object(Bucket=bucket_name, Key=key)
            with open(filename, "wb") as fout:
                fout.write(data["Body"].read())
            statistic.downloaded_tarballs += 1
        else:
            logger.warning(f"Incorrect object key {key}, skipping")
            statistic.tarballs_with_incorrect_key += 1
    except Exception as e:
        logger.error(f"Unable to download object: {e}")
        statistic.tarballs_not_downloaded += 1


def download_tarballs(args: argparse.Namespace) -> None:
    """Download all tarballs from Ceph."""
    # retrieve list of ignored clusters
    ignored_clusters = set(map(str.strip, args.ignore.split(",")))
    client = connect(args)
    bucket_name = args.bucket
    try:
        response = client.list_objects_v2(Bucket=bucket_name)
        logger.debug(response)
        if "Contents" in response:
            for obj in response["Contents"]:
                cluster_name = get_cluster_name(obj["Key"])
                if cluster_name not in ignored_clusters:
                    download_tarball(client, bucket_name, obj)
        else:
            logger.warning(f"Bucket '{bucket_name}' is empty.")
    except Exception as e:
        logger.error(f"Failed to list contents of bucket '{bucket_name}':", e)


def read_feedbacks_from_tarball(tarball: tarfile.TarFile) -> list[dict[str, Any]]:
    """Read feedbacks from tarball, skip over errors."""
    feedbacks = []
    for filename in tarball.getnames():
        if filename.startswith(FEEDBACK_DIRECTORY):
            try:
                f = tarball.extractfile(filename)
                if f is not None:
                    data = f.read().decode("UTF-8")
                    feedbacks.append(json.loads(data))
                    statistic.feedbacks_read += 1
                else:
                    logger.error(f"Nothing to extract from {filename}")
                    statistic.tarballs_without_feedback += 1
            except Exception as e:
                logger.error(f"Unable to read feedback: {e}")
                statistic.feedback_read_error += 1
    return feedbacks


def feedbacks_from_tarball(tarball_name: str) -> list[dict[str, Any]]:
    """Retrieve all feedbacks from tarball."""
    tarball = tarfile.open(tarball_name, "r:gz")
    filelist = tarball.getnames()

    # check if the tarball seems to be correct one
    if TOPLEVEL_MAGIC_FILE not in filelist:
        logger.warning(f"Incorrect tarball: missing {TOPLEVEL_MAGIC_FILE}")
        return []

    feedbacks = read_feedbacks_from_tarball(tarball)

    if len(feedbacks) == 0:
        logger.warning(f"Tarball {tarball_name} does not contain any feedback.")

    return feedbacks


def format_referenced_documents(docs: list[dict[str, str]]) -> str:
    """Format referenced documents section in full conversation history for feedback."""
    output = ""
    for doc in docs:
        title = doc["title"]
        url = doc["docs_url"]
        output += f"{title}: {url}\n"
    return output or "[None]"


def read_full_conversation_history(
    tarball_name: str, user_id: str, history_id: str, referenced_documents: bool
) -> str:
    """Read full conversation history from tarball."""
    logger.info(f"Reading full conversation history from {tarball_name}")
    tarball = tarfile.open(tarball_name, "r:gz")
    separator = 100 * "-"

    history = {}
    for filename in tarball.getnames():
        if filename.startswith(f"{HISTORY_DIRECTORY}{user_id}/{history_id}"):
            try:
                f = tarball.extractfile(filename)
                if f is not None:
                    data = f.read().decode("UTF-8")
                    conversation = json.loads(data)
                    timestamp = conversation["metadata"]["timestamp"]
                    query = conversation["redacted_query"].strip()
                    response = conversation["llm_response"].strip()
                    conversation_record = f"\nQ:{query}\nA:{response}\n"

                    if referenced_documents:
                        docs = format_referenced_documents(
                            conversation["referenced_documents"]
                        )
                        conversation_record += f"\nReferenced documents:\n{docs}\n"
                    conversation_record += f"{separator}\n"
                    history[timestamp] = conversation_record
                    statistic.conversation_history_included += 1
                else:
                    logger.error(f"Nothing to extract from {filename}")
            except Exception as e:
                logger.error(f"Unable to read conversation history: {e}")

    output = ""
    for timestamp in sorted(history):
        output += f"{timestamp}\n{history[timestamp]}\n"
    return output


def format_timestamp(text: str) -> str:
    """Format the timestamp into human readable format."""
    timestamp = datetime.strptime(text, "%Y_%m_%d_%H_%M_%S")
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def aggregate_user_feedback_from_files(
    filewriter,
    directory_name: str,
    referenced_documents: bool,
    conversation_history: bool,
) -> None:
    """Aggregate feedbacks from files in specified directory."""
    logger.info(f"Aggregating feedbacks from all tarballs in {directory_name}")
    directory = os.fsencode(directory_name)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".tar.gz"):
            cluster_id = filename[:36]
            timestamp = format_timestamp(filename[37:56])
            logger.info(f"Processing tarball {filename}")
            feedbacks = feedbacks_from_tarball(filename)
            for feedback in feedbacks:
                user_id = feedback["user_id"]
                conversation_id = feedback["conversation_id"]
                rows = [
                    timestamp,
                    cluster_id,
                    user_id,
                    conversation_id,
                    feedback["user_question"].strip(),
                    feedback["llm_response"].strip(),
                    feedback["sentiment"],
                    feedback["user_feedback"],
                ]
                if conversation_history:
                    full_history = read_full_conversation_history(
                        filename,
                        user_id,
                        conversation_id,
                        referenced_documents,
                    )
                    rows.append(full_history)
                filewriter.writerow(rows)
                statistic.feedbacks_aggregated += 1


def aggregate_feedbacks(args: argparse.Namespace) -> None:
    """Aggregate feedback and store it into CSV file."""
    output_filename = args.output
    logger.info(f"Generating {output_filename}")
    with open(output_filename, "w") as csvfile:
        filewriter = csv.writer(
            csvfile,
            delimiter=",",
            quotechar='"',
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
        )
        # column headers written as first row in CSV file
        column_headers = [
            "Timestamp",
            "Cluster ID",
            "User ID",
            "Conversation ID",
            "Question",
            "LLM response",
            "Sentiment",
            "Feedback",
        ]
        if args.conversation_history:
            column_headers.append("Conversation history")

        # write column headers into CSV
        filewriter.writerow(column_headers)

        # write all feedbacks and optionally conversation history into CSV
        aggregate_user_feedback_from_files(
            filewriter,
            args.work_directory,
            args.referenced_documents,
            args.conversation_history,
        )


def read_full_conversation_history_for_all_users(
    tarball_name: str, referenced_documents: bool
) -> list[tuple[str, str, str]]:
    """Read conversation history for all users and return it as list of conversations."""
    logger.info(f"Reading full conversation history from {tarball_name} for all users")
    tarball = tarfile.open(tarball_name, "r:gz")
    filenames = tarball.getnames()
    tarball.close()

    history = []
    for filename in filenames:
        if filename.startswith(f"{HISTORY_DIRECTORY}"):
            parts = filename.split("/")
            if len(parts) <= 2:
                continue
            user_id = parts[1]
            conversation_id = parts[2]
            conversation = read_full_conversation_history(
                tarball_name,
                user_id,
                conversation_id,
                referenced_documents,
            )
            history.append((user_id, conversation_id, conversation))

    return history


def aggregate_conversation_history_from_files(
    filewriter, directory_name: str, referenced_documents: bool
) -> None:
    """Aggregate feedbacks from files in specified directory."""
    logger.info(f"Aggregating feedbacks from all tarballs in {directory_name}")
    directory = os.fsencode(directory_name)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".tar.gz"):
            cluster_id = filename[:36]
            logger.info(f"Processing tarball {filename}")
            full_history = read_full_conversation_history_for_all_users(
                filename, referenced_documents
            )
            for history in full_history:
                row = [cluster_id, *history]
                filewriter.writerow(row)


def aggregate_conversation_history(args: argparse.Namespace) -> None:
    """Aggregate conversation history and store it into CSV file."""
    output_filename = args.conversation_history_output
    logger.info(f"Generating {output_filename}")
    with open(output_filename, "w") as csvfile:
        filewriter = csv.writer(
            csvfile,
            delimiter=",",
            quotechar='"',
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
        )
        # column headers written as first row in CSV file
        column_headers = [
            "Cluster ID",
            "User ID",
            "Conversation ID",
            "Conversation history",
        ]
        # write column headers into CSV
        filewriter.writerow(column_headers)

        # write conversation histories into CSV
        aggregate_conversation_history_from_files(
            filewriter, args.work_directory, args.referenced_documents
        )


def perform_cleanup(args: argparse.Namespace) -> None:
    """Cleanup downloaded files."""
    logger.info("Performing working directory cleanup")
    filenames = os.listdir(args.work_directory)

    for filename in filenames:
        if filename.endswith(".tgz") or filename.endswith(".tar.gz"):
            logger.info(f"Removing {filename}")
            os.remove(os.path.join(args.work_directory, filename))


def print_statistic(statistic: Statistic) -> None:
    """Print statistic onto the standard output."""
    print()
    print("-" * 100)
    print(f"Downloaded tarballs:            {statistic.downloaded_tarballs}")
    print(f"Tarballs not downloaded:        {statistic.tarballs_not_downloaded}")
    print(f"Tarballs with incorrect key:    {statistic.tarballs_with_incorrect_key}")
    print(f"Tarballs without feedback:      {statistic.tarballs_without_feedback}")
    print(f"Feedbacks read:                 {statistic.feedbacks_read}")
    print(f"Feedback read errors:           {statistic.feedback_read_error}")
    print(f"Feedbacks aggregated:           {statistic.feedbacks_aggregated}")
    print(f"Conversation history included:  {statistic.conversation_history_included}")
    print("-" * 100)


def main() -> None:
    """Download feedbacks from Ceph bucket and create CSV file with consolidated report."""
    args = args_parser(sys.argv[1:])
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logger.debug(f"Arguments passed: {args}")

    if args.ping:
        ping_ceph(args)
        return
    if args.list_objects:
        list_objects(args)
        return
    if not args.skip_downloading:
        download_tarballs(args)
    if args.download_only:
        return
    aggregate_feedbacks(args)
    aggregate_conversation_history(args)
    if not args.keep:
        perform_cleanup(args)
    if args.statistic:
        print_statistic(statistic)


# self-checks
correct_keys = (
    "archives/compressed/00/00000000-0000-0000-0000-000000000001/202404/08/073607.tar.gz",
    "archives/compressed/00/00000000-0000-0000-0000-000000000001/123456/67/890123.tar.gz",
    "archives/compressed/ff/ff000000-0000-0000-0000-000000000001/202404/08/073607.tar.gz",
    "archives/compressed/FF/FF000000-0000-0000-0000-000000000001/202404/08/073607.tar.gz",
    "archives/compressed/ff/01234567-89ab-cdef-0123-456789abcdef/202404/08/073607.tar.gz",
    "archives/compressed/FF/01234567-89AB-CDEF-0123-456789ABCDEF/202404/08/073607.tar.gz",
    "archives/compressed/ff/ffffffff-ffff-ffff-ffff-ffffffffffff/202404/08/073607.tar.gz",
    "archives/compressed/FF/FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF/202404/08/073607.tar.gz",
)


@pytest.mark.parametrize("key", correct_keys)
def test_check_key(key):
    """Test the function check_key with valid keys."""
    assert check_key(key)


incorrect_keys = (
    "bad-bad-/compressed/00/00000000-0000-0000-0000-000000000001/202404/08/073607.tar.gz",
    "archives/bad-bad---/00/00000000-0000-0000-0000-000000000001/202404/08/073607.tar.gz",
    "archives/compressed/0/00000000-0000-0000-0000-000000000001/123456/67/890123.tar.gz",
    "archives/compressed/ff/ff000000-0000-0000-0000-000000000001/22404/08/073607.tar.gz",
    "archives/compressed/ff/01234567-89ab-cdef-0123-456789abcdef/2024Z4/08/073607.tar.gz",
    "archives/compressed/fZ/01234567-89ab-cdef-0123-456789abcdef/202404/08/073607.tar.gz",
    "archives/compressed/ff/01234567-89ab-cdZf-0123-456789abcdef/202404/08/073607.tar.gz",
    "archives/compressed/00/00000000-0000-0000-0000-000000000001/20240/08/073607.tar.gz",
    "archives/compressed/00/00000000-0000-0000-0000-000000000001/202404/8/073607.tar.gz",
    "archives/compressed/00/00000000-0000-0000-0000-000000000001/202404/08/73607.tar.gz",
)


@pytest.mark.parametrize("key", incorrect_keys)
def test_check_key_negative(key):
    """Test the function check_key with invalid keys provided at input."""
    assert not check_key(key)


keys_filenames = (
    (
        "archives/compressed/00/00000000-0000-0000-0000-000000000001/202404/08/073607.tar.gz",
        "00000000-0000-0000-0000-000000000001_2024_04_08_07_36_07.tar.gz",
    ),
    (
        "archives/compressed/00/00000000-0000-0000-0000-000000000001/123456/67/890123.tar.gz",
        "00000000-0000-0000-0000-000000000001_1234_56_67_89_01_23.tar.gz",
    ),
    (
        "archives/compressed/ff/ff000000-0000-0000-0000-000000000001/202404/08/073607.tar.gz",
        "ff000000-0000-0000-0000-000000000001_2024_04_08_07_36_07.tar.gz",
    ),
    (
        "archives/compressed/FF/FF000000-0000-0000-0000-000000000001/202404/08/073607.tar.gz",
        "ff000000-0000-0000-0000-000000000001_2024_04_08_07_36_07.tar.gz",
    ),
    (
        "archives/compressed/ff/01234567-89ab-cdef-0123-456789abcdef/202404/08/073607.tar.gz",
        "01234567-89ab-cdef-0123-456789abcdef_2024_04_08_07_36_07.tar.gz",
    ),
    (
        "archives/compressed/FF/01234567-89AB-CDEF-0123-456789ABCDEF/202404/08/073607.tar.gz",
        "01234567-89ab-cdef-0123-456789abcdef_2024_04_08_07_36_07.tar.gz",
    ),
)


@pytest.mark.parametrize("key, filename", keys_filenames)
def test_construct_filename(key, filename):
    """Test the function construct_filename with valid keys provided at input."""
    assert construct_filename(key) == filename


@pytest.mark.parametrize("key", incorrect_keys)
def test_construct_filename_negative(key):
    """Test the function construct_filename - negative test cases with invalid keys."""
    with pytest.raises(Exception, match=f"Can not construct filename from key {key}"):
        construct_filename(key)


if __name__ == "__main__":
    main()
