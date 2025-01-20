"""Download object specified by its key from Ceph bucket.

Four environment variables should be set:
    ENDPOINT_URL - Ceph endpoint URL
    AWS_ACCESS_KEY - access key for user or service
    AWS_SECRET_ACCESS_KEY - secret access key for user or service
    BUCKET - bucket name, for example QA-OLS-ARCHIVES or PROD-OLS-ARCHIVES

Key needs to be specified on command line as the only parameter.
"""

import os
import re
import sys

import boto3


def construct_filename(key: str) -> str:
    """Construct filename from key."""
    groups = key_match(key)
    if groups is None:
        raise ValueError(f"Can not construct filename from key {key}")

    cluster_id = groups[1].lower()
    year = groups[2][:4]
    month = groups[2][4:]
    day = groups[3]
    hour = groups[4][:2]
    minute = groups[4][2:4]
    second = groups[4][4:]
    return f"{cluster_id}_{year}_{month}_{day}_{hour}_{minute}_{second}.tar.gz"


def key_match(key: str):
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


client = boto3.client(
    "s3",
    endpoint_url=os.getenv("ENDPOINT_URL"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
    region_name="us-east-1",
)

bucket_name = os.getenv("BUCKET")

if len(sys.argv) <= 1:
    print("Object name needs to be specified on command line")
    sys.exit(1)

name = sys.argv[1]
FILENAME = construct_filename(name)

print(f"Downloading object {name} into {FILENAME}")
data = client.get_object(Bucket=bucket_name, Key=name)

with open(FILENAME, "wb") as fout:
    fout.write(data["Body"].read())
