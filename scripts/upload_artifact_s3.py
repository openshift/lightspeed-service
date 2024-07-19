"""Upload artifact containing the pytest results and configuration to an s3 bucket.

This will in turn get picked by narberachka service,
which will either send it to ibutsu or report portal, based on the
prefix of the file name.
A dictionary containing the credentials of the S3 bucket must be specified, containing the keys:
AWS_BUCKET
AWS_REGION
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY

"""

import os

import boto3
from botocore.exceptions import ClientError

path = "."
extension = ".tar.gz"


def upload_artifact_s3(aws_env):
    """Upload artifact to the specified S3 bucket.

    Returns:
        True if upload successful.
        False otherwise.
    """
    s3_client = boto3.client(
        service_name="sqs",
        region_name=aws_env["AWS_REGION"],
        aws_access_key_id=aws_env["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=aws_env["AWS_SECRET_ACCESS_KEY"],
    )

    for file in os.listdir(path):
        if file.endswith(extension):
            file_name = file
    print(f"found a file to upload: {file_name}")
    try:
        os.open(file_name, os.O_RDONLY)
    except FileNotFoundError:
        print("Failed to open file")
        return False
    try:
        response = s3_client.upload_file(
            file_name, aws_env["AWS_BUCKET"], "artifact.tar.gz"
        )
        print("file uploaded to s3: ", response)
    except ClientError as e:
        print("failed to upload file: ", e)
        return False
    return True
