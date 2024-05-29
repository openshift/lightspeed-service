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
    s3_client = boto3.resource(
        service_name="s3",
        region_name=aws_env["aws-region"],
        aws_access_key_id=aws_env["aws-access-key-id"],
        aws_secret_access_key=aws_env["aws-secret-access-key"],
    )

    for file in os.listdir(path):
        if file.endswith(extension):
            file_name = file
    print(f"found a file to upload: {file_name}")
    try:
        if (
            s3_client.meta.client.head_bucket(Bucket=aws_env["aws-bucket"])[
                "ResponseMetadata"
            ]["HTTPStatusCode"]
            == 200
        ):
            with open(file_name, "rb") as tar:
                s3_client.meta.client.upload_fileobj(
                    tar, aws_env["aws-bucket"], file_name
                )
            print("file uploaded to s3")
    except (ClientError, FileNotFoundError) as e:
        print("failed to upload file: ", e)
        return False
    return True
