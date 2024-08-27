"""List all objects stored in Ceph bucket.

Four environment variables should be set:
    ENDPOINT_URL - Ceph endpoint URL
    AWS_ACCESS_KEY - access key for user or service
    AWS_SECRET_ACCESS_KEY - secret access key for user or service
    BUCKET - bucket name, for example QA-OLS-ARCHIVES or PROD-OLS-ARCHIVES

"""

import os
from pprint import pprint

import boto3


def list_bucket_content(client, bucket_name):
    """List the contents of the specified bucket."""
    try:
        response = client.list_objects_v2(Bucket=bucket_name)
        print("DEBUG: \n", response)
        if "Contents" in response:
            print(f"Contents of bucket '{bucket_name}':")
            for obj in response["Contents"]:
                print(obj["Key"])
        else:
            print(f"Bucket '{bucket_name}' is empty.")
    except Exception as e:
        print(f"Failed to list contents of bucket '{bucket_name}':", e)


def main():
    """Initialize connection to S3 storage, list buckets, and list bucket content."""
    client = boto3.client(
        "s3",
        endpoint_url=os.getenv("ENDPOINT_URL"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
        region_name="us-east-1",
    )

    pprint(client.list_buckets())

    bucket_name = os.getenv("BUCKET")

    list_bucket_content(client, bucket_name)


if __name__ == "__main__":
    main()
