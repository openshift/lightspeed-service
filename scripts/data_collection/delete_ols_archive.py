"""Delete selected OLS archive specified by its key from Ceph bucket.

Four environment variables should be set:
    ENDPOINT_URL - Ceph endpoint URL
    AWS_ACCESS_KEY - access key for user or service
    AWS_SECRET_ACCESS_KEY - secret access key for user or service
    BUCKET - bucket name, for example QA-OLS-ARCHIVES or PROD-OLS-ARCHIVES

Key needs to be specified on command line as the only parameter.
"""

import os
import sys

import boto3

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

print(f"Deleting object with key {name}")
client.delete_object(Bucket=bucket_name, Key=name)
print("Deleted")
