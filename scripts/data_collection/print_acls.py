"""Print ACLs defined in selected Ceph bucket for all users and tools.

Four environment variables should be set:
    ENDPOINT_URL - Ceph endpoint URL
    AWS_ACCESS_KEY - access key for user or service
    AWS_SECRET_ACCESS_KEY - secret access key for user or service
    BUCKET - bucket name, for example QA-OLS-ARCHIVES or PROD-OLS-ARCHIVES

"""

import os

import boto3

client = boto3.client(
    "s3",
    endpoint_url=os.getenv("ENDPOINT_URL"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
    region_name="us-east-1",
)

bucket_name = os.getenv("BUCKET")

# retrieve dictionary containing all information about ACL
grants = client.get_bucket_acl(Bucket=bucket_name)["Grants"]

# print ACLS in simple tabular format
for g in grants:
    print(g["Grantee"]["DisplayName"], g["Grantee"]["ID"], g["Permission"])
