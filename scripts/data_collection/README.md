# Data collection tools

## `user_data_aggregator.py`

### Description

Script to download feedbacks from Ceph bucket and create CSV file with consolidated report.

### Usage

```
usage: user_data_aggregator.py [-h] [-e ENDPOINT] [-b BUCKET]
                               [--access-key ACCESS_KEY] [--secret-access-key SECRET_ACCESS_KEY]
                               [-r REGION] [-p] [-k KEEP]
                               [-s] [-d] [-l] [-o OUTPUT] [-w WORK_DIRECTORY] [-t] [-v]
```


### Typical use cases

- Test if Ceph bucket is accessible:
  ```
  ./user_data_aggregator.py -e URL --access-key=KEY --secret-access-key=KEY --bucket=BUCKET -p
  ```
- List all objects stored in Ceph bucket:
  ```
  ./user_data_aggregator.py -e URL --access-key=KEY --secret-access-key=KEY --bucket=BUCKET -l
  ```
- Download tarballs, aggregate feedback, and cleanup tarballs:
  ```
  ./user_data_aggregator.py -e URL --access-key=KEY --secret-access-key=KEY --bucket=BUCKET
  ```
- Download tarballs, aggregate feedback, without cleanup:
  ```
  ./user_data_aggregator.py -e URL --access-key=KEY --secret-access-key=KEY --bucket=BUCKET -k
  ```
- Download tarballs only:
  ```
  ./user_data_aggregator.py -e URL --access-key=KEY --secret-access-key=KEY --bucket=BUCKET -d
  ```


## `list_objects_in_ceph.py`

### Description

List all objects stored in Ceph bucket.

### Usage

Four environment variables should be set before running the script:

- `ENDPOINT_URL` - Ceph endpoint URL
- `AWS_ACCESS_KEY` - access key for user or service
- `AWS_SECRET_ACCESS_KEY` - secret access key for user or service
- `BUCKET` - bucket name, for example QA-OLS-ARCHIVES or PROD-OLS-ARCHIVES

Run:

```
python list_objects_in_ceph.py
```

## `download_ols_archive.py`

### Description

Download object specified by its key from Ceph bucket. Usually the object
contains a tarball with conversation history and/or user feedback.

### Usage

Four environment variables should be set before running the script:

- `ENDPOINT_URL` - Ceph endpoint URL
- `AWS_ACCESS_KEY` - access key for user or service
- `AWS_SECRET_ACCESS_KEY` - secret access key for user or service
- `BUCKET` - bucket name, for example QA-OLS-ARCHIVES or PROD-OLS-ARCHIVES

Run:

```
python download_ols_archive.py key
```

An example:

```
python download_ols_archive.py archives/compressed/b6/b6cf044a-4870-4a8c-a849-1b0016cc8171/202404/25/185345.tar.gz
```

## `delete_ols_archive.py`

### Description

Delete OLS archive specified by its key from Ceph bucket. Usually the object
contains a tarball with conversation history and/or user feedback.

### Usage

Four environment variables should be set before running the script:

- `ENDPOINT_URL` - Ceph endpoint URL
- `AWS_ACCESS_KEY` - access key for user or service
- `AWS_SECRET_ACCESS_KEY` - secret access key for user or service
- `BUCKET` - bucket name, for example QA-OLS-ARCHIVES or PROD-OLS-ARCHIVES

Run:

```
python delete_ols_archive.py key
```

An example:

```
python delete_ols_archive.py archives/compressed/b6/b6cf044a-4870-4a8c-a849-1b0016cc8171/202404/25/185345.tar.gz
```

## `print_acls.py`

Print ACLs defined in selected Ceph bucket for all users and tools.

### Usage

Four environment variables should be set before running the script:

- `ENDPOINT_URL` - Ceph endpoint URL
- `AWS_ACCESS_KEY` - access key for user or service
- `AWS_SECRET_ACCESS_KEY` - secret access key for user or service
- `BUCKET` - bucket name, for example QA-OLS-ARCHIVES or PROD-OLS-ARCHIVES

Run:

```
python print_acls.py
```

An example of output:

```
user1 user1 READ
user1 user1 WRITE
user2 user2 READ
admin admin FULL_CONTROL
```

## `update_google_sheet.py`

### Description

Script to update Google sheet containing user feedback and conversation
history.

### Usage

```
usage: update_google_sheet.py [-h]
                              [--feedback-without-history FEEDBACK_WITHOUT_HISTORY]
                              [--feedback-with-history FEEDBACK_WITH_HISTORY]
                              [--conversation-history CONVERSATION_HISTORY]
                              [-s SPREADSHEET]

options:
  -h, --help            show this help message and exit
  --feedback-without-history FEEDBACK_WITHOUT_HISTORY
                        CSV file containing user feedbacks without conversation history
  --feedback-with-history FEEDBACK_WITH_HISTORY
                        CSV file containing user feedbacks with conversation history
  --conversation-history CONVERSATION_HISTORY
                        CSV file containing conversation history
  -s SPREADSHEET, --spreadsheet SPREADSHEET
                        Spreadsheed title as displayed on Google doc
```
