#!/usr/bin/env python3

"""Script to update Google sheet containing user feedback and conversation history.

In order to use this script, file with client secret to Google API must be present (symlinked etc.)

Please look into official documentation how to create service account and retrieve client secret:
https://cloud.google.com/iam/docs/service-accounts-create

usage: update_google_sheet.py [-h]
                              [--feedback-without-history FEEDBACK_WITHOUT_HISTORY]
                              [--feedback-with-history FEEDBACK_WITH_HISTORY]
                              [--conversation-history CONVERSATION_HISTORY]
                              [-s SPREADSHEET]

CSV Uploader

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
"""

import argparse
import csv
import logging
import sys

import pygsheets

logger = logging.getLogger("CSV uploader")


def args_parser(args: list[str]) -> argparse.Namespace:
    """Command line arguments parser."""
    parser = argparse.ArgumentParser(description="CSV uploader")
    parser.add_argument(
        "--feedback-without-history",
        type=str,
        default="feedbacks_without_history.csv",
        help="CSV file containing user feedbacks without conversation history",
    )
    parser.add_argument(
        "--feedback-with-history",
        type=str,
        default="feedbacks_with_history.csv",
        help="CSV file containing user feedbacks with conversation history",
    )
    parser.add_argument(
        "--conversation-history",
        type=str,
        default="conversation_history.csv",
        help="CSV file containing conversation history",
    )
    parser.add_argument(
        "-s",
        "--spreadsheet",
        type=str,
        default="Test",
        help="Spreadsheed title as displayed on Google doc",
    )
    return parser.parse_args(args)


def upload_csv_file(worksheet, filename):
    """Upload selected file into selected worksheet."""
    logger.info("Uploading file %s", filename)

    # Read CSV and update
    with open(filename, "r") as file:
        csv_reader = csv.reader(file)
        worksheet.update_values("A1", list(csv_reader))

    logger.info("Upload finished")


def upload_csv_files(
    spreadsheet_title, feedback, feedback_with_history, conversation_history
):
    """Upload all CSV files into Google spreadsheet."""
    # first authorize access to sheet
    logger.info("Authorizing to Google docs.")
    google_client = pygsheets.authorize(service_file="client_secret.json")

    # open Google sheet
    logger.info("Opening spreadsheet with title %s", spreadsheet_title)
    sheet = google_client.open(spreadsheet_title)
    logger.info("ID=%s", sheet.id)
    logger.info("Title=%s", sheet.title)
    logger.info("URL=%s", sheet.url)

    upload_csv_file(sheet[0], feedback)
    upload_csv_file(sheet[1], feedback_with_history)
    upload_csv_file(sheet[2], conversation_history)

    logger.info("Done")


def main() -> None:
    """Upload user feedbacks and conversation history into selected Google sheet."""
    logging.basicConfig(level=logging.INFO)
    args = args_parser(sys.argv[1:])
    upload_csv_files(
        args.spreadsheet,
        args.feedback_without_history,
        args.feedback_with_history,
        args.conversation_history,
    )


if __name__ == "__main__":
    main()
