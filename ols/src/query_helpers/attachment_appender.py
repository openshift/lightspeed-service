"""Function to append attachments to query."""

from typing import Optional

import yaml

from ols.app.models.models import Attachment

# mapping between content-type and language specification in Markdown
content_type_to_markdown = {
    "text/plain": "",
    "application/json": "json",
    "application/yaml": "yaml",
    "application/xml": "xml",
}


def append_attachments_to_query(query: str, attachments: list[Attachment]) -> str:
    """Append all attachments to query."""
    output = query
    for attachment in attachments:
        output += format_attachment(attachment)
    return output


def format_attachment(attachment: Attachment) -> str:
    """Format attachment to be included into query."""
    intro_message = ""

    if attachment.content_type == "application/yaml":
        intro_message = construct_intro_message(attachment.content)

    # attachments were tested for proper content types already, so
    # no KeyError should be thrown there
    header = "```" + content_type_to_markdown[attachment.content_type]
    footer = "```"

    return f"""

{intro_message}
{header}
{attachment.content}
{footer}
"""


def construct_intro_message(content: str) -> str:
    """Construct intro message for given attachment."""
    kind, name = retrieve_kind_name_from_yaml(content)
    if kind is not None and name is not None:
        return f"For reference, here is the full resource YAML for {kind} '{name}':"
    return "For reference, here is the full resource YAML:"


def retrieve_kind_name_from_yaml(content: str) -> tuple[Optional[str], Optional[str]]:
    """Try to parse YAML file and retrieve kind and name attributes from it."""
    try:
        d = yaml.safe_load(content)
        kind = d.get("kind")
        name = d.get("metadata", {}).get("name")
        return kind, name
    except Exception:
        return None, None
