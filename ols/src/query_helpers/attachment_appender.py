"""Function to append attachments to query."""

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
    # attachments were tested for proper content types already, so
    # no KeyError should be thrown there
    header = "```" + content_type_to_markdown[attachment.content_type]
    footer = "```"

    return f"""
{header}
{attachment.content}
{footer}
"""
