"""A class defining single conversation ."""

from typing import NamedTuple


class Conversation(NamedTuple):
    """Class represents a single conversation between human and assistant ."""

    user_message: str
    assistant_message: str
