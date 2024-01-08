import uuid


class Utils:
    """Utility class containing common methods."""

    @staticmethod
    def get_suid() -> str:
        """Generate a unique session ID (SUID) using UUID4.

        Returns:
            str: A unique session ID.
        """
        return str(uuid.uuid4().hex)
