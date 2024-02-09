"""Mocked FastAPI app."""


class MockApp:
    """Mocked FastAPI app."""

    def __init__(self):
        """Initialize call memory."""
        self.routers = []
        self.prefixes = []
        self.mounts = []

    def include_router(self, router, prefix=None):
        """Remember which router has been included."""
        self.routers.append(router)
        self.prefixes.append(prefix)

    def mount(self, mount, *args):
        """Remember which mount point has been registered."""
        self.mounts.append(mount)
