"""Fake ingress api for testing purposes."""

import io
import logging
import tarfile

from fastapi import FastAPI, UploadFile
from starlette.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()


@app.post("/test_endpoint")
async def test_endpoint(file: UploadFile):
    """Test endpoint for uploading a tarball."""
    contents = await file.read()
    logging.info("received something")
    tar = tarfile.open(fileobj=io.BytesIO(contents), mode="r:gz")
    for member in tar.getmembers():
        logging.info(member.name)
    return JSONResponse(status_code=202, content={"request_id": "some-request-id"})
