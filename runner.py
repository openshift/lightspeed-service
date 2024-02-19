"""Script that is able to start Uvicorn-based REST API service."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("ols.app.main:app", host="127.0.0.1", port=8080, reload=True)
