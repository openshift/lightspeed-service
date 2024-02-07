"""Web-based user interface handler."""

import json
import logging
from typing import Optional

import gradio as gr
import requests
from fastapi import FastAPI

logger = logging.getLogger(__name__)


class GradioUI:
    """Handlers for UI-related requests."""

    def __init__(
        self,
        ols_url: str = "http://127.0.0.1:8080/v1/query",
        conversation_id: Optional[str] = None,
    ) -> None:
        """Initialize UI API handlers."""
        # class variable
        self.ols_url = ols_url
        self.conversation_id = conversation_id

        # ui specific
        use_history = gr.Checkbox(value=True, label="Use history")
        provider = gr.Textbox(value=None, label="Provider")
        model = gr.Textbox(value=None, label="Model")
        self.ui = gr.ChatInterface(
            self.chat_ui, additional_inputs=[use_history, provider, model]
        )

    def chat_ui(
        self,
        prompt: str,
        history,
        use_history: Optional[bool] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """Handle requests from web-based user interface."""
        # Headers for the HTTP request
        headers = {"Accept": "application/json", "Content-Type": "application/json"}

        logger.info(f"Using history: {use_history!s}")
        # Body of the request (a JSON object with a "query" field)

        data = {"query": prompt}

        if not use_history:
            logger.info("Ignoring conversation history")
        elif use_history and self.conversation_id is not None:
            data["conversation_id"] = self.conversation_id
            logger.info(f"Using conversation ID: {self.conversation_id}")

        if provider:
            logger.info(f"Using provider: {provider}")
            data["provider"] = provider
        if model:
            logger.info(f"Using model: {model}")
            data["model"] = model

        # Convert the data dictionary to a JSON string
        json_data = json.dumps(data)

        try:
            # Make the HTTP POST request, wait for response with 30 seconds timeout
            response = requests.post(
                self.ols_url, headers=headers, data=json_data, timeout=30
            )

            # Check if the request was successful (status code 200)
            if response.status_code == requests.codes.ok:
                logger.info("Response JSON:", response.json())
                self.conversation_id = response.json().get("conversation_id")
                return response.json().get("response")
            else:
                logger.info(f"Request failed with status code {response.status_code}")
                logger.info(f"Response text: {response.text}")
                return f"Sorry, an error occurred: {response.text}"

        except (ValueError, requests.RequestException) as e:
            # Handle any exceptions that may occur during the request
            return f"An error occurred: {e}"

    def mount_ui(self, fast_api_instance: FastAPI, mount_path: str = "/ui"):
        """Register REST API endpoint to handle UI-related requests."""
        return gr.mount_gradio_app(fast_api_instance, self.ui, path=mount_path)


if __name__ == "__main__":
    GradioUI.ui.launch(show_api=False)
