"""Web-based user interface handler."""

import json

import gradio as gr
import requests

from ols.utils.logger import Logger


class GradioUI:
    """Handlers for UI-related requests."""

    def __init__(
        self,
        ols_url="http://127.0.0.1:8080/v1/query",
        conversation_id=None,
        logger=None,
    ) -> None:
        """Initialize UI API handlers."""
        self.logger = logger if logger is not None else Logger("gradio_ui").logger
        # class variable
        self.ols_url = ols_url
        self.conversation_id = conversation_id

        # ui specific
        use_history = gr.Checkbox(value=True, label="Use history")
        self.ui = gr.ChatInterface(self.chat_ui, additional_inputs=[use_history])

    def chat_ui(self, prompt, history, use_history=None):
        """Handle requests from web-based user interface."""
        # Headers for the HTTP request
        headers = {"Accept": "application/json", "Content-Type": "application/json"}

        self.logger.info(f"Using history: {use_history!s}")
        # Body of the request (a JSON object with a "query" field)

        data = {"query": prompt}

        if not use_history:
            self.logger.info("Ignoring conversation history")
        elif use_history and self.conversation_id is not None:
            data["conversation_id"] = self.conversation_id
            self.logger.info(f"Using conversation ID: {self.conversation_id}")

        # Convert the data dictionary to a JSON string
        json_data = json.dumps(data)

        try:
            # Make the HTTP POST request, wait for response with 30 seconds timeout
            response = requests.post(
                self.ols_url, headers=headers, data=json_data, timeout=30
            )

            # Check if the request was successful (status code 200)
            if response.status_code == requests.codes.ok:
                self.logger.info("Response JSON:", response.json())
                self.conversation_id = response.json().get("conversation_id")
                return response.json().get("response")
            else:
                self.logger.info(
                    f"Request failed with status code {response.status_code}"
                )
                self.logger.info(f"Response text: {response.text}")
                return f"Sorry, an error occurred: {response.text}"

        except (ValueError, requests.RequestException) as e:
            # Handle any exceptions that may occur during the request
            return f"An error occurred: {e}"

    def mount_ui(self, fast_api_instance, mount_path="/ui"):
        """Register REST API endpoint to handle UI-related requests."""
        return gr.mount_gradio_app(fast_api_instance, self.ui, path=mount_path)


if __name__ == "__main__":
    GradioUI.ui.launch(show_api=False)
