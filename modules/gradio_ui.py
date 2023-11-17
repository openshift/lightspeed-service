import gradio as gr
import requests
import json

class gradioUI:
    def __init__(self,
                 ols_url="http://127.0.0.1:8080/ols",
                 mount_path="/ui",
                 conversation_id=None) -> None:
        # class variable
        self.ols_url=ols_url
        self.mount_path=mount_path
        self.conversation_id = conversation_id
    
        # ui specific
        use_history=gr.Checkbox(value=True, label="Use history")
        self.ui = gr.ChatInterface(self.yaml_gen,
                                    additional_inputs=[
                                        use_history
                                    ])

    def yaml_gen(self, prompt, history, use_history=None):
        # URL of the HTTP endpoint
        url = self.ols_url

        # Headers for the HTTP request
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

        print("Using history: "+str(use_history))
        # Body of the request (a JSON object with a "query" field)
        
        data = {
            "query": prompt
        }

        if not use_history:
            print("Ignoring conversation history")
        elif use_history and self.conversation_id!=None:
            data["conversation_id"]=self.conversation_id
            print(f"Using conversation ID: {self.conversation_id}")

        # Convert the data dictionary to a JSON string
        json_data = json.dumps(data)

        try:
            # Make the HTTP POST request
            response = requests.post(url, headers=headers, data=json_data)

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                print("Response JSON:", response.json())
                self.conversation_id=response.json().get("conversation_id")
                return response.json().get("response")
            else:
                print(f"Request failed with status code {response.status_code}")
                print(f"Response text: {response.text}")
                return "Sorry, an error occurred: "+ response.text
            
        except requests.RequestException as e:
            # Handle any exceptions that may occur during the request
            return f"An error occurred: {e}"

    def mount_ui(self, fast_api_instance):
        return gr.mount_gradio_app(fast_api_instance, 
                                   self.ui, 
                                   path=self.mount_path)

if __name__ == "__main__":
    gradioUI.ui.launch(show_api=False)   


