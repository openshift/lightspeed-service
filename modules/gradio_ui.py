import gradio as gr
import requests
import json

conversation_id=None

def yaml_gen(prompt,history,use_history):
    global conversation_id
    # URL of the HTTP endpoint
    url = "http://127.0.0.1:8080/ols"

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
    elif use_history and conversation_id!=None:
        data["conversation_id"]=conversation_id
        print("Using conversation ID: "+conversation_id)

    # Convert the data dictionary to a JSON string
    json_data = json.dumps(data)

    try:
        # Make the HTTP POST request
        response = requests.post(url, headers=headers, data=json_data)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            print("Response JSON:", response.json())
            conversation_id=response.json().get("conversation_id")
            return response.json().get("response")
        else:
            print(f"Request failed with status code {response.status_code}")
            print("Response text:", response.text)
            return "Sorry, an error occurred: "+ response.text
    except requests.RequestException as e:
        # Handle any exceptions that may occur during the request
        return f"An error occurred: {e}"


use_history=gr.Checkbox(value=True, label="Use history")
ui = gr.ChatInterface(yaml_gen,
                        additional_inputs=[
                            use_history
                        ])


if __name__ == "__main__":
    ui.launch(show_api=False)   


