# base python things
import os
from dotenv import load_dotenv

# external deps
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# internal modules
from modules.model_context import get_watsonx_predictor

# internal tools
from tools.ols_logger import OLSLogger

load_dotenv()

DEFAULT_MODEL = os.getenv("YESNO_MODEL", "ibm/granite-13b-chat-v1")

class YesNoClassifier:
    def __init__(self):
        self.logger = OLSLogger("yes_no_classifier").logger

    def classify(self, conversation, string, **kwargs):
        if "model" in kwargs:
            model = kwargs["model"]
        else:
            model = DEFAULT_MODEL

        if "verbose" in kwargs:
            if kwargs["verbose"] == "True" or kwargs["verbose"] == "true":
                verbose = True
            else:
                verbose = False
        else:
            verbose = False

        settings_string = f"conversation: {conversation}, query: {string},model: {model}, verbose: {verbose}"
        self.logger.info(
            conversation
            + " call settings: "
            + settings_string
        )

        prompt_instructions = PromptTemplate.from_template(
            """Instructions:
        - determine if a statement is a yes or a no
        - return a 1 if the statement is a yes statement
        - return a 0 if the statement is a no statement
        - return a 9 if you cannot determine if the statement is a yes or no

        Examples:
        Statement: Yes, that sounds good.
        Response: 1

        Statement: No, I don't think that is wise.
        Response: 0

        Statement: Apples are red.
        Response: 9

        Statement: {statement}
        Response:
        """
        )

        self.logger.info(conversation + " using model: " + model)
        self.logger.info(conversation + " determining yes/no: " + string)
        query = prompt_instructions.format(statement=string)

        self.logger.info(conversation + " yes/no query: " + query)
        self.logger.info(conversation + " using model: " + model)


        bare_llm = get_watsonx_predictor(model=model)
        llm_chain = LLMChain(llm=bare_llm, prompt=prompt_instructions, verbose=verbose)

        response = llm_chain(inputs={"statement": string})

        # strip <|endoftext|> from the reponse
        clean_response = response["text"].split("<|endoftext|>")[0]

        self.logger.info(conversation + " yes/no response: " + clean_response)

        # TODO: handle when this doesn't end up with an integer
        return int(clean_response)


if __name__ == "__main__":
    """to execute, from the repo root, use python -m modules.yes_no_classifier.py"""
    import argparse

    parser = argparse.ArgumentParser(description="Process a list of tasks")
    parser.add_argument(
        "-c",
        "--conversation-id",
        default="1234",
        type=str,
        help="A short identifier for the conversation",
    )
    parser.add_argument(
        "-q",
        "--query",
        default="Yes, we have no bananas",
        type=str,
        help="The string to classify",
    )
    parser.add_argument(
        "-m", "--model", default=DEFAULT_MODEL, type=str, help="The model to use"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        help="Set Verbose status of langchains [True/False]",
    )

    args = parser.parse_args()

    yes_no_classifier = YesNoClassifier()
    yes_no_classifier.classify(
        args.conversation_id, args.query, model=args.model, verbose=args.verbose
    )
