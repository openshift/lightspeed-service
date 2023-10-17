import logging, sys
from string import Template
from model_context import get_watsonx_predictor
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from string import Template

DEFAULT_MODEL = "ibm/granite-13b-chat-grounded-v01"


class YesNoClassifier:
    def __init__(self):
        logging.basicConfig(
            stream=sys.stdout,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            level=logging.INFO,
        )
        self.logger = logging.getLogger("yes_no_classifier")

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

        # TODO: must be a smarter way to do this
        settings_string = Template(
            '{"conversation": "$conversation", "query": "$query","model": "$model", "verbose": "$verbose"}'
        )

        self.logger.info(
            conversation
            + " call settings: "
            + settings_string.substitute(
                conversation=conversation, query=string, model=model, verbose=verbose
            )
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

        self.logger.info(conversation + " usng model: " + model)
        self.logger.info(conversation + " determining yes/no: " + string)
        query = prompt_instructions.format(statement=string)

        self.logger.info(conversation + " yes/no query: " + query)
        self.logger.info(conversation + " usng model: " + model)

        bare_llm = get_watsonx_predictor(model=model)
        llm_chain = LLMChain(llm=bare_llm, prompt=prompt_instructions, verbose=verbose)

        response = llm_chain(inputs={"statement": string})

        # strip <|endoftext|> from the reponse
        clean_response = response["text"].split("<|endoftext|>")[0]

        self.logger.info(conversation + " yes/no response: " + clean_response)

        # TODO: handle when this doesn't end up with an integer
        return int(clean_response)


if __name__ == "__main__":
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
