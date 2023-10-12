import logging, sys
from string import Template
from model_context import get_watsonx_predictor

class YesNoClassifier():
    def __init__(self):
        logging.basicConfig(
            stream=sys.stdout,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            level=logging.INFO,
        )
        self.logger = logging.getLogger("yes_no_classifier")

    def classify(self, conversation, model, string):
        prompt_instructions = Template(
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

        Statement: ${statement}
        Response:
        """
        )

        self.logger.info(conversation + " usng model: " + model)
        self.logger.info(conversation + " determining yes/no: " + string)
        query = prompt_instructions.substitute(statement=string)

        self.logger.info(conversation + " yes/no query: " + query)
        self.logger.info(conversation + " usng model: " + model)
        bare_llm = get_watsonx_predictor(model=model)
        response = str(bare_llm(query))
        clean_response = response.split("<|endoftext|>")[0]

        self.logger.info(conversation + " yes/no response: " + clean_response)

        # TODO: handle when this doesn't end up with an integer
        return int(clean_response)

if __name__ == "__main__":
    yes_no_classifier = YesNoClassifier()
    # arg 1 is the conversation id
    # arg 2 is the desired model
    # arg 3 is the string to classify
    yes_no_classifier.classify(sys.argv[1], sys.argv[2], sys.argv[3])
