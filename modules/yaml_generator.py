# base python things
import re, os
from dotenv import load_dotenv

# external deps
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# internal modules
from modules.model_context import get_watsonx_predictor

# internal tools
from tools.ols_logger import OLSLogger

load_dotenv()

DEFAULT_MODEL = os.getenv("YAML_MODEL", "ibm/granite-20b-code-instruct-v1")


class YamlGenerator:
    def __init__(self):
        self.logger = OLSLogger("yaml_generator").logger

    def generate_yaml(self, conversation, string, **kwargs):
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

        self.logger.info(conversation + " using model: " + model)

        bare_llm = get_watsonx_predictor(model=model)

        prompt_instructions = PromptTemplate.from_template("{string}\n")
        llm_chain = LLMChain(llm=bare_llm, verbose=verbose, prompt=prompt_instructions)

        task_query = prompt_instructions.format(string=string)

        self.logger.info(conversation + " task query: " + task_query)
        
        response = llm_chain(inputs={"string": string})

        # https://stackoverflow.com/a/63082323/2328066
        regex = r"(?:\n+|\A)?(?P<code_block>\`{3,}yaml\n+[\S\s]*\`{3,})"

        match = re.search(regex, response["text"])

        if match:
            clean_response = match.group("code_block")
            self.logger.info(conversation + " generated yaml: " + clean_response)
            return clean_response
        else:
            # TODO: need to do the right thing here - raise an exception?
            self.logger.error(conversation + " could not parse response:\n"+ response["text"])
            return "some failure"

if __name__ == "__main__":
    """to execute, from the repo root, use python -m modules.yaml_generator"""
    import argparse

    parser = argparse.ArgumentParser(description="Call the YAML generation model")
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
        default="give me a kubernetes yaml for a project quota for the namespace foo that limits the number of pods to 10",
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

    yaml_generator = YamlGenerator()
    yaml_generator.generate_yaml(args.conversation_id, args.query, model=args.model, verbose=args.verbose)
