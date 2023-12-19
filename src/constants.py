# templates
SUMMARIZATION_TEMPLATE = """
The following context contains several pieces of documentation. Please summarize the context for the user.
Documentation context:
{context_str}

Summary:

"""

SUMMARY_TASK_BREAKDOWN_TEMPLATE = """
The following documentation contains a task list. Your job is to extract the list of tasks. If the user-supplied query seems unrelated to the list of tasks, please reply that you do not know what to do with the query and the summary documentation. Use only the supplied content and extract the task list.

Summary document:
{context_str}

User query:
{query_str}

What are the tasks?
"""

TASK_PERFORMER_PROMPT_TEMPLATE = """
Instructions:
- You are a helpful assistant.
- You are an expert in Kubernetes and OpenShift.
- Respond to questions about topics other than Kubernetes and OpenShift with: "I can only answer questions about Kubernetes and OpenShift"
- Refuse to participate in anything that could harm a human.
- Your job is to look at the following description and provide a response.
- Base your answer on the provided task and query and not on prior knowledge.

TASK:
{task}
QUERY:
{query}

Question:
Does the above query contain enough background information to complete the task? Provide a yes or no answer with explanation.

Response:
"""

TASK_REPHRASER_PROMPT_TEMPLATE = """
Instructions:
- You are a helpful assistant.
- Your job is to combine the information from the task and query into a single, new task.
- Base your answer on the provided task and query and not on prior knowledge.

TASK:
{task}
QUERY:
{query}

Please combine the information from the task and query into a single, new task.

Response:
"""

HAPPY_RESPONSE_GENERATOR_PROMPT_TEMPLATE = """
Instructions:
- you are a helpful assistant
- your job is to generate a pleasant response to a question
- you should try to paraphrase the question that was asked in your response
- here are several examples

Examples:
Question: How do I configure autoscaling for my cluster?
Response: I'd be happy to help you with configuring autoscaling for your cluster.

Question: ensure that all volumes created in the namespace backend-recommendations-staging are at least 2 gigabytes in size
Response: OK, I help you with ensuring the volumes are at least 2 gigabytes in size.

Question: give me 5 pod nginx deployment with the 200mi memory limit
Response: I can definitely help create a deployment for that.

Question: {question}
Response:
"""

YES_OR_NO_CLASSIFIER_PROMPT_TEMPLATE = """
Instructions:
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

QUESTION_VALIDATOR_PROMPT_TEMPLATE = """
Instructions:
- You are a question classifying tool
- You are an expert in kubernetes and openshift
- Your job is to determine if a question is about kubernetes or openshift and to provide a one-word response
- If a question is not about kubernetes or openshift, answer with only the word INVALID
- If a question is about kubernetes or openshift, answer with the word VALID
- If a question is not about creating kubernetes or openshift yaml, answer with the word NOYAML
- If a question is about creating kubernetes or openshift yaml, add the word YAML
- Use a comma to separate the words
- Do not provide explanation, only respond with the chosen words

Example Question:
Can you make me lunch with ham and cheese?
Example Response:
INVALID,NOYAML

Example Question:
Why is the sky blue?
Example Response:
INVALID,NOYAML

Example Question:
Can you help configure my cluster to automatically scale?
Example Response:
VALID,NOYAML

Example Question:
please give me a vertical pod autoscaler configuration to manage my frontend deployment automatically.  Don't update the workload if there are less than 2 pods running.
Example Response:
VALID,YAML

Question:
{query}
Response:
"""

YAML_GENERATOR_PROMPT_TEMPLATE = """
Instructions:
- Produce only a yaml response to the user request
- Do not augment the response with markdown or other formatting beyond standard yaml formatting
- Only provide a single yaml object containg a single resource type in your response, do not provide multiple yaml objects

User Request: {query}
"""

YAML_GENERATOR_WITH_HISTORY_PROMPT_TEMPLATE = """
Instructions:
- Produce only a yaml response to the user request
- Do not augment the response with markdown or other formatting beyond standard yaml formatting
- Only provide a single yaml object containg a single resource type in your response, do not provide multiple yaml objects

Here is the history of the conversation so far, you may find this relevant to the user request below:

{history}

User Request: {query}
"""

# providers
PROVIDER_BAM = "bam"
PROVIDER_OPENAI = "openai"
PROVIDER_WATSONX = "watsonx"
PROVIDER_TGI = "tgi"
PROVIDER_OLLAMA = "ollama"

# models
# embedding
TEI_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# BAM
GRANITE_13B_CHAT_V1 = "ibm/granite-13b-chat-v1"
GRANITE_13B_CHAT_V2 = "ibm/granite-13b-chat-v2"
GRANITE_20B_CODE_INSTRUCT_V1 = "ibm/granite-20b-code-instruct-v1"

# OpenAI
GPT35_TURBO_1106 = "gpt-3.5-turbo-1106"
GPT35_TURBO = "gpt-3.5-turbo"

# indexing constants
PRODUCT_INDEX = "product"
PRODUCT_DOCS_PERSIST_DIR = "./vector-db/ocp-product-docs"
SUMMARY_INDEX = "summary"
SUMMARY_DOCS_PERSIST_DIR = "./vector-db/summary-docs"

# cache constants
IN_MEMORY_CACHE = "in-memory"
IN_MEMORY_CACHE_MAX_ENTRIES = 1000
REDIS_CACHE = "redis"
REDIS_CACHE_HOST = "redis-stack.ols.svc"
REDIS_CACHE_PORT = 6379
REDIS_CACHE_MAX_MEMORY = "500mb"
REDIS_CACHE_MAX_MEMORY_POLICY = "allkeys-lru"
