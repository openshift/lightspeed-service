import llama_index
from llama_index import StorageContext, load_index_from_storage
from model_context import get_watsonx_context
from llama_index.prompts import Prompt, PromptTemplate

import logging

def task_breakdown(conversation, query):
    llama_index.set_global_handler("simple")
    summary_task_breakdown_template_str = (
        "{context_str}\n"
        "Given the previous smmary documentation, what steps would you take to answer the following question: {query_str}\n"
    )
    summary_task_breakdown_template = PromptTemplate(summary_task_breakdown_template_str)
    
    logging.info(conversation + ' Getting sevice context')
    service_context = get_watsonx_context(model="ibm/granite-13b-instruct-v1")
    
    storage_context = StorageContext.from_defaults(persist_dir="../vector-db")
    logging.info(conversation + ' Setting up index')
    index = load_index_from_storage(storage_context=storage_context, index_id="summary", service_context=service_context)
    
    logging.info(conversation + ' Setting up query engine')
    query_engine = index.as_query_engine(text_qa_template=summary_task_breakdown_template, verbose=True, streaming=False)

    logging.info(conversation + ' Submitting task breakdown query')
    task_breakdown_response = query_engine.query(query)

    referenced_documents = ""
    for source_node in task_breakdown_response.source_nodes:
        #print(source_node.node.metadata['file_name'])
        referenced_documents += source_node.node.metadata['file_name'] + '\n'
    
    for line in str(task_breakdown_response).splitlines():
        logging.info(conversation + ' Task breakdown response: ' + line)
    for line in referenced_documents.splitlines():
        logging.info(conversation + ' Referenced documents: ' + line)
    
    return str(task_breakdown_response).splitlines(), referenced_documents
