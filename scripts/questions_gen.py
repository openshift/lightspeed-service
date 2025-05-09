"""Utility script to generate automatic questions."""

# TODO: OLS-505 Refactor script scripts/question_get.py to adhere to Python standards and idioms
# pylint: disable=C0413

import argparse
import asyncio
import json
import os
import sys
import time
from enum import IntEnum

import nest_asyncio
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.evaluation import (
    BatchEvalRunner,
    CorrectnessEvaluator,
    DatasetGenerator,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)

# search path accordingly
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

# pylint: disable-next=C0413
from ols.src.llms.llm_loader import load_llm  # pylint: disable=C0413

from our_ols import config

# pylint: disable-next=C0413
from our_ols.constants import (
    CONFIGURATION_FILE_NAME_ENV_VARIABLE,
    DEFAULT_CONFIGURATION_FILE,
)

cfg_file = os.environ.get(
    CONFIGURATION_FILE_NAME_ENV_VARIABLE, DEFAULT_CONFIGURATION_FILE
)
config.reload_from_yaml_file(cfg_file)


def dirs_all_files(folder):
    """Fetch sub dirs of root folder as list."""
    all_dirs = []
    for root, dirs, files in os.walk(folder):
        if files and not dirs:
            all_dirs.append(os.path.join(root))
    return all_dirs


class EvalResponseLength(IntEnum):
    """Enumeration for the possible lengths of the evaluation response."""

    NO_DATA_LINES = 0
    SINGLE_LINES = 1
    DOUBLE_LINES = 2
    TRIPLE_LINES = 3


def eval_parser(eval_response: str):
    """Parse response.

    Args:
        eval_response (str): The response string from the evaluation.

    Returns:
        Tuple[float, str]: A tuple containing the score as a float and the reasoning as a string.
    """
    print("eval_response:", eval_response)

    eval_response_parsed = eval_response.split("\n")
    eval_len = len(eval_response_parsed)

    match eval_len:
        case EvalResponseLength.NO_DATA_LINES:
            return 0, ""
        case EvalResponseLength.SINGLE_LINES:
            return 0, eval_response_parsed[0]
        case EvalResponseLength.DOUBLE_LINES:
            score_str = eval_response_parsed[0]
            score = float(score_str) if score_str else 0
            reasoning = eval_response_parsed[1]
            return score, reasoning
        case EvalResponseLength.TRIPLE_LINES:
            score_str, reasoning_str = eval_response_parsed[1], eval_response_parsed[2]
            score = float(score_str) if score_str else 0
            reasoning = reasoning_str.lstrip("\n")
            return score, reasoning
        case _:
            return 0, eval_response


def get_eval_results(key, eval_results):
    """Evaluate LLM response."""
    results = eval_results[key]
    correct = 0
    for result in results:
        if result.passing:
            correct += 1
    score = correct / len(results)
    print(f"{key} Score: {score}")
    return score


def get_eval_breakdown(key, eval_results):
    """Write evaluation result."""
    results = eval_results[key]
    response = []
    for result in results:
        res = {
            "query": None if result.query is None else result.query,
            "response": result.response,
            "score": result.score,
            "passing": result.passing,
            "invalid_reason": result.invalid_reason,
            "feedback": result.feedback,
        }
        response.append(res)
    return response


def generate_summary(
    start_time,
    args,
    persist_folder,
    product_index,
    full_results,
    total_correctness_score,
):
    """Generate evaluation final summary."""
    end_time = time.time()
    execution_time_seconds = end_time - start_time
    print(f"** Total execution time in min: {execution_time_seconds/60}")

    metadata = create_metadata(
        args,
        execution_time_seconds,
        product_index,
        total_correctness_score,
        full_results,
    )
    write_json_metadata(persist_folder, args, metadata)
    write_markdown_metadata(persist_folder, args, metadata, full_results)


def create_metadata(
    args, execution_time_seconds, product_index, total_correctness_score, full_results
):
    """Create metadata for the summary."""
    metadata = {
        "execution-time-MIN": execution_time_seconds,
        "llm": args.provider,
        "model": args.model,
        "index-id": product_index,
    }
    if args.include_evaluation == "true":
        metadata["correctness-results"] = sum(total_correctness_score) / len(
            total_correctness_score
        )
    metadata["evaluation-results"] = full_results
    return metadata


def write_json_metadata(persist_folder, args, metadata):
    """Write the JSON metadata to a file."""
    if not os.path.exists(persist_folder):
        os.makedirs(persist_folder)

    model_name_formatted = args.model.replace("/", "_")
    file_path = (
        f"{persist_folder}/questions-{args.provider}-{model_name_formatted}.json"
    )

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file)


def write_markdown_metadata(persist_folder, args, metadata, full_results):
    """Write the metadata to a markdown file."""
    full_results_markdown_content = generate_full_results_markdown(full_results)
    metadata["evaluation-results"] = full_results_markdown_content

    markdown_content = convert_metadata_to_markdown(metadata)

    model_name_formatted = args.model.replace("/", "_")
    file_path = (
        f"{persist_folder}/questions-{args.provider}-{model_name_formatted}_metadata.md"
    )

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(markdown_content)


def generate_full_results_markdown(full_results):
    """Generate markdown content for full results."""
    full_results_markdown_content = "    \n"
    for res in full_results:
        for key, value in res.items():
            if (
                isinstance(value, list)
                and len(value) > 0
                and isinstance(value[0], dict)
            ):
                new = "\n"
                for d in value:
                    for k, v in d.items():
                        new += f"   - {k}: {v}\n"
            full_results_markdown_content += f"    - {key}: {new}\n"
    return full_results_markdown_content


def convert_metadata_to_markdown(metadata):
    """Convert metadata dictionary to markdown formatted string."""
    markdown_content = "```markdown\n"
    for key, value in metadata.items():
        markdown_content += f"- {key}: {value}\n"
    markdown_content += "```"
    return markdown_content


async def questions_eval(
    start_time,
    similarity,
    index,
    full_results,
    total_correctness_score,
    results,
    eval_questions,
):
    """Evaluate questions."""
    print("*** start evaluation")

    faithfulness_results, relevancy_results = await evaluate_faithfulness_relevancy(
        index, similarity, eval_questions
    )
    results["faithfulness"] = faithfulness_results
    results["relevancy"] = relevancy_results

    print_evaluation_time("faithfulness, relevancy", start_time)

    correctness_results = evaluate_correctness(
        index, similarity, eval_questions, total_correctness_score
    )
    results["correctness"] = correctness_results

    full_results.append(results)
    print_evaluation_time("correctness", start_time)


async def evaluate_faithfulness_relevancy(index, similarity, eval_questions):
    """Evaluate faithfulness and relevancy."""
    faithfulness = FaithfulnessEvaluator()
    relevancy = RelevancyEvaluator()

    runner = BatchEvalRunner(
        {
            "faithfulness": faithfulness,
            "relevancy": relevancy,
        },
        workers=100,
        show_progress=True,
    )

    eval_results = await runner.aevaluate_queries(
        index.as_query_engine(similarity_top_k=similarity),
        queries=eval_questions,
    )

    faithfulness_results = get_eval_breakdown("faithfulness", eval_results)
    relevancy_results = get_eval_breakdown("relevancy", eval_results)

    return faithfulness_results, relevancy_results


def evaluate_correctness(index, similarity, eval_questions, total_correctness_score):
    """Evaluate correctness."""
    correctness = CorrectnessEvaluator(
        score_threshold=2.0,
        parser_function=eval_parser,
    )

    engine = index.as_query_engine(similarity_top_k=similarity)
    correctness_results = []

    for query in eval_questions:
        res_row = evaluate_single_query(
            engine, query, correctness, total_correctness_score
        )
        correctness_results.append(res_row)

    return correctness_results


def evaluate_single_query(engine, query, correctness, total_correctness_score):
    """Evaluate a single query for correctness."""
    summary = engine.query(query)
    referenced_documents = "\n".join(
        [source_node.node.metadata["file_name"] for source_node in summary.source_nodes]
    )

    result = correctness.evaluate(
        query=query,
        response=summary.response,
        reference=summary.source_nodes[0].text,
    )

    res_row = {
        "query": query,
        "response": summary.response,
        "ref": referenced_documents,
        "ref_doc": summary.source_nodes[0].text,
        "ref_doc_score": summary.source_nodes[0].score,
        "passing": result.passing,
        "feedback": result.feedback,
        "score": result.score,
    }

    # it is not possible to convert the value w/o checks
    if result.score is not None:
        total_correctness_score.append(float(result.score))

    return res_row


def print_evaluation_time(stage, start_time):
    """Print evaluation time for a specific stage."""
    end_time = time.time()
    execution_time_seconds = end_time - start_time
    print(
        f"Completed {stage} evaluation: execution time in min:{execution_time_seconds/60}"
    )


# pylint: disable-next=R0915
async def main():
    """Execute main function."""
    start_time = time.time()
    # collect args
    parser = argparse.ArgumentParser(description="question gen cli for task execution")
    parser.add_argument(
        "-p",
        "--provider",
        default="bam",
        help="LLM provider supported value: bam, openai",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="local:sentence-transformers/all-mpnet-base-v2",
        help="the valid models are:\
            - ibm/granite-3-8b-instruct \
            - gpt-3.5-turbo-1106, gpt-3.5-turbo, gpt-4o-mini for openai",
    )
    parser.add_argument(
        "-x", "--product-index", default="applications", help="storage product index"
    )
    parser.add_argument("-i", "--input-persist-dir", help="path to persist file dir")
    parser.add_argument(
        "-q", "--question-main-folder", default="", help="docs folder for questions gen"
    )
    parser.add_argument(
        "-n",
        "--number-of-questions",
        type=int,
        default="5",
        help="number of questions per file for evaluation",
    )
    parser.add_argument(
        "-s", "--similarity", type=int, default="5", help="similarity_top_k"
    )
    parser.add_argument(
        "-e",
        "--include-evaluation",
        default=False,
        action="store_true",
        help="Include evaluation",
    )
    parser.add_argument(
        "-c", "--chunk", type=int, default="512", help="chunk size for embedding"
    )
    parser.add_argument(
        "-l", "--overlap", type=int, default="50", help="chunk overlap for embedding"
    )
    parser.add_argument("-o", "--output", default="./output", help="persist folder")

    # execute
    args = parser.parse_args()

    persist_folder = args.output
    product_index = args.product_index
    product_docs_persist_dir = args.input_persist_dir
    num_of_questions = args.number_of_questions
    similarity = args.similarity
    chunk_size = args.chunk
    chunk_overlap = args.overlap

    print("** settings params")

    Settings.chunk_size = chunk_size
    Settings.chunk_overlap = chunk_overlap
    Settings.embed_model = "local:sentence-transformers/all-mpnet-base-v2"
    Settings.llm = load_llm(args.provider, args.model)

    print("** settings context")
    storage_context = StorageContext.from_defaults(persist_dir=product_docs_persist_dir)

    print("** load embeddings evaluating")

    index = load_index_from_storage(
        storage_context=storage_context,
        index_id=product_index,
    )
    nest_asyncio.apply()

    print("*** generating questions ")
    question_folder = (
        args.folder if args.question_main_folder is None else args.question_main_folder
    )
    dir_list = dirs_all_files(question_folder)

    if len(dir_list) == 0:
        raise Exception("couldn't find dirs in questions folder")

    print("*** starting question iteration ")

    full_results = []
    total_correctness_score = []
    for directory in dir_list:
        print(f"gen questions for: {directory}")
        results = {}
        reader = SimpleDirectoryReader(directory)
        question = reader.load_data()
        data_generator = DatasetGenerator.from_documents(question)
        eval_questions = data_generator.generate_questions_from_nodes(
            num=num_of_questions
        )

        results["dir_name"] = directory
        results["questions"] = eval_questions
        print(eval_questions)

        if args.include_evaluation == "true":
            await questions_eval(
                start_time,
                similarity,
                index,
                full_results,
                total_correctness_score,
                results,
                eval_questions,
            )
        else:
            full_results.append(results)

    generate_summary(
        start_time,
        args,
        persist_folder,
        product_index,
        full_results,
        total_correctness_score,
    )

    return "Completed"


asyncio.run(main())
