"""Response evaluation using pre-defined question/answer pair."""

import json
import os
from collections import defaultdict
from datetime import UTC, datetime

import requests
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pandas import DataFrame, read_csv, read_parquet
from rouge_score.rouge_scorer import RougeScorer
from scipy.spatial.distance import cosine, euclidean

# Same Provider/Model combination must be used while launching OLS.
INSCOPE_MODELS = {
    "bam+granite13b-chatv2": ("bam", "ibm/granite-13b-chat-v2"),
    "watsonx+granite13b-chatv2": ("watsonx", "ibm/granite-13b-chat-v2"),
    "openai+gpt35-turbo": ("openai", "gpt-3.5-turbo"),
    "azure+gpt35-turbo-4k": ("azure_openai", "gpt-3.5-turbo"),
    "azure+gpt35-turbo-16k": ("azure_openai", "gpt-3.5-turbo"),
    "azure+gpt4o": ("azure_openai", "gpt-4o"),
}

# Cut-off similarity score used for response evaluation.
EVAL_THRESHOLD = 0.2  # low score is better


# TODO: OLS-712 Enrichment of Q+A pairs to contain questions with attachments
# TODO: Refactor, make it more modular
class ResponseEvaluation:
    """Evaluate LLM response."""

    def __init__(self, eval_args, api_client):
        """Initialize."""
        print(f"Response evaluation arguments: {eval_args}")
        self._args = eval_args
        self._api_client = api_client

        self._embedding_model = HuggingFaceEmbedding(
            "sentence-transformers/all-mpnet-base-v2"
        )
        self._rouge_scorer = RougeScorer(["rougeL"], use_stemmer=True)

        eval_dir = os.path.dirname(__file__)
        self._input_dir = os.path.join(eval_dir, "eval_data")
        with open(os.path.join(self._input_dir, "question_answer_pair.json")) as qna_f:
            self._qa_pairs = json.load(qna_f)["evaluation"]

        self._result_dir = os.path.join(
            (self._args.eval_out_dir or eval_dir), "eval_result"
        )
        os.makedirs(self._result_dir, exist_ok=True)

    def _get_api_response(self, question, provider, model):
        """Get api response for a question/query."""
        retry_counter = 1
        while retry_counter <= 3:
            print(f"OLS call; attempt: {retry_counter}")
            response = self._api_client.post(
                "/v1/query",
                json={
                    "query": question,
                    "provider": provider,
                    "model": model,
                },
                timeout=120,
            )
            if response.status_code == requests.codes.ok:
                break
            retry_counter += 1

        if response.status_code != requests.codes.ok:
            print(f"Unable to get response for {provider}+{model}; query: {question}")
            raise Exception(response)

        print(f"API request is successful for {provider}+{model}; Query: {question}")
        return response.json()["response"].strip()

    def _get_recent_response(self, recent_resp_df, question, provider, model):
        """Get llm response from the stored data, if available."""
        if recent_resp_df is not None:
            try:
                return recent_resp_df[recent_resp_df.question == question][
                    "llm_response"
                ].iloc[0]
            except IndexError:
                print(
                    "Recent response for query is not found in the file. "
                    "Separate api call is required to get response."
                )
        # Recent response is not found, call api to get response
        return self._get_api_response(question, provider, model)

    def _similarity_score(self, response, answer):
        """Calculate similarity score between two strings."""
        res_vec = self._embedding_model.get_text_embedding(response)
        ans_vec = self._embedding_model.get_text_embedding(answer)

        # Distance score
        cos_score = cosine(res_vec, ans_vec)
        euc_score = euclidean(res_vec, ans_vec)

        # Naive length consideration with reduced weightage.
        len_res, len_ans = len(response), len(answer)
        len_score = (abs(len_res - len_ans) / (len_res + len_ans)) * 0.1

        score = len_score + (cos_score + euc_score) / 2
        # TODO: OLS-409 Use non-contextual score to evaluate response

        print(
            f"cos_score: {cos_score}, "
            f"euc_score: {euc_score}, "
            f"len_score: {len_score}\n"
            f"final_score: {score}"
        )
        return score

    def _f1_score(self, response, answer):
        """Get f1 score."""
        score = self._rouge_scorer.score(answer, response)
        print(f"Rouge score: \n{score}")

        # Use f1-score
        return score["rougeL"].fmeasure

    def _get_evaluation_score(self, answer_id):
        """Get response evaluation score."""
        result_dict = defaultdict(list)

        query_ids = self._args.eval_query_ids
        if not query_ids:
            query_ids = self._qa_pairs.keys()

        for query_id in query_ids:
            answer_data = self._qa_pairs[query_id]["answer"][answer_id]
            if not answer_data.get("in_use", True):
                continue

            question = self._qa_pairs[query_id]["question"]
            answers = answer_data["text"]
            eval_threshold = answer_data.get("cutoff_score", EVAL_THRESHOLD)

            response = self._get_api_response(
                question, self._args.eval_provider, self._args.eval_model
            )

            print(f"Calculating score for query: {question}")
            for answer in answers:
                score = self._similarity_score(response, answer)

                if score > eval_threshold:
                    print(
                        f"Response is not as expected for question: {question}\n"
                        f"Score: {score} is above cut-off value: {eval_threshold}"
                    )

                result_dict["query_id"].append(query_id)
                result_dict["question"].append(question)
                result_dict["answer"].append(answer)
                result_dict["llm_response"].append(response)
                result_dict["consistency_score"].append(score)
                result_dict["cutoff_score"].append(eval_threshold)

        return DataFrame.from_dict(result_dict)

    def validate_response(self):
        """Validate LLM response."""
        answer_id = (
            f"{self._args.eval_provider}+"
            f"{self._args.eval_model}+"
            f"{self._args.eval_scenario}"
        )
        result_df = self._get_evaluation_score(answer_id)

        if len(result_df) > 0:
            result_df["answer_eval_fail_flag"] = (
                result_df.consistency_score > result_df.cutoff_score
            )
            # If none of the answer for any question has score below threshold,
            # then mark as evaluation failure.
            result_df["question_eval_fail_flag"] = result_df.groupby(
                "query_id"
            ).answer_eval_fail_flag.transform("min")

            result_file = (
                f"{self._result_dir}/response_evaluation_result-"
                f"{answer_id.replace('/', '-')}.csv"
            )
            result_df.to_csv(result_file, index=False)
            print(f"Result is saved to {result_file}")

            if result_df.question_eval_fail_flag.max() == 1:
                # If evaluation has failed for any question,
                # then return False (Failed validation scenario)
                print(
                    "Response is not matching for question(s):\n"
                    f"Please check result in {result_file}."
                )
                return False
        else:
            print("No result. Nothing to process.")

        return True

    def _populate_eval_data_for_qna_json(
        self, provider_and_model, answer_id, result_dict
    ):
        """Populate evaluation data for QnAs from json file."""
        provider, model = provider_and_model
        provider_model = "+".join(provider_and_model)

        try:
            recent_resp_df = read_csv(
                f"{self._result_dir}/response_evaluation_result-"
                f"{provider_model.replace('/', '-')}+with_rag.csv"
            )
        except FileNotFoundError:
            print(
                "File with recent model response not found. "
                "Separate api calls are required to get the response."
            )
            recent_resp_df = None

        for query_id in self._qa_pairs.keys():
            question = self._qa_pairs[query_id]["question"]
            answer_data = self._qa_pairs[query_id]["answer"][answer_id]
            answer = answer_data["text"][0]

            response = self._get_recent_response(
                recent_resp_df, question, provider, model
            )
            sim_score = self._similarity_score(response, answer)
            f1_score = self._f1_score(response, answer)

            result_dict["query_id"].append(query_id)
            result_dict["question"].append(question)
            result_dict[answer_id].append(answer)
            result_dict[f"{provider_model}_response"].append(response)
            result_dict[f"{provider_model}_similiarity-score"].append(sim_score)
            result_dict[f"{provider_model}_f1-score"].append(f1_score)
            result_dict["query_source"].append("transcript")
            result_dict["doc_page"].append(None)
            result_dict["doc_title"].append(None)

        return result_dict

    def _populate_eval_data_for_qna_parquet(
        self, provider_and_model, answer_id, result_dict
    ):
        """Populate evaluation data for QnAs from parquet file."""
        provider, model = provider_and_model
        provider_model = "+".join(provider_and_model)

        qa_pool_df = read_parquet(self._args.qna_pool_file)

        for indx in range(qa_pool_df.shape[0]):
            qa_series = qa_pool_df.iloc[indx]
            question = qa_series["Question"]
            answer = qa_series["Answer"]
            response = self._get_api_response(question, provider, model)
            sim_score = self._similarity_score(response, answer)
            f1_score = self._f1_score(response, answer)

            result_dict["query_id"].append(f"qa_pool{indx}")
            result_dict["question"].append(question)
            result_dict[answer_id].append(answer)
            result_dict[f"{provider_model}_response"].append(response)
            result_dict[f"{provider_model}_similiarity-score"].append(sim_score)
            result_dict[f"{provider_model}_f1-score"].append(f1_score)
            result_dict["query_source"].append(qa_series["doc_source"])
            result_dict["doc_page"].append(qa_series["doc_page"])
            result_dict["doc_title"].append(qa_series["doc_title"])

        return result_dict

    def _evaluate_model(self, provider_and_model, answer_id):
        """Evaluate selected provider + model using groundtruth."""
        print(f"Model evaluation for {provider_and_model}")

        result_dict = defaultdict(list)

        result_dict = self._populate_eval_data_for_qna_json(
            provider_and_model, answer_id, result_dict
        )

        if self._args.qna_pool_file is not None:
            result_dict = self._populate_eval_data_for_qna_parquet(
                provider_and_model, answer_id, result_dict
            )

        return result_dict

    def evaluate_models(self):
        """Evaluate models against groundtruth answer."""
        print("Running model evaluation using groundtruth...")

        answer_id = "ground_truth+with_rag"
        common_cols = [
            "query_id",
            "question",
            answer_id,
            "query_source",
            "doc_page",
            "doc_title",
        ]
        result_df = DataFrame(columns=common_cols)

        provider_model_ids = self._args.eval_provider_model_id
        if not provider_model_ids:
            provider_model_ids = INSCOPE_MODELS.keys()

        for provider_model_id in provider_model_ids:
            provider_model = INSCOPE_MODELS[provider_model_id]
            result_dict = self._evaluate_model(provider_model, answer_id)

            model_result_df = DataFrame.from_dict(result_dict)
            result_df = result_df.merge(model_result_df, on=common_cols, how="outer")

        result_df.to_csv(f"{self._result_dir}/model_evaluation_result.csv", index=False)

        sim_score_summary = {}
        f1_score_summary = {}
        for p_m_id in provider_model_ids:
            provider_model = "+".join(INSCOPE_MODELS[p_m_id])
            sim_score_summary[provider_model] = result_df[
                f"{provider_model}_similiarity-score"
            ].mean()
            f1_score_summary[provider_model] = result_df[
                f"{provider_model}_f1-score"
            ].quantile(0.5)

        # TODO: Better structure for summary json file
        summary_dict = {
            "timestamp": str(datetime.now(UTC)),
            "similarity_score-avg": sim_score_summary,
            "f1_score-p50": f1_score_summary,
            "eval_set": result_df.query_id.unique().tolist(),
        }
        # Save model evaluation summary report
        with open(f"{self._result_dir}/model_evaluation_summary.json", "w") as f:
            json.dump(summary_dict, f)
