# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import requests
import sys
import os
import time
import pandas as pd

from urllib.parse import urlencode, urljoin
from tqdm import tqdm
from pyfiglet import Figlet
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import PyPDF2

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Tuple

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import AnswerAccuracy, ContextRelevance, ResponseGroundedness
from ragas.llms import LangchainLLMWrapper

CORPUS_DIRECTORY = "corpus"
EVAL_DATA = "train.json"
TIMEOUT = 180
VERSION = "v1"

DEFAULT_BATCH_SIZE = 1000  # 10000 is the max batch size for the ingestion server

NV_METRIC_NV_ACCURACY = "nv_accuracy"
NV_METRIC_NV_CONTEXT_RELEVANCE = "nv_context_relevance"
NV_METRIC_NV_RESPONSE_GROUNDEDNESS = "nv_response_groundedness"


_DEFAULT_JUDGE_MODEL = "mistralai/mixtral-8x22b-instruct-v0.1"
_JUDGE_MODEL_ENV = "RAG_EVAL_JUDGE_MODEL"
_judge_raw = (os.environ.get(_JUDGE_MODEL_ENV) or "").strip()
JUDGE_MODEL = _judge_raw if _judge_raw else _DEFAULT_JUDGE_MODEL


class IngestionMetrics(BaseModel):
    ingestion_time: float = Field(description="Time taken to ingest the documents", default=0.0)
    total_pages: int = Field(description="Total pages ingested", default=0)
    pages_per_second: float = Field(description="Pages per second", default=0.0)
    total_files: int = Field(description="Total files ingested", default=0)

class EvaluationMetrics(BaseModel):
    nv_accuracy: float = Field(description="RAGAS Answer Accuracy (nv_accuracy)", default=0.0)
    nv_context_relevance: float = Field(description="RAGAS Context Relevance (nv_context_relevance)", default=0.0)
    nv_response_groundedness: float = Field(description="RAGAS Response Groundedness (nv_response_groundedness)", default=0.0)


class TokenUsageMetrics(BaseModel):
    """Token usage KPI from RAG generate API"""
    total_tokens: int = Field(description="Total tokens (prompt + completion)", default=0)
    prompt_tokens: int = Field(description="Prompt/input tokens", default=0)
    completion_tokens: int = Field(description="Completion/output tokens", default=0)
    sample_count: int = Field(description="Number of samples with usage data", default=0)
    mean_prompt_tokens: float = Field(description="Mean prompt tokens per query", default=0.0)
    mean_completion_tokens: float = Field(description="Mean completion tokens per query", default=0.0)


class RagEvaluationMetrics(BaseModel):
    ingestion_metrics_list: List[IngestionMetrics] = Field(description="List of ingestion metrics", default=[])
    evaluation_metrics: EvaluationMetrics = Field(description="Evaluation metrics", default=EvaluationMetrics())
    token_usage: TokenUsageMetrics = Field(description="Token usage KPI from RAG generate", default=TokenUsageMetrics())


class RAGClient:
    def __init__(
        self,
        host,
        port,
        ingestor_server_url,
        collection_name,
        max_worker,
        result_dir,
        skip_ingestion,
        skip_evaluation,
        top_k: int | None = None,
        llm_model: str | None = None,
        llm_endpoint: str | None = None,
        vdb_top_k: int | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        force_ingestion: bool = False,
        dataset_root: str = ".",
        run_label: str = "dataset",
        file_type: str = "pdf",
        timeout: int = TIMEOUT,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        enable_reranker: bool | None = None,
        enable_query_rewriting: bool | None = None,
    ):
        self.host = host
        self.port = port
        self.ingestor_server_url = ingestor_server_url
        self.collection_name = collection_name
        self.rag_server_url = f"http://{host}:{port}/{VERSION}"
        self.dataset_root = os.path.abspath(dataset_root)
        self.dataset_path = os.path.join(self.dataset_root, CORPUS_DIRECTORY)
        self.eval_data = os.path.join(self.dataset_root, EVAL_DATA)
        self.batch_size = batch_size
        self.top_k = top_k
        self.max_worker = max_worker
        self.result_dir = result_dir
        self.retries = 3
        self.verbose = True
        self.skip_ingestion = skip_ingestion
        self.skip_evaluation = skip_evaluation
        self.run_label = run_label
        self.file_type = file_type

        self.rag_evaluation_metrics = RagEvaluationMetrics()
        
        # RAG server config (omit unset keys from generate requests so the server uses its defaults)
        self.llm_config = {
            "model": None if llm_model is None else str(llm_model).strip() or None,
            "llm_endpoint": None if llm_endpoint is None else str(llm_endpoint).strip() or None,
        }
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.enable_reranker = enable_reranker
        self.enable_query_rewriting = enable_query_rewriting
        self.vdb_top_k = vdb_top_k
        self.timeout = timeout
        self.error_response = "Error from rag-server. Please check rag-server logs for more details."
        self.error_count = 0
        self.force_ingestion = force_ingestion
        self.error_lock = Lock()  # Add a lock for thread-safe error counting
        
        print(f"    - Dataset path: {self.dataset_path}")
        print(f"    - Evaluation data path: {self.eval_data}")

    def check_collection_exists(self):
        """Check if the collection exists in the vector database"""
        url = f"{self.ingestor_server_url}/collections"
        response = None
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            collections_data = response.json()
            if collections_data and "collections" in collections_data:
                for collection in collections_data["collections"]:
                    if collection["collection_name"] == self.collection_name:
                        return True
            return False
        except requests.exceptions.RequestException as e:
            print(f"    - Error checking collection: {e}")
            if response is not None and hasattr(response, "status_code"):
                if response.status_code != 200:
                    print(f"    - Response Content: {response.text}") # print the response text for more debugging information
            return False
    
    def create_collection(self):
        """Create a collection in the vector database"""
        url = f"{self.ingestor_server_url}/collection"

        data = {
            "collection_name": self.collection_name,
            "metadata_schema": [],
        }
        print("    - Creating collection (no custom metadata schema)")

        response = None
        try:
            print(f"    - Creating collection {self.collection_name}")
            response = requests.post(url, json=data, timeout=self.timeout)
            print("    - Creating collection Response: ", response.json())
            response.raise_for_status()  # Raise an exception for HTTP errors
            return response.json()  # Return response JSON if successful
        except requests.exceptions.RequestException as e:
            print(f"    - Error creating collection: {e}")
            if response is not None:
                print(f"    - Response Code: {response.status_code} | Response Content: {response.text}")
            return None
    
    def get_ingested_documents(self):
        """Get list of ingested documents to avoid reingestion."""
        url = f"{self.ingestor_server_url}/documents"
        params = {
            "collection_name": self.collection_name,
        }
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            documents_data = response.json()
            if documents_data and "documents" in documents_data:
                return [doc.get("document_name") for doc in documents_data.get("documents")]
            return []  # Return empty list if no documents found
        except requests.exceptions.RequestException as e:
            print(f"    - Error getting ingested documents: {e}")
            if response is not None:
                print(f"    - Response Code: {response.status_code} | Response Content: {response.text}")
            return []  # Return empty list in case of error

    def bulk_upload(self, file_paths):
        """Bulk uploads files to the RAG ingestion server."""
        url = f"{self.ingestor_server_url}/documents"
        files = []
        file_objects = []  # Keep track of file objects

        # Get number of pages for pdf files
        if "pdf" in self.file_type:
            num_pages = self.get_number_of_pages_pdf(file_paths)
            print(f"    - Number of pages in {len(file_paths)} files: {num_pages}")

        for file_path in file_paths:
            try:
                f = open(file_path, "rb")  # Keep file handle open
                file_objects.append(f)  # Store reference to close later
                files.append(("documents", (os.path.basename(file_path), f)))
            except FileNotFoundError:
                print(f"File not found: {file_path}")
                return None
            except Exception as e:
                print(f"Error opening file {file_path}: {e}")
                return None

        # Omit split_options so the ingestor uses its configured defaults (deployed RAG server).
        upload_data = {
            "collection_name": self.collection_name,
            "blocking": True,
        }

        data = {"data": json.dumps(upload_data)}

        try:
            ingestion_start_time = time.time()
            response = requests.post(url, files=files, data=data, timeout=self.timeout)
            response.raise_for_status()
            ingestion_time = time.time() - ingestion_start_time
            if "pdf" in self.file_type:
                pages_per_second = num_pages / ingestion_time

            print("\n" + "="*80)
            print("INGESTION PERFORMANCE METRICS:")
            print("="*80)
            print(f"    - Number of files:                    {len(file_paths)}")
            print(f"    - Ingestion time for files:           {ingestion_time} seconds")
            if "pdf" in self.file_type:
                print(f"    - Total number of pages in files:     {num_pages}")
                print(f"    - Pages per second:                   {pages_per_second}")
            else:
                num_pages = 0
                pages_per_second = 0
                print(f"    - Total number of files:              {len(file_paths)}")
                print(f"    - Files per second:                   {len(file_paths) / ingestion_time}")
            print("-"*80, end="\n\n")
            # Add ingestion metrics to the rag evaluation metrics
            self.rag_evaluation_metrics.ingestion_metrics_list.append(IngestionMetrics(
                ingestion_time=ingestion_time,
                total_pages=num_pages,
                pages_per_second=pages_per_second,
                total_files=len(file_paths)
            ))
            return response.json()
        except Exception as e:
            print(f"Error uploading documents: {e}")
            if response is not None:
                print(f"Response Code: {response.status_code} | Response Content: {response.text}")
            return None
        finally:
            # Close all file handles
            for f in file_objects:
                f.close()

    def get_number_of_pages_pdf(self, file_paths):
        """Get the number of pages in a PDF file."""
        print(f"    - Getting number of pages for {len(file_paths)} files")
        num_pages = 0
        for file_path in tqdm(file_paths):
            try:
                if file_path.endswith(".pdf"):
                    with open(file_path, 'rb') as file:
                        reader = PyPDF2.PdfReader(file)
                        num_pages += len(reader.pages)
            except Exception as e:
                print(f"    - Error getting number of pages for {file_path}: {e}")
        return num_pages

    def upload_documents(self, files_to_upload, max_retries=3):
        """Upload documents to the RAG ingestion server with retry on failure.

        Args:
            files_to_upload: List of file paths to upload.
            max_retries: Optional max number of attempts per batch (default 3).
        """
        total_files = len(files_to_upload)
        successful_files = []
        failed_files = []
        total_batches = (total_files + self.batch_size - 1) // self.batch_size

        for i in range(0, total_files, self.batch_size):
            batch = files_to_upload[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            print(f"\n=== Uploading batch {batch_num} of {total_batches} for {self.collection_name} ===")

            upload_ok = False
            for attempt in range(1, max_retries + 1):
                if self.bulk_upload(batch):
                    upload_ok = True
                    break
                if attempt < max_retries:
                    print(f"\n    - Ingestion attempt {attempt}/{max_retries} failed for batch {batch_num}; retrying...")
                    time.sleep(2)  # Brief delay before retry

            if upload_ok:
                successful_files.extend(batch)
                print(f"\n✅ Successfully uploaded batch {batch_num}:")
            else:
                failed_files.extend(batch)
                print(f"\n❌ Failed to upload batch {batch_num} after {max_retries} attempts:")
                for file in batch:
                    print(f"    - {os.path.basename(file)}")

        print(f"\n📊 Upload Summary:")
        print(f"    - Total files: {total_files}")
        print(f"    - Successfully uploaded: {len(successful_files)}")
        print(f"    - Failed uploads: {len(failed_files)}")

        if failed_files:
            print("\n❌ List of failed files:")
            for file in failed_files:
                print(f"    - {os.path.basename(file)}")

    def _corpus_file_count(self) -> int:
        """Count regular files under corpus/ (recursive)."""
        n = 0
        for _, _, files in os.walk(self.dataset_path):
            n += len(files)
        return n

    def collect_files_to_upload(self, ingested_documents):
        """Collect all files that need to be uploaded by walking through dataset directory"""
        files_to_upload = []
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file not in ingested_documents:
                    file_path = os.path.join(root, file)
                    files_to_upload.append(file_path)
        return files_to_upload

    def get_eval_data(self):
        """Get evaluation data from the dataset"""
        with open(self.eval_data, 'r') as file:
            data = json.load(file)  # Load JSON data into a Python dictionary
        self.eval_data = data

    def create_eval_dict(self):
        """Create a evaulation dictionary with generated response"""
        eval_data = []
        self.get_eval_data()
        if not isinstance(self.eval_data, list):
            print(
                "Error: train.json must be a JSON array of objects with question/answer fields. "
            )
            sys.exit(1)
        total_questions = len(self.eval_data)  # Total number of queries to process
        
        # Define a worker function for parallel execution
        def process_query(d):
            try:
                generated_answer, generated_contexts, retrieved_docs, usage = self.get_rag_response(
                    query=d.get('question'),
                    collection_name=self.collection_name,
                    host=self.host,
                    port=self.port,
                    top_k=self.top_k
                )
                return {
                    'id': d.get('id') if 'id' in d else d.get('query_id') if 'query_id' in d else None,
                    'question': d.get('question'),
                    'answer': d.get('answer'),
                    'generated_answer': generated_answer,
                    "contexts": [],
                    "generated_contexts": generated_contexts,
                    "retrieved_docs": retrieved_docs,
                    "usage": usage,
                }
            except Exception as e:
                print(f"Error processing question {d.get('question')}: {e}")
                with self.error_lock:
                    self.error_count += 1
                return None

        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=self.max_worker) as executor:
            # Submit tasks for each data point
            futures = {executor.submit(process_query, d): d for d in self.eval_data}

            # tqdm progress bar
            with tqdm(total=len(self.eval_data), desc="Running Inference", unit="query") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        eval_data.append(result)
                    pbar.update(1)  # Update progress bar

        # Save all results to a file
        fname = os.path.join(self.result_dir, f"rag_{self.run_label}_evaluation_data.json")
        with open(fname, "w") as json_file:
            json.dump(eval_data, json_file, indent=4)

        # Check if more than 50% of queries failed
        if self.error_count > total_questions * 0.5:
            fail_pct = (self.error_count / total_questions) * 100
            print(
                f"⚠️ WARNING: High failure rate detected! "
                f"{self.error_count} failures out of {total_questions} queries ({fail_pct:.2f}%)."
            )

        return eval_data

    def get_rag_response(
        self,
        query: str,
        collection_name: str,
        host: str,
        port: int,
        top_k: Optional[int],
    ) -> Tuple[str, List[str], List[str], Optional[Dict[str, Any]]]:
        """Interact with rag server and get generate response and relevant docs."""
        # Request data to generate endpoint (only include optional fields when the client set them)
        data = {
            "messages": [
                {
                "role": "user",
                "content": query
                }
            ],
            "collection_names": [collection_name],
            }
        if top_k is not None:
            data["reranker_top_k"] = top_k
        if self.vdb_top_k is not None:
            data["vdb_top_k"] = self.vdb_top_k
        if self.llm_config.get("model"):
            data["model"] = self.llm_config["model"]
        if self.llm_config.get("llm_endpoint"):
            data["llm_endpoint"] = self.llm_config["llm_endpoint"]
        if self.temperature is not None:
            data["temperature"] = self.temperature
        if self.top_p is not None:
            data["top_p"] = self.top_p
        if self.max_tokens is not None:
            data["max_tokens"] = self.max_tokens
        if self.enable_reranker is not None:
            data["enable_reranker"] = self.enable_reranker
        if self.enable_query_rewriting is not None:
            data["enable_query_rewriting"] = self.enable_query_rewriting

        try:
            resp = ""
            docs = []
            filenames = []
            usage = None  # Token usage from last chunk (total_tokens, prompt_tokens, completion_tokens)
            url_generate = f"http://{host}:{port}/v1/generate"

            with requests.post(url_generate, stream=True, json=data, timeout=self.timeout) as req:
                req.raise_for_status()

                is_first_token = True
                for chunk in req.iter_lines():
                    if chunk:  # Check if the chunk is not empty
                        try:
                            raw_resp = chunk.decode("UTF-8")
                            resp_dict = json.loads(raw_resp[6:]) if raw_resp.startswith("data:") else json.loads(raw_resp) # Handle both data: and direct json responses
                            # Capture usage from chunk when present (last chunk typically has it)
                            if "usage" in resp_dict:
                                raw_usage = resp_dict.get("usage")
                                if isinstance(raw_usage, dict) and raw_usage:
                                    token_fields = ("total_tokens", "prompt_tokens", "completion_tokens")
                                    if any(raw_usage.get(k, 0) > 0 for k in token_fields):
                                        usage = raw_usage.copy()
                            resp_choices = resp_dict.get("choices", [])
                            if resp_choices:
                                resp_str = resp_choices[0].get("message", {}).get("content", "")
                                resp += resp_str

                                if is_first_token:
                                    for result in resp_dict.get("citations", {}).get("results", []):
                                        filename = result.get("document_name")
                                        page_num = result.get("metadata").get("page_number")
                                        if page_num != -1:
                                            filename += "_"+str(page_num)
                                        filenames.append(filename)
                                        description = result.get("metadata", {}).get("description")
                                        if description:
                                            docs.append(description)
                                    is_first_token = False

                        except json.JSONDecodeError as e:
                            print(f"Invalid JSON response: {chunk.decode('UTF-8') if chunk else 'Empty Chunk'}", e)
                            with self.error_lock:
                                self.error_count += 1
                            return "", [], [], None
                        except Exception as e:
                            print(f"An unexpected error occurred: {e}")
                            with self.error_lock:
                                self.error_count += 1
                            return "", [], [], None

            # Check if the response matches the error message
            if resp == self.error_response or self.error_response in resp:
                print(f"Response contained error message: {resp}")
                with self.error_lock:
                    self.error_count += 1

            return resp, docs, filenames, usage

        except requests.exceptions.RequestException as e:
            print(f"Failed to get response from rag-server. Error details: {e}")
            with self.error_lock:
                self.error_count += 1
            return "", [], [], None

        except Exception as e:
            print(f"A general error occurred: {e}")
            with self.error_lock:
                self.error_count += 1
            return "", [], [], None

    def run_pipeline(self):
        """Run the pipeline to get the evaluation results"""
        if self.force_ingestion:
            try:
                print(f"    - Force ingestion enabled, deleting collection {self.collection_name} if already exists")
                self.delete_collection(self.collection_name)
            except Exception as e:
                pass

        if not self.skip_ingestion:
            if self.check_collection_exists():
                print(f"    - Collection {self.collection_name} already exists")
            else:
                print(f"    - Collection {self.collection_name} does not exist, creating collection")
                if not self.create_collection():
                    print(f"    - Failed to create collection {self.collection_name}")
                    print(f"    - Aborting evaluation of {self.collection_name} dataset")
                    return

            ingested_documents =  self.get_ingested_documents()
            if ingested_documents:
                print(f"    - Number of documents already in {self.collection_name} collection is", len(ingested_documents))
            else:
                print(f"    - No documents found in {self.collection_name} collection")
            
            files_to_upload = self.collect_files_to_upload(ingested_documents)
            print("    - Number of files to upload: ", len(files_to_upload))
            self.upload_documents(files_to_upload)

            # Validate all documents are ingested
            self.validate_ingestion()

        if not self.skip_evaluation:
            ingested_documents =  self.get_ingested_documents()
            files_to_upload = self.collect_files_to_upload(ingested_documents)

            if len(files_to_upload) > 0:
                corpus_files = self._corpus_file_count()
                ingested_count = len(ingested_documents) if ingested_documents else 0
                count_gap = corpus_files - ingested_count
                docs_qs = urlencode({"collection_name": self.collection_name})
                docs_url = f"{self.ingestor_server_url}/documents?{docs_qs}"
                print(
                    f"    - WARNING: {corpus_files} corpus file(s) vs {ingested_count} ingested "
                    f"(gap {count_gap}); {len(files_to_upload)} unmatched. "
                    f"Check GET {docs_url}. Accuracy may be affected."
                )

            self.validate_ingestion()

            return self.create_eval_dict()

    def validate_ingestion(self):
        """Validate ingestion completed (no fixed catalog counts; corpus defines expected files)."""
        ingested_documents = self.get_ingested_documents()
        corpus_files = self._corpus_file_count()
        print(
            f"    - Corpus directory: {corpus_files} file(s) under {self.dataset_path}"
        )
        print(
            f"    - Ingestion check: {len(ingested_documents)} document(s) in collection "
            f"{self.collection_name}"
        )

    def delete_collection(self, collection_name):
        """Delete collection from ingestor server"""
        url = f"{self.ingestor_server_url}/collections"
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }
        data = [collection_name]
        response = requests.delete(url, headers=headers, json=data, timeout=self.timeout)
        if response.status_code == 200:
            print(f"    - Collection {collection_name} deleted successfully")
        else:
            print(f"    - Failed to delete collection {collection_name}, reason: {response.text}")

def validate_dataset_roots(dataset_roots: list[str]) -> None:
    """Ensure each path is a dataset root with corpus/ and train.json."""
    for root in dataset_roots:
        abs_root = os.path.abspath(root)
        if not os.path.isdir(abs_root):
            print(f"Error: dataset path is not a directory: {abs_root}")
            sys.exit(1)
        corpus_path = os.path.join(abs_root, CORPUS_DIRECTORY)
        eval_path = os.path.join(abs_root, EVAL_DATA)
        if not os.path.isdir(corpus_path):
            print(f"Error: missing corpus directory: {corpus_path}")
            sys.exit(1)
        if not os.path.isfile(eval_path):
            print(f"Error: missing {EVAL_DATA} under dataset root: {eval_path}")
            sys.exit(1)

def evaluate_result(
    dataset_root: str,
    top_k,
    eval_data,
):
    """
    Evaluate RAG outputs with RAGAS NVIDIA metrics (same classes as notebooks/evaluation_01_ragas.ipynb).
    """
    print("Starting RAG evaluation process...")
    print(
        f"    - Evaluation started for dataset_root={dataset_root}, top_k={top_k}"
    )

    llm = ChatNVIDIA(model=JUDGE_MODEL)

    all_results: dict = {}

    has_contexts = any(
        sample.get("generated_contexts") for sample in eval_data
    )

    if has_contexts:
        eval_dataset = EvaluationDataset([
            SingleTurnSample(
                user_input=sample["question"],
                reference=sample["answer"],
                response=sample["generated_answer"],
                reference_contexts=sample["contexts"],
                retrieved_contexts=sample["generated_contexts"],
            ) for sample in eval_data
        ])

        df = evaluate(
            dataset=eval_dataset,
            metrics=[
                AnswerAccuracy(),
                ContextRelevance(),
                ResponseGroundedness(),
            ],
            llm=LangchainLLMWrapper(llm),
        ).to_pandas()

        all_results = {
            NV_METRIC_NV_ACCURACY: df[NV_METRIC_NV_ACCURACY].tolist()
            if NV_METRIC_NV_ACCURACY in df.columns
            else [],
            NV_METRIC_NV_CONTEXT_RELEVANCE: df[NV_METRIC_NV_CONTEXT_RELEVANCE].tolist()
            if NV_METRIC_NV_CONTEXT_RELEVANCE in df.columns
            else [],
            NV_METRIC_NV_RESPONSE_GROUNDEDNESS: df[NV_METRIC_NV_RESPONSE_GROUNDEDNESS].tolist()
            if NV_METRIC_NV_RESPONSE_GROUNDEDNESS in df.columns
            else [],
        }
    else:
        print(
            "    - No retrieved contexts found; computing Answer Accuracy (nv_accuracy) only "
            f"(skipping {NV_METRIC_NV_CONTEXT_RELEVANCE}, {NV_METRIC_NV_RESPONSE_GROUNDEDNESS})"
        )
        eval_dataset = EvaluationDataset([
            SingleTurnSample(
                user_input=sample["question"],
                reference=sample["answer"],
                response=sample["generated_answer"],
                reference_contexts=sample.get("contexts", []),
                retrieved_contexts=[],
            ) for sample in eval_data
        ])

        df = evaluate(
            dataset=eval_dataset,
            metrics=[AnswerAccuracy()],
            llm=LangchainLLMWrapper(llm),
        ).to_pandas()

        all_results = {
            NV_METRIC_NV_ACCURACY: df[NV_METRIC_NV_ACCURACY].tolist()
            if NV_METRIC_NV_ACCURACY in df.columns
            else [],
            NV_METRIC_NV_CONTEXT_RELEVANCE: [],
            NV_METRIC_NV_RESPONSE_GROUNDEDNESS: [],
        }

    usages = [s.get("usage") for s in eval_data if s.get("usage") and isinstance(s["usage"], dict)]
    if usages:
        total_tokens = sum(u.get("total_tokens", 0) for u in usages)
        prompt_tokens = sum(u.get("prompt_tokens", 0) for u in usages)
        completion_tokens = sum(u.get("completion_tokens", 0) for u in usages)
        n = len(usages)
        all_results["token_usage"] = {
            "total_tokens": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "sample_count": n,
            "mean_prompt_tokens": prompt_tokens / n if n else 0.0,
            "mean_completion_tokens": completion_tokens / n if n else 0.0,
        }
        all_results["token_usage_per_sample"] = usages

    return all_results

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG against benchmark datasets (filesystem paths to dataset roots)."
    )
    parser.add_argument(
        "--dataset-paths",
        nargs="+",
        required=True,
        help="One or more dataset root directories, each containing a corpus/ folder and train.json.",
    )
    parser.add_argument(
        "--file-type",
        default="pdf",
        help='Ingestion file for metrics (e.g. "pdf", "txt", "txt,html", "mp3"). Substring "pdf" enables PDF page counts.',
    )
    parser.add_argument("--host", required=True, help="Host where the rag server is running.")
    parser.add_argument("--port", required=True, type=int, help="Port where the rag server is running.")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose output.")
    parser.add_argument("--thread", default=4, type=int, help="Number of thread to run for response generation.")
    parser.add_argument("--output_dir", default="results", help="File to store the evaluation results.")
    parser.add_argument("--batch_size", default=DEFAULT_BATCH_SIZE, type=int, help="Batch size for ingestion.")
    
    # RAG params (optional: omitted keys are not sent; RAG server uses its own defaults)
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Reranker / retrieval top-k (sent as reranker_top_k). If omitted, not sent to the RAG server.",
    )
    parser.add_argument(
        "--vdb_top_k",
        type=int,
        default=None,
        help="Vector DB candidate pool size (sent as vdb_top_k). If omitted, not sent to the RAG server.",
    )
    parser.add_argument("--ingestor_server_url", default="http://localhost:8082", help="Endpoint of the ingestor server in format http://ip:port.")
    parser.add_argument("--skip_ingestion", action='store_true', help="Skip ingestion of documents.")
    parser.add_argument("--skip_evaluation", action='store_true', help="Skip evaluation of results.")
    parser.add_argument("--delete_collection", action='store_true', help="Delete collection from ingestor server.")
    parser.add_argument("--force_ingestion", action='store_true', help="Remove collection from ingestor server and ingest again.")
    parser.add_argument("--collection", default=None, help="Override collection name to ingest the data.")
    
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model id for RAG generate. If omitted, not sent (server default).",
    )
    parser.add_argument(
        "--llm_endpoint",
        default=None,
        help="LLM API endpoint URL. If omitted, not sent (server default).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature for /v1/generate. If omitted, not sent (server default).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        dest="top_p",
        help="Top-p for /v1/generate. If omitted, not sent (server default).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        dest="max_tokens",
        help="Max tokens for /v1/generate. If omitted, not sent (server default).",
    )
    rerank_group = parser.add_mutually_exclusive_group()
    rerank_group.add_argument(
        "--enable-reranker",
        dest="enable_reranker",
        action="store_true",
        default=None,
        help="Send enable_reranker=true on /v1/generate.",
    )
    rerank_group.add_argument(
        "--disable-reranker",
        dest="enable_reranker",
        action="store_false",
        default=None,
        help="Send enable_reranker=false on /v1/generate.",
    )
    qr_group = parser.add_mutually_exclusive_group()
    qr_group.add_argument(
        "--enable-query-rewriting",
        dest="enable_query_rewriting",
        action="store_true",
        default=None,
        help="Send enable_query_rewriting=true on /v1/generate.",
    )
    qr_group.add_argument(
        "--disable-query-rewriting",
        dest="enable_query_rewriting",
        action="store_false",
        default=None,
        help="Send enable_query_rewriting=false on /v1/generate.",
    )
    parser.add_argument("--timeout", default=TIMEOUT, type=int, help="Timeout in seconds for RAG server requests (default: 180).")
    args = parser.parse_args()

    if "NVIDIA_API_KEY" not in os.environ:
        raise ValueError("NVIDIA_API_KEY environment variable is not set. Please set it before running this script.")

    validate_dataset_roots(list(args.dataset_paths))

    f = Figlet(font='slant')
    print(f.renderText('RAG Evaluation'))
    
    # Print the configuration
    print("\n" + "="*80)
    print("CONFIGURATION USED FOR EVALUATION")
    print("="*80)
    print(f"RAG Server IP:          {args.host}:{args.port}")
    print(f"Output directory:       {args.output_dir}")
    print(f"Dataset roots:          {args.dataset_paths}")
    print(f"File type:         {args.file_type}")
    print(f"Ingestor server URL:    {args.ingestor_server_url}")
    print(f"Skip ingestion:         {args.skip_ingestion}")
    print(f"Skip evaluation:        {args.skip_evaluation}")
    if args.model is not None:
        print(f"LLM Model:              {args.model}")
    if args.llm_endpoint is not None:
        print(f"LLM Endpoint:           {args.llm_endpoint}")
    print(f"RAGAS judge model:      {JUDGE_MODEL} (env {_JUDGE_MODEL_ENV})")
    print(f"Ingestion blocking:     True")
    print(f"Batch size:             {args.batch_size}")
    if args.top_k is not None:
        print(f"Top k (reranker_top_k): {args.top_k}")
    if args.vdb_top_k is not None:
        print(f"vdb_top_k:              {args.vdb_top_k}")
    if args.temperature is not None:
        print(f"Temperature:            {args.temperature}")
    if args.top_p is not None:
        print(f"top_p:                  {args.top_p}")
    if args.max_tokens is not None:
        print(f"max_tokens:             {args.max_tokens}")
    if args.enable_reranker is not None:
        print(f"enable_reranker:        {args.enable_reranker}")
    if args.enable_query_rewriting is not None:
        print(f"enable_query_rewriting: {args.enable_query_rewriting}")
    print(f"Force ingestion:        {args.force_ingestion}")
    print("RAG query endpoint:     /v1/generate")
    print(f"Request timeout:        {args.timeout}s")
    print("-"*80, end="\n\n")

    # Add version to ingestor server url
    args.ingestor_server_url = urljoin(args.ingestor_server_url, VERSION)

    # Evaluate each dataset root
    results = {}
    for dataset_root in args.dataset_paths:
        dataset_root = os.path.abspath(dataset_root)
        run_label = os.path.basename(dataset_root.rstrip(os.sep)) or "dataset"
        print(f"\n=== Evaluating {run_label} ({dataset_root}) ===")

        dataset_path = os.path.join(dataset_root, CORPUS_DIRECTORY)

        print(f"    - Path to corpus: {dataset_path}")
        collection_name = args.collection if args.collection else run_label
        print(f"    - Using collection name `{collection_name}` to ingest the data")

        output_dir = os.path.join(args.output_dir, run_label)
        # Create a directory to store the results
        os.makedirs(output_dir, exist_ok=True)

        rag_client = RAGClient(
            host=args.host,
            port=args.port,
            ingestor_server_url=args.ingestor_server_url,
            collection_name=collection_name,
            max_worker=args.thread,
            result_dir=output_dir,
            skip_ingestion=args.skip_ingestion,
            skip_evaluation=args.skip_evaluation,
            top_k=args.top_k,
            llm_model=args.model,
            llm_endpoint=args.llm_endpoint,
            vdb_top_k=args.vdb_top_k,
            batch_size=args.batch_size,
            force_ingestion=args.force_ingestion,
            dataset_root=dataset_root,
            run_label=run_label,
            file_type=args.file_type,
            timeout=args.timeout,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            enable_reranker=args.enable_reranker,
            enable_query_rewriting=args.enable_query_rewriting,
        )

        eval_data = rag_client.run_pipeline()

        if not args.skip_evaluation:
            all_result = evaluate_result(
                dataset_root,
                args.top_k,
                eval_data,
            )

            print("\n" + "="*80)
            print("EVALUATION RESULTS")
            print("="*80)
            # Helper to compute mean from lists safely
            def _mean(values):
                try:
                    if not values:
                        return 0.0
                    return float(pd.Series(values).mean())
                except (TypeError, ValueError):
                    return 0.0
            print(f"        - nv_accuracy (Answer Accuracy):       {_mean(all_result.get(NV_METRIC_NV_ACCURACY, []))}")
            if all_result.get(NV_METRIC_NV_CONTEXT_RELEVANCE):
                print(
                    f"        - nv_context_relevance:               {_mean(all_result.get(NV_METRIC_NV_CONTEXT_RELEVANCE, []))}"
                )
                print(
                    f"        - nv_response_groundedness:           {_mean(all_result.get(NV_METRIC_NV_RESPONSE_GROUNDEDNESS, []))}"
                )

            if isinstance(all_result, dict) and "token_usage" in all_result:
                tu = all_result["token_usage"]
                print("        - Token Usage")
                print(f"            -Total tokens:                  {tu.get('total_tokens', 0)}")
                print(f"            -Prompt tokens:                 {tu.get('prompt_tokens', 0)}")
                print(f"            -Completion tokens:             {tu.get('completion_tokens', 0)}")
                print(f"            -Samples with usage:            {tu.get('sample_count', 0)}")
                print(f"            -Mean prompt tokens/query:      {tu.get('mean_prompt_tokens', 0):.1f}")
                print(f"            -Mean completion tokens/query:  {tu.get('mean_completion_tokens', 0):.1f}")
            print("-"*80, end="\n\n")
            
            summary_metrics = {
                f"{NV_METRIC_NV_ACCURACY}_mean": _mean(all_result.get(NV_METRIC_NV_ACCURACY, [])),
            }
            if all_result.get(NV_METRIC_NV_CONTEXT_RELEVANCE):
                summary_metrics[f"{NV_METRIC_NV_CONTEXT_RELEVANCE}_mean"] = _mean(
                    all_result[NV_METRIC_NV_CONTEXT_RELEVANCE]
                )
                summary_metrics[f"{NV_METRIC_NV_RESPONSE_GROUNDEDNESS}_mean"] = _mean(
                    all_result[NV_METRIC_NV_RESPONSE_GROUNDEDNESS]
                )

            if isinstance(all_result, dict) and "token_usage" in all_result:
                summary_metrics["token_usage"] = all_result["token_usage"]
            results[run_label] = summary_metrics

            # Add evaluation metrics to the rag evaluation metrics
            if rag_client is not None:
                rag_client.rag_evaluation_metrics.evaluation_metrics = EvaluationMetrics(
                    nv_accuracy=_mean(all_result.get(NV_METRIC_NV_ACCURACY, [])),
                    nv_context_relevance=_mean(all_result.get(NV_METRIC_NV_CONTEXT_RELEVANCE, [])),
                    nv_response_groundedness=_mean(all_result.get(NV_METRIC_NV_RESPONSE_GROUNDEDNESS, [])),
                )

            if rag_client is not None and isinstance(all_result, dict) and "token_usage" in all_result:
                tu = all_result["token_usage"]
                rag_client.rag_evaluation_metrics.token_usage = TokenUsageMetrics(
                    total_tokens=tu.get("total_tokens", 0),
                    prompt_tokens=tu.get("prompt_tokens", 0),
                    completion_tokens=tu.get("completion_tokens", 0),
                    sample_count=tu.get("sample_count", 0),
                    mean_prompt_tokens=tu.get("mean_prompt_tokens", 0.0),
                    mean_completion_tokens=tu.get("mean_completion_tokens", 0.0),
                )

            fname = os.path.join(output_dir, f"rag_{run_label}_evaluation_summary.json")
            with open(fname, "w") as summary_file:
                json.dump(summary_metrics, summary_file, indent=4)

            fname = os.path.join(output_dir, f"rag_{run_label}_evaluation_results.json")
            # Save the nested results dictionary to a JSON file
            with open(fname, "w") as json_file:
                json.dump(all_result, json_file, indent=4)
    
        if rag_client is not None:
            fname = os.path.join(output_dir, f"rag_{run_label}_evaluation_metrics.json")
            with open(fname, "w") as json_file:
                json.dump(rag_client.rag_evaluation_metrics.model_dump(), json_file, indent=4)

        # Delete collection from ingestor server
        if args.delete_collection and rag_client is not None:
            rag_client.delete_collection(collection_name=collection_name)
        
    print(f"Evaluation complete. Results stored in directory: {args.output_dir}")
    for result in results:
        print(result, results[result])

if __name__ == "__main__":
    main()
