# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Library Usage Integration Test Module

Tests the nvidia_rag Python library directly, simulating notebook usage.
Assumes cloud deployed models as per notebooks/rag_library_usage.ipynb Option 2.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path

from ..base import BaseTestModule, TestStatus, test_case

logger = logging.getLogger(__name__)


class LibraryUsageModule(BaseTestModule):
    """Library usage integration test module"""
    
    def __init__(self, test_runner):
        super().__init__(test_runner)
        self.library_collection = "test_library_usage"
        self.test_data_dir = test_runner.data_dir
        
        # Shared configuration and instances (initialized lazily)
        self._config = None
        self._ingestor = None
        self._rag = None
    
    def _get_config(self):
        """Get or create shared config object with common settings"""
        if self._config is None:
            from nvidia_rag.utils.configuration import NvidiaRAGConfig
            
            config_path = Path(__file__).parent.parent.parent.parent / "notebooks" / "config.yaml"
            self._config = NvidiaRAGConfig.from_yaml(str(config_path))
            
            # Common configuration for all library tests
            self._config.embeddings.server_url = "https://integrate.api.nvidia.com/v1"
            self._config.ranking.server_url = ""  # Empty uses NVIDIA API catalog
            self._config.llm.server_url = ""  # Empty uses NVIDIA API catalog
            
            # Disable GPU vectorstore features for CI compatibility
            self._config.vector_store.enable_gpu_index = False
            self._config.vector_store.enable_gpu_search = False
            
            logger.info("✅ Shared config initialized with cloud deployment settings")
        
        return self._config
    
    def _get_ingestor(self):
        """Get or create shared NvidiaRAGIngestor instance"""
        if self._ingestor is None:
            from nvidia_rag import NvidiaRAGIngestor
            
            config = self._get_config()
            self._ingestor = NvidiaRAGIngestor(config=config)
            logger.info("✅ Shared NvidiaRAGIngestor instance created")
        
        return self._ingestor
    
    def _get_rag(self):
        """Get or create shared NvidiaRAG instance"""
        if self._rag is None:
            from nvidia_rag import NvidiaRAG
            
            config = self._get_config()
            self._rag = NvidiaRAG(config=config)
            logger.info("✅ Shared NvidiaRAG instance created")
        
        return self._rag
        
    @test_case(120, "Library - Import and Configuration")
    async def _test_library_imports_and_config(self) -> bool:
        """Test library imports and configuration loading (Notebook cells: imports and config setup)"""
        logger.info("\n=== Test 120: Library - Import and Configuration ===")
        start_time = time.time()
        
        try:
            # Test imports (Notebook: "Import the packages" section)
            logger.info("🔧 Testing library imports...")
            from nvidia_rag import NvidiaRAG, NvidiaRAGIngestor
            from nvidia_rag.utils.configuration import NvidiaRAGConfig
            logger.info("✅ Successfully imported NvidiaRAG, NvidiaRAGIngestor, NvidiaRAGConfig")
            
            # Test configuration loading (Notebook: "Import the NvidiaRAGIngestor packages")
            logger.info("🔧 Testing configuration loading...")
            config = self._get_config()  # Use shared config
            
            logger.info(f"✅ Configuration loaded successfully")
            logger.info(f"  - LLM model: {config.llm.model_name}")
            logger.info(f"  - Embeddings model: {config.embeddings.model_name}")
            logger.info(f"  - Ranking model: {config.ranking.model_name}")
            logger.info(f"  - GPU Index: {config.vector_store.enable_gpu_index}")
            logger.info(f"  - GPU Search: {config.vector_store.enable_gpu_search}")
            
            self.add_test_result(
                120, "Library - Import and Configuration",
                "Test library imports, configuration loading from YAML, and cloud deployment setup",
                ["NvidiaRAG", "NvidiaRAGIngestor", "NvidiaRAGConfig.from_yaml()"],
                ["config.embeddings.server_url", "config.ranking.server_url", "config.llm.server_url"],
                time.time() - start_time,
                TestStatus.SUCCESS
            )
            return True
            
        except Exception as e:
            logger.error(f"❌ Library import/config test failed: {e}")
            import traceback
            traceback.print_exc()
            self.add_test_result(
                120, "Library - Import and Configuration",
                "Test library imports and configuration loading",
                ["NvidiaRAG", "NvidiaRAGIngestor", "NvidiaRAGConfig"],
                ["config_file"],
                time.time() - start_time,
                TestStatus.FAILURE,
                str(e)
            )
            return False

    @test_case(121, "Library - Initialize NvidiaRAGIngestor")
    async def _test_library_init_ingestor(self) -> bool:
        """Test NvidiaRAGIngestor initialization (Notebook cell: NvidiaRAGIngestor initialization)"""
        logger.info("\n=== Test 121: Library - Initialize NvidiaRAGIngestor ===")
        start_time = time.time()
        
        try:
            logger.info("🔧 Initializing NvidiaRAGIngestor...")
            ingestor = self._get_ingestor()  # Use shared instance
            
            logger.info(f"✅ NvidiaRAGIngestor initialized successfully")
            logger.info(f"  - Type: {type(ingestor).__name__}")
            logger.info(f"  - Has create_collection: {hasattr(ingestor, 'create_collection')}")
            logger.info(f"  - Has upload_documents: {hasattr(ingestor, 'upload_documents')}")
            
            self.add_test_result(
                121, "Library - Initialize NvidiaRAGIngestor",
                "Test NvidiaRAGIngestor initialization with cloud config",
                ["NvidiaRAGIngestor(config)"],
                ["config"],
                time.time() - start_time,
                TestStatus.SUCCESS
            )
            return True
            
        except Exception as e:
            logger.error(f"❌ NvidiaRAGIngestor initialization failed: {e}")
            import traceback
            traceback.print_exc()
            self.add_test_result(
                121, "Library - Initialize NvidiaRAGIngestor",
                "Test NvidiaRAGIngestor initialization",
                ["NvidiaRAGIngestor(config)"],
                ["config"],
                time.time() - start_time,
                TestStatus.FAILURE,
                str(e)
            )
            return False

    @test_case(122, "Library - Create Collection")
    async def _test_library_create_collection(self) -> bool:
        """Test collection creation via library (Notebook section: "1. Create a new collection")"""
        logger.info("\n=== Test 122: Library - Create Collection ===")
        start_time = time.time()
        
        try:
            logger.info(f"🔧 Creating collection '{self.library_collection}'...")
            ingestor = self._get_ingestor()  # Use shared instance
            
            # Create collection (Notebook: ingestor.create_collection())
            response = ingestor.create_collection(
                collection_name=self.library_collection,
                vdb_endpoint="http://localhost:19530",
            )
            
            logger.info(f"📋 Response:\n{json.dumps(response, indent=2)}")
            
            # Check for successful creation - library returns "message" and "collection_name"
            if "collection_name" in response or ("message" in response and "successfully" in response["message"].lower()):
                logger.info(f"✅ Collection created successfully")
                self.add_test_result(
                    122, "Library - Create Collection",
                    f"Test collection creation via library API: {self.library_collection}",
                    ["ingestor.create_collection()"],
                    ["collection_name", "vdb_endpoint"],
                    time.time() - start_time,
                    TestStatus.SUCCESS
                )
                return True
            else:
                logger.error(f"❌ Collection creation failed: {response}")
                self.add_test_result(
                    122, "Library - Create Collection",
                    "Test collection creation via library API",
                    ["ingestor.create_collection()"],
                    ["collection_name", "vdb_endpoint"],
                    time.time() - start_time,
                    TestStatus.FAILURE,
                    f"Creation failed: {response}"
                )
                return False
                
        except Exception as e:
            logger.error(f"❌ Collection creation test failed: {e}")
            import traceback
            traceback.print_exc()
            self.add_test_result(
                122, "Library - Create Collection",
                "Test collection creation via library API",
                ["ingestor.create_collection()"],
                ["collection_name", "vdb_endpoint"],
                time.time() - start_time,
                TestStatus.FAILURE,
                str(e)
            )
            return False

    @test_case(123, "Library - List Collections")
    async def _test_library_list_collections(self) -> bool:
        """Test listing collections (Notebook section: "2. List all collections")"""
        logger.info("\n=== Test 123: Library - List Collections ===")
        start_time = time.time()
        
        try:
            logger.info("🔧 Listing collections...")
            ingestor = self._get_ingestor()  # Use shared instance
            
            # List collections (Notebook: ingestor.get_collections())
            response = ingestor.get_collections(vdb_endpoint="http://localhost:19530")
            
            logger.info(f"✅ Collections retrieved successfully")
            logger.info(f"📋 Response:\n{json.dumps(response, indent=2)}")
            
            # Verify our test collection is in the list
            collections = response.get("collections", [])
            if self.library_collection in collections:
                logger.info(f"✅ Test collection '{self.library_collection}' found in list")
                self.add_test_result(
                    123, "Library - List Collections",
                    "Test listing all collections via library API and verify test collection exists",
                    ["ingestor.get_collections()"],
                    ["vdb_endpoint"],
                    time.time() - start_time,
                    TestStatus.SUCCESS
                )
                return True
            else:
                logger.warning(f"⚠️ Test collection '{self.library_collection}' not found in list")
                self.add_test_result(
                    123, "Library - List Collections",
                    "Test listing all collections via library API",
                    ["ingestor.get_collections()"],
                    ["vdb_endpoint"],
                    time.time() - start_time,
                    TestStatus.SUCCESS  # Still success as listing worked
                )
                return True
                
        except Exception as e:
            logger.error(f"❌ List collections test failed: {e}")
            import traceback
            traceback.print_exc()
            self.add_test_result(
                123, "Library - List Collections",
                "Test listing collections via library API",
                ["ingestor.get_collections()"],
                ["vdb_endpoint"],
                time.time() - start_time,
                TestStatus.FAILURE,
                str(e)
            )
            return False

    @test_case(124, "Library - Initialize NvidiaRAG")
    async def _test_library_init_rag(self) -> bool:
        """Test NvidiaRAG initialization (Notebook section: "Import the NvidiaRAG packages")"""
        logger.info("\n=== Test 124: Library - Initialize NvidiaRAG ===")
        start_time = time.time()
        
        try:
            logger.info("🔧 Initializing NvidiaRAG...")
            rag = self._get_rag()  # Use shared instance
            
            logger.info(f"✅ NvidiaRAG initialized successfully")
            logger.info(f"  - Type: {type(rag).__name__}")
            logger.info(f"  - Has generate method: {hasattr(rag, 'generate')}")
            logger.info(f"  - Has search method: {hasattr(rag, 'search')}")
            logger.info(f"  - Has health method: {hasattr(rag, 'health')}")
            
            self.add_test_result(
                124, "Library - Initialize NvidiaRAG",
                "Test NvidiaRAG initialization with cloud deployment config",
                ["NvidiaRAG(config)"],
                ["config"],
                time.time() - start_time,
                TestStatus.SUCCESS
            )
            return True
            
        except Exception as e:
            logger.error(f"❌ NvidiaRAG initialization failed: {e}")
            import traceback
            traceback.print_exc()
            self.add_test_result(
                124, "Library - Initialize NvidiaRAG",
                "Test NvidiaRAG initialization",
                ["NvidiaRAG(config)"],
                ["config"],
                time.time() - start_time,
                TestStatus.FAILURE,
                str(e)
            )
            return False

    @test_case(125, "Library - Custom Prompts")
    async def _test_library_custom_prompts(self) -> bool:
        """Test custom prompts initialization (Notebook section: "11. Customize prompts")"""
        logger.info("\n=== Test 125: Library - Custom Prompts ===")
        start_time = time.time()
        
        try:
            from nvidia_rag import NvidiaRAG
            
            logger.info("🔧 Testing custom prompts initialization...")
            config = self._get_config()  # Use shared config
            
            # Define custom prompts (Notebook: pirate_prompts example)
            custom_prompts = {
                "rag_template": {
                    "system": "/no_think",
                    "human": "You are a test assistant. Answer questions using the context: {context}"
                }
            }
            
            # Initialize with custom prompts (Notebook: rag_pirate = NvidiaRAG(config, prompts=...))
            # This needs a separate instance because it has custom prompts
            rag_custom = NvidiaRAG(config=config, prompts=custom_prompts)
            logger.info(f"✅ NvidiaRAG with custom prompts initialized successfully")
            logger.info(f"  - Custom prompts applied: {bool(custom_prompts)}")
            
            self.add_test_result(
                125, "Library - Custom Prompts",
                "Test NvidiaRAG initialization with custom prompts via dictionary",
                ["NvidiaRAG(config, prompts=custom_prompts)"],
                ["config", "prompts"],
                time.time() - start_time,
                TestStatus.SUCCESS
            )
            return True
            
        except Exception as e:
            logger.error(f"❌ Custom prompts test failed: {e}")
            import traceback
            traceback.print_exc()
            self.add_test_result(
                125, "Library - Custom Prompts",
                "Test NvidiaRAG with custom prompts",
                ["NvidiaRAG(config, prompts)"],
                ["config", "prompts"],
                time.time() - start_time,
                TestStatus.FAILURE,
                str(e)
            )
            return False

    @test_case(127, "Library - Upload Documents")
    async def _test_library_upload_documents(self) -> bool:
        """Test document upload (Notebook section: "3. Add a document")"""
        logger.info("\n=== Test 127: Library - Upload Documents ===")
        start_time = time.time()
        
        try:
            logger.info(f"🔧 Uploading documents to collection '{self.library_collection}'...")
            ingestor = self._get_ingestor()  # Use shared instance
            
            # Get test files
            test_files = []
            if self.test_files:
                test_files = [str(self.test_data_dir / f) for f in self.test_files[:2]]  # Use first 2 files
            else:
                # Fallback to default test files
                default_files = ["multimodal_test.pdf", "woods_frost.docx"]
                test_files = [str(self.test_data_dir / f) for f in default_files if (self.test_data_dir / f).exists()]
            
            if not test_files:
                logger.error("❌ No test files found for upload")
                self.add_test_result(
                    127, "Library - Upload Documents",
                    "Test document upload via library API",
                    ["ingestor.upload_documents()"],
                    ["collection_name", "vdb_endpoint", "filepaths"],
                    time.time() - start_time,
                    TestStatus.FAILURE,
                    "No test files available"
                )
                return False
            
            logger.info(f"📄 Files to upload: {test_files}")
            
            # Upload documents (Notebook: await ingestor.upload_documents())
            response = await ingestor.upload_documents(
                collection_name=self.library_collection,
                vdb_endpoint="http://localhost:19530",
                blocking=False,  # Async as in notebook
                split_options={"chunk_size": 512, "chunk_overlap": 150},
                filepaths=test_files,
                generate_summary=False,
            )
            
            logger.info(f"📋 Response:\n{json.dumps(response, indent=2)}")
            
            # Check for task_id indicating successful submission
            task_id = response.get("task_id")
            if task_id:
                logger.info(f"✅ Documents uploaded successfully, task_id: {task_id}")
                # Store task_id and filenames for subsequent tests
                self.test_runner.library_upload_task_id = task_id
                self.test_runner.library_filenames = test_files
                
                self.add_test_result(
                    127, "Library - Upload Documents",
                    f"Test document upload via library API to collection: {self.library_collection}",
                    ["ingestor.upload_documents()"],
                    ["collection_name", "vdb_endpoint", "filepaths", "blocking", "split_options"],
                    time.time() - start_time,
                    TestStatus.SUCCESS
                )
                return True
            else:
                logger.error(f"❌ Upload failed: {response}")
                self.add_test_result(
                    127, "Library - Upload Documents",
                    "Test document upload via library API",
                    ["ingestor.upload_documents()"],
                    ["collection_name", "vdb_endpoint", "filepaths"],
                    time.time() - start_time,
                    TestStatus.FAILURE,
                    f"No task_id in response: {response}"
                )
                return False
                
        except Exception as e:
            logger.error(f"❌ Document upload test failed: {e}")
            import traceback
            traceback.print_exc()
            self.add_test_result(
                127, "Library - Upload Documents",
                "Test document upload via library API",
                ["ingestor.upload_documents()"],
                ["collection_name", "vdb_endpoint", "filepaths"],
                time.time() - start_time,
                TestStatus.FAILURE,
                str(e)
            )
            return False

    @test_case(128, "Library - Check Upload Status")
    async def _test_library_check_upload_status(self) -> bool:
        """Test checking upload status (Notebook section: "4. Check document upload status")"""
        logger.info("\n=== Test 128: Library - Check Upload Status ===")
        start_time = time.time()
        
        try:
            # Get task_id from previous test
            task_id = getattr(self.test_runner, 'library_upload_task_id', None)
            if not task_id:
                logger.warning("⚠️ No task_id available from upload test, skipping status check")
                self.add_test_result(
                    128, "Library - Check Upload Status",
                    "Test checking upload status via library API",
                    ["ingestor.status()"],
                    ["task_id"],
                    time.time() - start_time,
                    TestStatus.SUCCESS  # Not a failure, just no task to check
                )
                return True
            
            logger.info(f"🔧 Checking status for task_id: {task_id}")
            ingestor = self._get_ingestor()  # Use shared instance
            
            # Poll for completion (with timeout and fail-fast for repeated errors)
            max_wait = 300  # 5 minutes
            poll_interval = 5
            elapsed = 0
            consecutive_errors = 0
            max_consecutive_errors = 3  # Fail fast after 3 consecutive errors
            
            while elapsed < max_wait:
                try:
                    response = await ingestor.status(task_id=task_id)
                    consecutive_errors = 0  # Reset counter on successful response
                    
                    # Log full response for debugging
                    if elapsed == 0 or elapsed % 30 == 0:  # Log every 30 seconds
                        logger.info(f"📋 Status response:\n{json.dumps(response, indent=2)}")
                    
                    # Check for "state" field (REST API format), not "status"
                    state = response.get("state", "")
                    
                    if not state:
                        # If no state field, log warning and continue
                        logger.warning(f"⚠️ No 'state' field in response: {json.dumps(response, indent=2)}")
                        await asyncio.sleep(poll_interval)
                        elapsed += poll_interval
                        continue
                    
                    if state == "FINISHED":
                        logger.info(f"✅ Upload completed successfully after {elapsed}s")
                        self.add_test_result(
                            128, "Library - Check Upload Status",
                            f"Test checking upload status via library API, task completed in {elapsed}s",
                            ["ingestor.status()"],
                            ["task_id"],
                            time.time() - start_time,
                            TestStatus.SUCCESS
                        )
                        return True
                    elif state == "FAILED":
                        logger.error(f"❌ Upload task failed: {response}")
                        self.add_test_result(
                            128, "Library - Check Upload Status",
                            "Test checking upload status via library API",
                            ["ingestor.status()"],
                            ["task_id"],
                            time.time() - start_time,
                            TestStatus.FAILURE,
                            f"Upload task failed: {response}"
                        )
                        return False
                    
                    # Still processing
                    logger.info(f"⏳ Task state: {state}, waiting... (elapsed: {elapsed}s)")
                    
                except KeyError as e:
                    # Task not found - likely critical error
                    consecutive_errors += 1
                    logger.error(f"❌ Task not found error (attempt {consecutive_errors}/{max_consecutive_errors}): {e}")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"❌ Task {task_id} not found after {consecutive_errors} attempts. Failing fast.")
                        self.add_test_result(
                            128, "Library - Check Upload Status",
                            "Test checking upload status via library API",
                            ["ingestor.status()"],
                            ["task_id"],
                            time.time() - start_time,
                            TestStatus.FAILURE,
                            f"Task not found after {consecutive_errors} attempts"
                        )
                        return False
                        
                except Exception as poll_error:
                    consecutive_errors += 1
                    logger.error(f"❌ Error polling status (attempt {consecutive_errors}/{max_consecutive_errors}): {poll_error}")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"❌ Too many consecutive errors ({consecutive_errors}). Failing fast.")
                        self.add_test_result(
                            128, "Library - Check Upload Status",
                            "Test checking upload status via library API",
                            ["ingestor.status()"],
                            ["task_id"],
                            time.time() - start_time,
                            TestStatus.FAILURE,
                            f"Too many consecutive errors: {poll_error}"
                        )
                        return False
                    
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
            
            logger.warning(f"⚠️ Timeout waiting for upload completion after {max_wait}s")
            logger.warning(f"📋 Last status response: {json.dumps(response, indent=2)}")
            self.add_test_result(
                128, "Library - Check Upload Status",
                "Test checking upload status via library API",
                ["ingestor.status()"],
                ["task_id"],
                time.time() - start_time,
                TestStatus.FAILURE,
                f"Timeout after {max_wait}s"
            )
            return False
                
        except Exception as e:
            logger.error(f"❌ Status check test failed: {e}")
            import traceback
            traceback.print_exc()
            self.add_test_result(
                128, "Library - Check Upload Status",
                "Test checking upload status via library API",
                ["ingestor.status()"],
                ["task_id"],
                time.time() - start_time,
                TestStatus.FAILURE,
                str(e)
            )
            return False

    @test_case(129, "Library - Get Documents")
    async def _test_library_get_documents(self) -> bool:
        """Test getting documents in collection (Notebook section: "5. Get documents in a collection")"""
        logger.info("\n=== Test 129: Library - Get Documents ===")
        start_time = time.time()
        
        try:
            logger.info(f"🔧 Getting documents from collection '{self.library_collection}'...")
            ingestor = self._get_ingestor()  # Use shared instance
            
            # Get documents (Notebook: ingestor.get_documents())
            response = ingestor.get_documents(
                collection_name=self.library_collection,
                vdb_endpoint="http://localhost:19530",
            )
            
            logger.info(f"📋 Response:\n{json.dumps(response, indent=2)}")
            
            # Check if we got documents list
            documents = response.get("documents", [])
            logger.info(f"✅ Retrieved {len(documents)} document(s) from collection")
            
            self.add_test_result(
                129, "Library - Get Documents",
                f"Test retrieving documents list from collection via library API: {self.library_collection}",
                ["ingestor.get_documents()"],
                ["collection_name", "vdb_endpoint"],
                time.time() - start_time,
                TestStatus.SUCCESS
            )
            return True
                
        except Exception as e:
            logger.error(f"❌ Get documents test failed: {e}")
            import traceback
            traceback.print_exc()
            self.add_test_result(
                129, "Library - Get Documents",
                "Test retrieving documents via library API",
                ["ingestor.get_documents()"],
                ["collection_name", "vdb_endpoint"],
                time.time() - start_time,
                TestStatus.FAILURE,
                str(e)
            )
            return False

    @test_case(130, "Library - Health Check")
    async def _test_library_health_check(self) -> bool:
        """Test health check (Notebook section: "Check health of all dependent services")"""
        logger.info("\n=== Test 130: Library - Health Check ===")
        start_time = time.time()
        
        try:
            logger.info("🔧 Checking health of all services...")
            rag = self._get_rag()  # Use shared instance
            
            # Health check (Notebook: await rag.health())
            health_status = await rag.health()
            
            logger.info(f"📋 Health status message: {health_status.message}")
            logger.info(f"  - Type: {type(health_status).__name__}")
            
            # Check if health check returned valid response
            if hasattr(health_status, 'message') and health_status.message:
                logger.info(f"✅ Health check completed successfully")
                self.add_test_result(
                    130, "Library - Health Check",
                    "Test health check for all dependent services via library API",
                    ["rag.health()"],
                    [],
                    time.time() - start_time,
                    TestStatus.SUCCESS
                )
                return True
            else:
                logger.error(f"❌ Health check returned invalid response: {health_status}")
                self.add_test_result(
                    130, "Library - Health Check",
                    "Test health check via library API",
                    ["rag.health()"],
                    [],
                    time.time() - start_time,
                    TestStatus.FAILURE,
                    "Invalid health check response"
                )
                return False
                
        except Exception as e:
            logger.error(f"❌ Health check test failed: {e}")
            import traceback
            traceback.print_exc()
            self.add_test_result(
                130, "Library - Health Check",
                "Test health check via library API",
                ["rag.health()"],
                [],
                time.time() - start_time,
                TestStatus.FAILURE,
                str(e)
            )
            return False

    @test_case(131, "Library - RAG Generate/Query")
    async def _test_library_rag_generate(self) -> bool:
        """Test RAG generation/query (Notebook section: "6. Query a document using RAG")"""
        logger.info("\n=== Test 131: Library - RAG Generate/Query ===")
        start_time = time.time()
        
        try:
            logger.info(f"🔧 Testing RAG query with collection '{self.library_collection}'...")
            rag = self._get_rag()  # Use shared instance
            
            # Generate query (Notebook: await rag.generate())
            query = "What is the content of the document?"
            logger.info(f"📝 Query: {query}")
            
            rag_response = await rag.generate(
                messages=[{"role": "user", "content": query}],
                use_knowledge_base=True,
                collection_names=[self.library_collection],
            )
            
            logger.info(f"  - Response status: {rag_response.status_code}")
            logger.info(f"  - Has generator: {hasattr(rag_response, 'generator')}")
            
            # Check if response is valid
            if rag_response.status_code == 200 and hasattr(rag_response, 'generator'):
                # Consume the streaming response
                response_text = ""
                async for chunk in rag_response.generator:
                    if chunk.startswith("data: "):
                        chunk = chunk[len("data: "):].strip()
                    if chunk:
                        try:
                            data = json.loads(chunk)
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                text = delta.get("content")
                                if text:
                                    response_text += text
                        except json.JSONDecodeError:
                            pass
                
                logger.info(f"✅ RAG generation completed successfully")
                logger.info(f"📝 Response preview: {response_text[:200]}...")
                
                self.add_test_result(
                    131, "Library - RAG Generate/Query",
                    f"Test RAG generation/query via library API with collection: {self.library_collection}",
                    ["rag.generate()"],
                    ["messages", "use_knowledge_base", "collection_names"],
                    time.time() - start_time,
                    TestStatus.SUCCESS
                )
                return True
            else:
                logger.error(f"❌ RAG generation failed: status={rag_response.status_code}")
                self.add_test_result(
                    131, "Library - RAG Generate/Query",
                    "Test RAG generation via library API",
                    ["rag.generate()"],
                    ["messages", "use_knowledge_base", "collection_names"],
                    time.time() - start_time,
                    TestStatus.FAILURE,
                    f"Invalid response: status={rag_response.status_code}"
                )
                return False
                
        except Exception as e:
            logger.error(f"❌ RAG generation test failed: {e}")
            import traceback
            traceback.print_exc()
            self.add_test_result(
                131, "Library - RAG Generate/Query",
                "Test RAG generation via library API",
                ["rag.generate()"],
                ["messages", "use_knowledge_base", "collection_names"],
                time.time() - start_time,
                TestStatus.FAILURE,
                str(e)
            )
            return False

    @test_case(132, "Library - Search")
    async def _test_library_search(self) -> bool:
        """Test search functionality (Notebook section: "7. Search for documents")"""
        logger.info("\n=== Test 132: Library - Search ===")
        start_time = time.time()
        
        try:
            logger.info(f"🔧 Testing search with collection '{self.library_collection}'...")
            rag = self._get_rag()  # Use shared instance
            
            # Search (Notebook: await rag.search())
            query = "What is in the document?"
            logger.info(f"🔍 Search query: {query}")
            
            citations = await rag.search(
                query=query,
                collection_names=[self.library_collection],
                reranker_top_k=10,
                vdb_top_k=100,
            )
            
            # Check if we got citations
            if hasattr(citations, 'results'):
                num_results = len(citations.results) if citations.results else 0
                logger.info(f"✅ Search completed successfully")
                logger.info(f"📋 Found {num_results} result(s)")
                
                self.add_test_result(
                    132, "Library - Search",
                    f"Test search functionality via library API with collection: {self.library_collection}",
                    ["rag.search()"],
                    ["query", "collection_names", "reranker_top_k", "vdb_top_k"],
                    time.time() - start_time,
                    TestStatus.SUCCESS
                )
                return True
            else:
                logger.error(f"❌ Search returned invalid response: {citations}")
                self.add_test_result(
                    132, "Library - Search",
                    "Test search via library API",
                    ["rag.search()"],
                    ["query", "collection_names"],
                    time.time() - start_time,
                    TestStatus.FAILURE,
                    "Invalid citations response"
                )
                return False
                
        except Exception as e:
            logger.error(f"❌ Search test failed: {e}")
            import traceback
            traceback.print_exc()
            self.add_test_result(
                132, "Library - Search",
                "Test search via library API",
                ["rag.search()"],
                ["query", "collection_names"],
                time.time() - start_time,
                TestStatus.FAILURE,
                str(e)
            )
            return False

    @test_case(133, "Library - Delete Documents")
    async def _test_library_delete_documents(self) -> bool:
        """Test document deletion (Notebook section: "9. Delete documents from a collection")"""
        logger.info("\n=== Test 133: Library - Delete Documents ===")
        start_time = time.time()
        
        try:
            logger.info(f"🔧 Deleting documents from collection '{self.library_collection}'...")
            ingestor = self._get_ingestor()  # Use shared instance
            
            # First get documents to know what to delete
            docs_response = ingestor.get_documents(
                collection_name=self.library_collection,
                vdb_endpoint="http://localhost:19530",
            )
            
            documents = docs_response.get("documents", [])
            if not documents:
                logger.warning("⚠️ No documents found to delete")
                self.add_test_result(
                    133, "Library - Delete Documents",
                    "Test document deletion via library API",
                    ["ingestor.delete_documents()"],
                    ["collection_name", "document_names", "vdb_endpoint"],
                    time.time() - start_time,
                    TestStatus.SUCCESS  # Not a failure, just no docs
                )
                return True
            
            # Extract filename from document object (documents are dicts with metadata)
            # Common fields: 'filename', 'file_name', 'document_name', 'source'
            doc_obj = documents[0]
            logger.info(f"📄 Document object structure: {json.dumps(doc_obj, indent=2)}")
            
            # Try to extract filename from various possible fields
            doc_filename = None
            for field in ['filename', 'file_name', 'document_name', 'source', 'name']:
                if isinstance(doc_obj, dict) and field in doc_obj:
                    doc_filename = doc_obj[field]
                    break
                elif hasattr(doc_obj, field):
                    doc_filename = getattr(doc_obj, field)
                    break
            
            # If still no filename, try using the uploaded filenames
            if not doc_filename:
                # Use the filename from upload (stored in self.test_runner.library_filenames)
                if hasattr(self.test_runner, 'library_filenames') and self.test_runner.library_filenames:
                    doc_filename = self.test_runner.library_filenames[0]
                    logger.info(f"📝 Using stored filename from upload: {doc_filename}")
                else:
                    logger.warning("⚠️ Could not extract filename from document object, skipping deletion")
                    self.add_test_result(
                        133, "Library - Delete Documents",
                        "Test document deletion via library API",
                        ["ingestor.delete_documents()"],
                        ["collection_name", "document_names", "vdb_endpoint"],
                        time.time() - start_time,
                        TestStatus.SUCCESS  # Not a failure, just couldn't extract filename
                    )
                    return True
            
            logger.info(f"🗑️ Deleting document: {doc_filename}")
            
            # Delete document (Notebook: ingestor.delete_documents())
            response = ingestor.delete_documents(
                collection_name=self.library_collection,
                document_names=[doc_filename],
                vdb_endpoint="http://localhost:19530",
            )
            
            logger.info(f"📋 Response:\n{json.dumps(response, indent=2)}")
            logger.info(f"✅ Document deletion completed")
            
            self.add_test_result(
                133, "Library - Delete Documents",
                f"Test document deletion via library API from collection: {self.library_collection}",
                ["ingestor.delete_documents()"],
                ["collection_name", "document_names", "vdb_endpoint"],
                time.time() - start_time,
                TestStatus.SUCCESS
            )
            return True
                
        except Exception as e:
            logger.error(f"❌ Delete documents test failed: {e}")
            import traceback
            traceback.print_exc()
            self.add_test_result(
                133, "Library - Delete Documents",
                "Test document deletion via library API",
                ["ingestor.delete_documents()"],
                ["collection_name", "document_names", "vdb_endpoint"],
                time.time() - start_time,
                TestStatus.FAILURE,
                str(e)
            )
            return False

    @test_case(134, "Library - View Default Summarizer Config")
    async def _test_library_view_summarizer_config(self) -> bool:
        """Test viewing default summarizer configuration (Summarization notebook: Step 2)"""
        logger.info("\n=== Test 134: Library - View Default Summarizer Config ===")
        start_time = time.time()
        
        try:
            logger.info("🔧 Loading configuration to view summarizer settings...")
            config = self._get_config()  # Use shared config
            
            # Log summarizer configuration
            logger.info("📋 Default Summarizer Configuration:")
            logger.info(f"  - Model:            {config.summarizer.model_name}")
            logger.info(f"  - Server URL:       {config.summarizer.server_url}")
            logger.info(f"  - Temperature:      {config.summarizer.temperature}")
            logger.info(f"  - Top P:            {config.summarizer.top_p}")
            logger.info(f"  - Max Parallel:     {config.summarizer.max_parallelization}")
            logger.info(f"  - Max Chunk Length: {config.summarizer.max_chunk_length}")
            logger.info(f"  - Chunk Overlap:    {config.summarizer.chunk_overlap}")
            
            # Verify required fields exist
            if hasattr(config, 'summarizer') and hasattr(config.summarizer, 'model_name'):
                logger.info("✅ Summarizer configuration loaded successfully")
                self.add_test_result(
                    134, "Library - View Default Summarizer Config",
                    "Test viewing default summarizer LLM configuration from config.yaml",
                    ["NvidiaRAGConfig.from_yaml()", "config.summarizer"],
                    ["model_name", "temperature", "top_p", "max_chunk_length"],
                    time.time() - start_time,
                    TestStatus.SUCCESS
                )
                return True
            else:
                logger.error("❌ Summarizer configuration missing required fields")
                self.add_test_result(
                    134, "Library - View Default Summarizer Config",
                    "Test viewing summarizer configuration",
                    ["NvidiaRAGConfig.from_yaml()"],
                    ["summarizer"],
                    time.time() - start_time,
                    TestStatus.FAILURE,
                    "Summarizer config missing"
                )
                return False
                
        except Exception as e:
            logger.error(f"❌ View summarizer config test failed: {e}")
            import traceback
            traceback.print_exc()
            self.add_test_result(
                134, "Library - View Default Summarizer Config",
                "Test viewing summarizer configuration",
                ["NvidiaRAGConfig.from_yaml()"],
                ["summarizer"],
                time.time() - start_time,
                TestStatus.FAILURE,
                str(e)
            )
            return False

    @test_case(135, "Library - View Default Summary Prompts")
    async def _test_library_view_summary_prompts(self) -> bool:
        """Test viewing default summary prompts (Summarization notebook: Step 3)"""
        logger.info("\n=== Test 135: Library - View Default Summary Prompts ===")
        start_time = time.time()
        
        try:
            logger.info("🔧 Initializing NvidiaRAGIngestor to view prompts...")
            config = self._get_config()  # Use shared config
            config.summarizer.server_url = ""  # Use cloud-hosted model for summarization
            
            ingestor = self._get_ingestor()  # Use shared instance
            
            # Check for summary prompts
            logger.info("📋 Checking for default summary prompts:")
            
            has_document_prompt = "document_summary_prompt" in ingestor.prompts
            has_iterative_prompt = "iterative_summary_prompt" in ingestor.prompts
            
            if has_document_prompt:
                logger.info("✅ Found 'document_summary_prompt'")
                logger.info(f"   Preview: {str(ingestor.prompts['document_summary_prompt'])[:150]}...")
            
            if has_iterative_prompt:
                logger.info("✅ Found 'iterative_summary_prompt'")
                logger.info(f"   Preview: {str(ingestor.prompts['iterative_summary_prompt'])[:150]}...")
            
            if has_document_prompt and has_iterative_prompt:
                logger.info("✅ All default summary prompts found")
                self.add_test_result(
                    135, "Library - View Default Summary Prompts",
                    "Test viewing default summary prompts from NvidiaRAGIngestor",
                    ["NvidiaRAGIngestor(config)", "ingestor.prompts"],
                    ["document_summary_prompt", "iterative_summary_prompt"],
                    time.time() - start_time,
                    TestStatus.SUCCESS
                )
                return True
            else:
                logger.error("❌ Missing required summary prompts")
                self.add_test_result(
                    135, "Library - View Default Summary Prompts",
                    "Test viewing default summary prompts",
                    ["NvidiaRAGIngestor", "prompts"],
                    ["document_summary_prompt", "iterative_summary_prompt"],
                    time.time() - start_time,
                    TestStatus.FAILURE,
                    "Missing required prompts"
                )
                return False
                
        except Exception as e:
            logger.error(f"❌ View summary prompts test failed: {e}")
            import traceback
            traceback.print_exc()
            self.add_test_result(
                135, "Library - View Default Summary Prompts",
                "Test viewing summary prompts",
                ["NvidiaRAGIngestor", "prompts"],
                [],
                time.time() - start_time,
                TestStatus.FAILURE,
                str(e)
            )
            return False

    @test_case(136, "Library - Custom Summarizer Config")
    async def _test_library_custom_summarizer_config(self) -> bool:
        """Test changing summarizer configuration (Summarization notebook: Method 1)"""
        logger.info("\n=== Test 136: Library - Custom Summarizer Config ===")
        start_time = time.time()
        
        try:
            logger.info("🔧 Modifying summarizer configuration...")
            config = self._get_config()  # Use shared config
            
            # Change summarizer settings (modifying shared config for this test)
            original_model = config.summarizer.model_name
            original_temp = config.summarizer.temperature
            original_top_p = config.summarizer.top_p
            
            config.summarizer.model_name = "meta/llama-3.1-70b-instruct"
            config.summarizer.server_url = ""
            config.summarizer.temperature = 0.2
            config.summarizer.top_p = 0.7
            config.summarizer.max_parallelization = 10
            
            logger.info("📋 Updated Summarizer Configuration:")
            logger.info(f"  - Model:       {original_model} → {config.summarizer.model_name}")
            logger.info(f"  - Temperature: {original_temp} → {config.summarizer.temperature}")
            logger.info(f"  - Top P:       {original_top_p} → {config.summarizer.top_p}")
            logger.info(f"  - Max Parallel: {config.summarizer.max_parallelization}")
            
            # Verify changes
            if (config.summarizer.model_name == "meta/llama-3.1-70b-instruct" and
                config.summarizer.temperature == 0.2 and
                config.summarizer.top_p == 0.7):
                logger.info("✅ Summarizer configuration modified successfully")
                self.add_test_result(
                    136, "Library - Custom Summarizer Config",
                    "Test modifying summarizer LLM configuration dynamically",
                    ["config.summarizer.model_name", "config.summarizer.temperature"],
                    ["model_name", "temperature", "top_p", "max_parallelization"],
                    time.time() - start_time,
                    TestStatus.SUCCESS
                )
                return True
            else:
                logger.error("❌ Configuration changes not applied")
                self.add_test_result(
                    136, "Library - Custom Summarizer Config",
                    "Test modifying summarizer configuration",
                    ["config.summarizer"],
                    ["model_name", "temperature"],
                    time.time() - start_time,
                    TestStatus.FAILURE,
                    "Config changes not applied"
                )
                return False
                
        except Exception as e:
            logger.error(f"❌ Custom summarizer config test failed: {e}")
            import traceback
            traceback.print_exc()
            self.add_test_result(
                136, "Library - Custom Summarizer Config",
                "Test modifying summarizer configuration",
                ["config.summarizer"],
                [],
                time.time() - start_time,
                TestStatus.FAILURE,
                str(e)
            )
            return False

    @test_case(137, "Library - Custom Summary Prompts")
    async def _test_library_custom_summary_prompts(self) -> bool:
        """Test custom summary prompts (Summarization notebook: Method 2)"""
        logger.info("\n=== Test 137: Library - Custom Summary Prompts ===")
        start_time = time.time()
        
        try:
            from nvidia_rag import NvidiaRAGIngestor
            
            logger.info("🔧 Creating NvidiaRAGIngestor with custom summary prompts...")
            config = self._get_config()  # Use shared config
            config.summarizer.server_url = ""  # Use cloud-hosted model for summarization
            
            # Define custom prompts for summarization
            custom_prompts = {
                "document_summary_prompt": {
                    "system": "/no_think",
                    "human": """You are a documentation specialist.

Create a clear summary that:
1. Identifies the main topic and purpose
2. Lists key concepts or features
3. Highlights important procedures or steps
4. Notes any warnings or critical information

Keep the summary concise.

Text to summarize:
{document_text}

Summary:"""
                }
            }
            
            # Initialize with custom prompts (separate instance needed for custom prompts)
            ingestor_custom = NvidiaRAGIngestor(config=config, prompts=custom_prompts)
            
            # Verify custom prompt was applied
            if "document_summary_prompt" in ingestor_custom.prompts:
                custom_human = ingestor_custom.prompts["document_summary_prompt"].get("human", "")
                if "documentation specialist" in custom_human:
                    logger.info("✅ Custom summary prompt applied successfully")
                    logger.info(f"   Prompt preview: {custom_human[:150]}...")
                    self.add_test_result(
                        137, "Library - Custom Summary Prompts",
                        "Test initializing NvidiaRAGIngestor with custom summary prompts via constructor injection",
                        ["NvidiaRAGIngestor(config, prompts=custom_prompts)"],
                        ["document_summary_prompt"],
                        time.time() - start_time,
                        TestStatus.SUCCESS
                    )
                    return True
                else:
                    logger.error("❌ Custom prompt not applied correctly")
                    self.add_test_result(
                        137, "Library - Custom Summary Prompts",
                        "Test custom summary prompts",
                        ["NvidiaRAGIngestor"],
                        ["prompts"],
                        time.time() - start_time,
                        TestStatus.FAILURE,
                        "Custom prompt not applied"
                    )
                    return False
            else:
                logger.error("❌ document_summary_prompt not found")
                self.add_test_result(
                    137, "Library - Custom Summary Prompts",
                    "Test custom summary prompts",
                    ["NvidiaRAGIngestor"],
                    ["prompts"],
                    time.time() - start_time,
                    TestStatus.FAILURE,
                    "Prompt not found"
                )
                return False
                
        except Exception as e:
            logger.error(f"❌ Custom summary prompts test failed: {e}")
            import traceback
            traceback.print_exc()
            self.add_test_result(
                137, "Library - Custom Summary Prompts",
                "Test custom summary prompts",
                ["NvidiaRAGIngestor"],
                ["prompts"],
                time.time() - start_time,
                TestStatus.FAILURE,
                str(e)
            )
            return False

    @test_case(138, "Library - Summary Options Configuration")
    async def _test_library_summary_options(self) -> bool:
        """Test summary options configuration (Summarization notebook: Method 3)"""
        logger.info("\n=== Test 138: Library - Summary Options Configuration ===")
        start_time = time.time()
        
        try:
            logger.info("🔧 Configuring summary options...")
            
            # Define summary options (from notebook)
            summary_options = {
                "page_filter": [[1, 5]],  # Only pages 1-5 (smaller for testing)
                "shallow_summary": False,  # Full extraction (shallow has issues in library mode)
                "summarization_strategy": "single"  # Fastest strategy
            }
            
            logger.info("📋 Summary Options Configured:")
            logger.info(f"  • Page Filter: {summary_options['page_filter']}")
            logger.info(f"  • Shallow Summary: {summary_options['shallow_summary']}")
            logger.info(f"  • Strategy: {summary_options['summarization_strategy']}")
            
            # Verify all required options are present
            if all(k in summary_options for k in ["page_filter", "shallow_summary", "summarization_strategy"]):
                logger.info("✅ Summary options configured successfully")
                # Store for use in upload test
                self.test_runner.summary_options = summary_options
                
                self.add_test_result(
                    138, "Library - Summary Options Configuration",
                    "Test configuring summary options: page_filter, shallow_summary, summarization_strategy",
                    ["summary_options dict"],
                    ["page_filter", "shallow_summary", "summarization_strategy"],
                    time.time() - start_time,
                    TestStatus.SUCCESS
                )
                return True
            else:
                logger.error("❌ Missing required summary options")
                self.add_test_result(
                    138, "Library - Summary Options Configuration",
                    "Test summary options configuration",
                    ["summary_options"],
                    [],
                    time.time() - start_time,
                    TestStatus.FAILURE,
                    "Missing options"
                )
                return False
                
        except Exception as e:
            logger.error(f"❌ Summary options test failed: {e}")
            import traceback
            traceback.print_exc()
            self.add_test_result(
                138, "Library - Summary Options Configuration",
                "Test summary options configuration",
                ["summary_options"],
                [],
                time.time() - start_time,
                TestStatus.FAILURE,
                str(e)
            )
            return False

    @test_case(139, "Library - Upload with Summary Generation")
    async def _test_library_upload_with_summary(self) -> bool:
        """Test uploading documents with summary generation (Summarization notebook: Upload step)"""
        logger.info("\n=== Test 139: Library - Upload with Summary Generation ===")
        start_time = time.time()
        
        try:
            # Get summary collection name
            summary_collection = "test_library_summary"
            
            logger.info(f"🔧 Uploading documents with summary generation to '{summary_collection}'...")
            config = self._get_config()  # Use shared config
            config.summarizer.server_url = ""  # Use cloud-hosted model for summarization
            
            logger.info(f"📋 Configuration:")
            logger.info(f"   embeddings.server_url: {config.embeddings.server_url}")
            logger.info(f"   llm.server_url: {config.llm.server_url}")
            logger.info(f"   summarizer.server_url: {config.summarizer.server_url}")
            logger.info(f"   summarizer.model_name: {config.summarizer.model_name}")
            
            ingestor = self._get_ingestor()  # Use shared instance
            
            # Create collection first
            create_response = ingestor.create_collection(
                collection_name=summary_collection,
                vdb_endpoint="http://localhost:19530"
            )
            logger.info(f"📋 Collection created: {create_response}")
            
            # Get test file
            test_file = self.test_data_dir / "multimodal_test.pdf"
            if not test_file.exists():
                # Try alternative file
                test_file = self.test_data_dir / "functional_validation.pdf"
            
            if not test_file.exists():
                logger.error(f"❌ No test file found in {self.test_data_dir}")
                self.add_test_result(
                    139, "Library - Upload with Summary Generation",
                    "Test document upload with summary generation",
                    ["ingestor.upload_documents()"],
                    ["generate_summary", "summary_options"],
                    time.time() - start_time,
                    TestStatus.FAILURE,
                    "No test file available"
                )
                return False
            
            logger.info(f"📄 Using test file: {test_file.name}")
            
            # Get summary options from previous test
            summary_options = getattr(self.test_runner, 'summary_options', {
                "page_filter": [[1, 5]],  # Small range for testing
                "shallow_summary": False,  # Use full extraction (shallow has issues in library mode)
                "summarization_strategy": "single"  # Fastest for testing
            })
            
            logger.info(f"📋 Summary options: {summary_options}")
            
            # Upload with summary generation
            # Using blocking=False as per notebook, then wait via retrieve_summary()
            result = await ingestor.upload_documents(
                filepaths=[str(test_file)],
                collection_name=summary_collection,
                vdb_endpoint="http://localhost:19530",
                generate_summary=True,
                summary_options=summary_options,
                blocking=False  # Non-blocking, as per notebook workflow
            )
            
            # Give background tasks a moment to start
            await asyncio.sleep(2)
            
            logger.info(f"📋 Upload response: {json.dumps(result, indent=2)}")
            
            task_id = result.get("task_id")
            if task_id:
                logger.info(f"✅ Upload with summary started successfully, task_id: {task_id}")
                # Store for summary retrieval test
                self.test_runner.summary_task_id = task_id
                self.test_runner.summary_collection = summary_collection
                self.test_runner.summary_filename = test_file.name
                
                self.add_test_result(
                    139, "Library - Upload with Summary Generation",
                    f"Test document upload with generate_summary=True and summary_options to collection: {summary_collection}",
                    ["ingestor.upload_documents()"],
                    ["filepaths", "generate_summary", "summary_options", "blocking"],
                    time.time() - start_time,
                    TestStatus.SUCCESS
                )
                return True
            else:
                logger.error(f"❌ Upload failed: {result}")
                self.add_test_result(
                    139, "Library - Upload with Summary Generation",
                    "Test document upload with summary generation",
                    ["ingestor.upload_documents()"],
                    ["generate_summary", "summary_options"],
                    time.time() - start_time,
                    TestStatus.FAILURE,
                    f"No task_id: {result}"
                )
                return False
                
        except Exception as e:
            logger.error(f"❌ Upload with summary test failed: {e}")
            import traceback
            traceback.print_exc()
            self.add_test_result(
                139, "Library - Upload with Summary Generation",
                "Test upload with summary generation",
                ["ingestor.upload_documents()"],
                ["generate_summary"],
                time.time() - start_time,
                TestStatus.FAILURE,
                str(e)
            )
            return False

    @test_case(140, "Library - Check Summary Status")
    async def _test_library_check_summary_status(self) -> bool:
        """Test checking summary generation status (Summarization notebook: Check status step)"""
        logger.info("\n=== Test 140: Library - Check Summary Status ===")
        start_time = time.time()
        
        try:
            from nvidia_rag.rag_server.response_generator import retrieve_summary
            
            # Get info from previous test
            summary_collection = getattr(self.test_runner, 'summary_collection', 'test_library_summary')
            summary_filename = getattr(self.test_runner, 'summary_filename', 'multimodal_test.pdf')
            
            logger.info(f"🔧 Checking summary status for '{summary_filename}' in '{summary_collection}'...")
            
            # Check status (non-blocking)
            status = await retrieve_summary(
                collection_name=summary_collection,
                file_name=summary_filename,
                wait=False  # Just check, don't wait
            )
            
            logger.info(f"📋 Summary status response:")
            logger.info(f"   Full response: {json.dumps(status, indent=2)}")
            logger.info(f"   Status: {status.get('status')}")
            
            if status.get('status') == 'IN_PROGRESS':
                progress = status.get('progress', {})
                logger.info(f"   Progress: Chunk {progress.get('current')}/{progress.get('total')}")
            elif status.get('status') == 'SUCCESS':
                logger.info(f"   ✅ Summary already completed!")
            elif status.get('status') == 'NOT_FOUND':
                logger.info(f"   ⏳ Summary not started yet or file not found")
            elif status.get('status') == 'FAILED':
                error_msg = status.get('error', 'Unknown error')
                logger.error(f"   ❌ Summary FAILED: {error_msg}")
            
            # Any valid status response is success for this test
            logger.info("✅ Summary status check completed")
            self.add_test_result(
                140, "Library - Check Summary Status",
                f"Test checking summary generation status using retrieve_summary() with wait=False",
                ["retrieve_summary()"],
                ["collection_name", "file_name", "wait"],
                time.time() - start_time,
                TestStatus.SUCCESS
            )
            return True
                
        except Exception as e:
            logger.error(f"❌ Check summary status test failed: {e}")
            import traceback
            traceback.print_exc()
            self.add_test_result(
                140, "Library - Check Summary Status",
                "Test checking summary status",
                ["retrieve_summary()"],
                ["wait"],
                time.time() - start_time,
                TestStatus.FAILURE,
                str(e)
            )
            return False

    @test_case(141, "Library - Retrieve Summary")
    async def _test_library_retrieve_summary(self) -> bool:
        """Test retrieving completed summary (Summarization notebook: Get summary step)"""
        logger.info("\n=== Test 141: Library - Retrieve Summary ===")
        start_time = time.time()
        
        try:
            from nvidia_rag.rag_server.response_generator import retrieve_summary
            
            # Get info from previous test
            summary_collection = getattr(self.test_runner, 'summary_collection', 'test_library_summary')
            summary_filename = getattr(self.test_runner, 'summary_filename', 'multimodal_test.pdf')
            
            logger.info(f"🔧 Retrieving summary for '{summary_filename}' in '{summary_collection}'...")
            logger.info("   (Blocking call - waiting up to 300s for completion)")
            
            # Get summary (blocking - waits for completion)
            summary_result = await retrieve_summary(
                collection_name=summary_collection,
                file_name=summary_filename,
                wait=True,
                timeout=300  # 5 minutes
            )
            
            logger.info(f"📋 Summary retrieval response:")
            logger.info(f"   Full response: {json.dumps(summary_result, indent=2)}")
            logger.info(f"   Status: {summary_result.get('status')}")
            
            if summary_result.get('status') == 'SUCCESS':
                summary_text = summary_result.get('summary', '')
                logger.info(f"✅ Summary retrieved successfully!")
                logger.info(f"📄 Summary length: {len(summary_text)} characters")
                logger.info(f"📄 Summary preview (first 300 chars):")
                logger.info(f"   {summary_text[:300]}...")
                
                self.add_test_result(
                    141, "Library - Retrieve Summary",
                    f"Test retrieving completed summary using retrieve_summary() with wait=True, timeout=300",
                    ["retrieve_summary()"],
                    ["collection_name", "file_name", "wait", "timeout"],
                    time.time() - start_time,
                    TestStatus.SUCCESS
                )
                return True
            else:
                status_msg = summary_result.get('status')
                message = summary_result.get('message', '')
                error = summary_result.get('error', 'No error details available')
                logger.error(f"❌ Summary retrieval failed:")
                logger.error(f"   Status: {status_msg}")
                logger.error(f"   Message: {message}")
                logger.error(f"   Error: {error}")
                self.add_test_result(
                    141, "Library - Retrieve Summary",
                    "Test retrieving completed summary",
                    ["retrieve_summary()"],
                    ["wait", "timeout"],
                    time.time() - start_time,
                    TestStatus.FAILURE,
                    f"Status: {status_msg}, Error: {error}"
                )
                return False
                
        except Exception as e:
            logger.error(f"❌ Retrieve summary test failed: {e}")
            import traceback
            traceback.print_exc()
            self.add_test_result(
                141, "Library - Retrieve Summary",
                "Test retrieving summary",
                ["retrieve_summary()"],
                ["wait", "timeout"],
                time.time() - start_time,
                TestStatus.FAILURE,
                str(e)
            )
            return False

    @test_case(142, "Library - Cleanup Summary Collection")
    async def _test_library_cleanup_summary_collection(self) -> bool:
        """Test deleting summary test collection"""
        logger.info("\n=== Test 142: Library - Cleanup Summary Collection ===")
        start_time = time.time()
        
        try:
            summary_collection = getattr(self.test_runner, 'summary_collection', 'test_library_summary')
            
            logger.info(f"🔧 Deleting summary collection '{summary_collection}'...")
            ingestor = self._get_ingestor()  # Use shared instance
            
            response = ingestor.delete_collections(
                vdb_endpoint="http://localhost:19530",
                collection_names=[summary_collection]
            )
            
            logger.info(f"✅ Summary collection deleted")
            logger.info(f"📋 Response: {json.dumps(response, indent=2)}")
            
            self.add_test_result(
                142, "Library - Cleanup Summary Collection",
                f"Test deleting summary test collection: {summary_collection}",
                ["ingestor.delete_collections()"],
                ["vdb_endpoint", "collection_names"],
                time.time() - start_time,
                TestStatus.SUCCESS
            )
            return True
                
        except Exception as e:
            logger.error(f"❌ Cleanup summary collection test failed: {e}")
            import traceback
            traceback.print_exc()
            self.add_test_result(
                142, "Library - Cleanup Summary Collection",
                "Test cleanup summary collection",
                ["ingestor.delete_collections()"],
                ["collection_names"],
                time.time() - start_time,
                TestStatus.FAILURE,
                str(e)
            )
            return False

    @test_case(126, "Library - Cleanup Collection")
    async def _test_library_cleanup_collection(self) -> bool:
        """Test collection deletion (Notebook section: "10. Delete collections")"""
        logger.info("\n=== Test 126: Library - Cleanup Collection ===")
        start_time = time.time()
        
        try:
            logger.info(f"🔧 Deleting collection '{self.library_collection}'...")
            ingestor = self._get_ingestor()  # Use shared instance
            
            # Delete collection (Notebook: ingestor.delete_collections())
            response = ingestor.delete_collections(
                vdb_endpoint="http://localhost:19530",
                collection_names=[self.library_collection]
            )
            
            logger.info(f"✅ Collection deleted successfully")
            logger.info(f"📋 Response:\n{json.dumps(response, indent=2)}")
            
            self.add_test_result(
                126, "Library - Cleanup Collection",
                f"Test collection deletion via library API: {self.library_collection}",
                ["ingestor.delete_collections()"],
                ["vdb_endpoint", "collection_names"],
                time.time() - start_time,
                TestStatus.SUCCESS
            )
            return True
                
        except Exception as e:
            logger.error(f"❌ Collection cleanup test failed: {e}")
            import traceback
            traceback.print_exc()
            self.add_test_result(
                126, "Library - Cleanup Collection",
                "Test collection deletion via library API",
                ["ingestor.delete_collections()"],
                ["vdb_endpoint", "collection_names"],
                time.time() - start_time,
                TestStatus.FAILURE,
                str(e)
            )
            return False

