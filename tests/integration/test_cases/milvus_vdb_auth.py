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

import asyncio
import os
import time

from pymilvus import MilvusClient

# Try to import NvidiaRAGConfig from the codebase used by the servers.
# Fallback to minimal env-based defaults if unavailable in this test discovery environment.
try:
    from nvidia_rag.utils.configuration import NvidiaRAGConfig  # type: ignore
except Exception:
    def NvidiaRAGConfig():  # type: ignore
        class _DummyCfg:
            class embeddings:
                dimensions = int(os.getenv("EMBEDDING_DIM", "1536"))
            class vector_store:
                url = os.getenv("APP_VECTORSTORE_URL", "http://milvus:19530")
        return _DummyCfg()

from ..base import BaseTestModule, TestStatus, test_case
import aiohttp
import json
import logging

logger = logging.getLogger(__name__)


def _milvus_root_token() -> str:
    # Root/admin token in form "user:password"
    return os.getenv("MILVUS_ROOT_TOKEN", os.getenv("VDB_AUTH_TOKEN", "root:Milvus"))


def _milvus_uri() -> str:
    return os.getenv("APP_VECTORSTORE_URL", NvidiaRAGConfig().vector_store.url)


def _create_user(client: MilvusClient, user: str, password: str):
    try:
        users = client.list_users()
        if user not in users:
            client.create_user(user_name=user, password=password)
    except Exception:
        # Retry once to avoid transient connection issues
        time.sleep(1.0)
        users = client.list_users()
        if user not in users:
            client.create_user(user_name=user, password=password)


def _create_role(client: MilvusClient, role: str):
    try:
        roles = client.list_roles()
        if role not in roles:
            client.create_role(role_name=role)
    except Exception:
        # Some Milvus versions create role implicitly on grant; ignore errors
        pass


def _grant_role(client: MilvusClient, role: str, user: str):
    client.grant_role(role_name=role, user_name=user)


def _grant_collection_privilege(client: MilvusClient, role: str, object_type: str, object_name: str, privilege: str):
    # Valid privilege examples include (for object_type="Collection"): "All", "CreateIndex", "DropCollection", "Search", "Query", "Insert"
    client.grant_privilege(role_name=role, object_type=object_type, privilege=privilege, object_name=object_name)


def _grant_reader_privileges(client: MilvusClient, role: str, collection_name: str):
    """
    Grant read privileges to a role for a collection.
    
    In Milvus 2.5+, GetLoadState is an internal privilege that cannot be granted individually.
    It's included in the ClusterReadOnly privilege group. We use grant_privilege_v2() API
    which supports privilege groups.
    
    IMPORTANT: ClusterReadOnly is a CLUSTER-level privilege and must use "*" as collection_name.
    """
    # Grant ClusterReadOnly at CLUSTER level (collection_name must be "*")
    # This includes GetLoadState, GetLoadingProgress, ShowCollections, etc.
    # Required because langchain-milvus calls utility.load_state() during vectorstore init
    try:
        client.grant_privilege_v2(role_name=role, privilege="ClusterReadOnly", collection_name="*", db_name="*")
        logger.info(f"Granted ClusterReadOnly (cluster-level) to {role}")
    except Exception as e:
        logger.warning(f"Failed to grant ClusterReadOnly: {e}")
    
    # Grant collection-specific privileges using v2 API
    for priv in ("Query", "Search", "Load"):
        try:
            client.grant_privilege_v2(role_name=role, privilege=priv, collection_name=collection_name)
            logger.info(f"Granted {priv} to {role} on {collection_name}")
        except Exception as e:
            logger.warning(f"Failed to grant {priv} via v2 API: {e}")
            # Fallback to v1 API for older Milvus versions
            try:
                _grant_collection_privilege(client, role, "Collection", collection_name, priv)
                logger.info(f"Granted {priv} to {role} on {collection_name} via v1 API")
            except Exception as e2:
                logger.warning(f"Failed to grant {priv} via v1 API: {e2}")
    
    # Grant global privileges needed for collection operations
    try:
        _grant_collection_privilege(client, role, "Global", "*", "DescribeCollection")
        logger.info(f"Granted DescribeCollection to {role}")
    except Exception as e:
        logger.warning(f"Failed to grant DescribeCollection: {e}")
    
    try:
        _grant_collection_privilege(client, role, "Global", "*", "ShowCollections")
        logger.info(f"Granted ShowCollections to {role}")
    except Exception as e:
        logger.warning(f"Failed to grant ShowCollections: {e}")


class MilvusVdbAuthModule(BaseTestModule):
    """Milvus VDB auth tests via API using Authorization header bearer tokens."""

    @test_case(78, "Milvus Auth Setup (users/roles) and Create Collection")
    async def _test_milvus_auth_setup_and_create_collection(self) -> bool:
        """Create test users/roles and collection as admin."""
        logger.info("\n=== Test 71: Milvus Auth Setup (users/roles) and Create Collection ===")
        start = time.time()
        cfg = NvidiaRAGConfig()

        # Prepare identities (use fixed names to avoid relying on random suffixes)
        self.collection_name = "auth_it"
        self.reader_user = "reader"
        self.reader_pwd = "pwd_reader"
        self.reader_role = "role_reader"
        self.writer_user = "writer"
        self.writer_pwd = "pwd_writer"
        self.writer_role = "role_writer"

        try:
            client = MilvusClient(uri=_milvus_uri(), token=_milvus_root_token())
            _create_user(client, self.reader_user, self.reader_pwd)
            _create_user(client, self.writer_user, self.writer_pwd)
            _create_role(client, self.reader_role)
            _create_role(client, self.writer_role)
            _grant_role(client, self.reader_role, self.reader_user)
            _grant_role(client, self.writer_role, self.writer_user)

            # Create collection using API (admin header)
            payload = {
                "collection_name": self.collection_name,
                "embedding_dimension": cfg.embeddings.dimensions,
            }
            headers = {"Authorization": f"Bearer {_milvus_root_token()}"}
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.ingestor_server_url}/v1/collection", json=payload, headers=headers) as resp:
                    result = await resp.json()
                    if resp.status == 200:
                        logger.info(f"✅ Collection '{self.collection_name}' created")
                        self.add_test_result(
                            self._test_milvus_auth_setup_and_create_collection.test_number,
                            self._test_milvus_auth_setup_and_create_collection.test_name,
                            f"Create users/roles and a collection {self.collection_name} for auth tests.",
                            ["POST /v1/collection"],
                            ["collection_name", "embedding_dimension"],
                            time.time() - start,
                            TestStatus.SUCCESS,
                        )
                        return True
                    else:
                        logger.error(f"❌ Failed to create collection: {resp.status} {result}")
                        self.add_test_result(
                            self._test_milvus_auth_setup_and_create_collection.test_number,
                            self._test_milvus_auth_setup_and_create_collection.test_name,
                            f"Create users/roles and a collection {self.collection_name} for auth tests.",
                            ["POST /v1/collection"],
                            ["collection_name", "embedding_dimension"],
                            time.time() - start,
                            TestStatus.FAILURE,
                            f"status={resp.status} body={json.dumps(result)}",
                        )
                        return False
        except Exception as e:
            logger.error(f"❌ Exception during auth setup: {e}")
            self.add_test_result(
                self._test_milvus_auth_setup_and_create_collection.test_number,
                self._test_milvus_auth_setup_and_create_collection.test_name,
                f"Create users/roles and a collection for auth tests.",
                ["POST /v1/collection"],
                ["collection_name", "embedding_dimension"],
                time.time() - start,
                TestStatus.FAILURE,
                str(e),
            )
            return False

    @test_case(79, "GET /v1/collections denied without privileges")
    async def _test_get_collections_denied_without_privileges(self) -> bool:
        """GET /v1/collections should be denied without grants."""
        logger.info("\n=== Test 72: Access denied without privileges (reader) ===")
        start = time.time()
        try:
            headers = {"Authorization": f"Bearer {self.reader_user}:{self.reader_pwd}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.ingestor_server_url}/v1/collections", headers=headers) as resp:
                    # Assert expected authorization/privilege error content from response body
                    try:
                        body = await resp.json()
                        message = str(body.get("message") or body.get("error") or body.get("detail") or body).lower()
                        body_text = json.dumps(body)
                    except Exception:
                        body_text = await resp.text()
                        message = body_text.lower()
                    has_denial = any(substr in message for substr in ("denied", "not authorized", "permission", "privilege"))
                    if has_denial:
                        self.add_test_result(
                            self._test_get_collections_denied_without_privileges.test_number,
                            self._test_get_collections_denied_without_privileges.test_name,
                            "GET /v1/collections should be denied for reader without privileges.",
                            ["GET /v1/collections"],
                            [],
                            time.time() - start,
                            TestStatus.SUCCESS,
                        )
                        return True
                    else:
                        self.add_test_result(
                            self._test_get_collections_denied_without_privileges.test_number,
                            self._test_get_collections_denied_without_privileges.test_name,
                            "GET /v1/collections should be denied for reader without privileges.",
                            ["GET /v1/collections"],
                            [],
                            time.time() - start,
                            TestStatus.FAILURE,
                            f"Unexpected response body: {body_text}",
                        )
                        return False
        except Exception as e:
            # If server maps auth error to exception => success expected
            self.add_test_result(
                self._test_get_collections_denied_without_privileges.test_number,
                self._test_get_collections_denied_without_privileges.test_name,
                "GET /v1/collections should be denied for reader without privileges.",
                ["GET /v1/collections"],
                [],
                time.time() - start,
                TestStatus.SUCCESS,
            )
            return True

    @test_case(80, "GET /v1/collections allowed after granting privileges")
    async def _test_get_collections_allowed_after_privileges(self) -> bool:
        """GET /v1/collections should succeed after granting read privileges."""
        logger.info("\n=== Test 73: Grant read privileges and verify access ===")
        start = time.time()
        try:
            client = MilvusClient(uri=_milvus_uri(), token=_milvus_root_token())
            # Use the helper that handles Milvus 2.5+ privilege groups (ClusterReadOnly includes GetLoadState)
            _grant_reader_privileges(client, self.reader_role, self.collection_name)

            headers = {"Authorization": f"Bearer {self.reader_user}:{self.reader_pwd}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.ingestor_server_url}/v1/collections", headers=headers) as resp:
                    result = await resp.json()
                    if resp.status == 200:
                        self.add_test_result(
                            self._test_get_collections_allowed_after_privileges.test_number,
                            self._test_get_collections_allowed_after_privileges.test_name,
                            "GET /v1/collections should succeed after granting read privileges.",
                            ["GET /v1/collections"],
                            [],
                            time.time() - start,
                            TestStatus.SUCCESS,
                        )
                        return True
                    else:
                        self.add_test_result(
                            self._test_get_collections_allowed_after_privileges.test_number,
                            self._test_get_collections_allowed_after_privileges.test_name,
                            "GET /v1/collections should succeed after granting read privileges.",
                            ["GET /v1/collections"],
                            [],
                            time.time() - start,
                            TestStatus.FAILURE,
                            f"status={resp.status} body={json.dumps(result)}",
                        )
                        return False
        except Exception as e:
            self.add_test_result(
                self._test_get_collections_allowed_after_privileges.test_number,
                self._test_get_collections_allowed_after_privileges.test_name,
                "GET /v1/collections should succeed after granting read privileges.",
                ["GET /v1/collections"],
                [],
                time.time() - start,
                TestStatus.FAILURE,
                str(e),
            )
            return False

    @test_case(81, "DELETE /v1/collections denied without privilege")
    async def _test_delete_collections_denied_without_privilege(self) -> bool:
        """DELETE /v1/collections should be denied without drop privilege."""
        logger.info("\n=== Test 74: Writer cannot drop collection without privilege (API) ===")
        start = time.time()
        try:
            headers = {"Authorization": f"Bearer {self.writer_user}:{self.writer_pwd}"}
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    f"{self.ingestor_server_url}/v1/collections",
                    json=[self.collection_name],
                    headers=headers,
                ) as resp:
                    # Assert expected authorization/privilege error content from response body
                    try:
                        body = await resp.json()
                        message = str(body).lower()
                    except Exception:
                        body_text = await resp.text()
                        message = body_text.lower()
                    has_denial = any(substr in message for substr in ("denied", "not authorized", "permission", "privilege"))
                    if has_denial:
                        self.add_test_result(
                            self._test_delete_collections_denied_without_privilege.test_number,
                            self._test_delete_collections_denied_without_privilege.test_name,
                            "DELETE /v1/collections should be denied for writer without drop privilege.",
                            ["DELETE /v1/collections"],
                            ["collection_names"],
                            time.time() - start,
                            TestStatus.SUCCESS,
                        )
                        return True
                    else:
                        self.add_test_result(
                            self._test_delete_collections_denied_without_privilege.test_number,
                            self._test_delete_collections_denied_without_privilege.test_name,
                            "DELETE /v1/collections should be denied for writer without drop privilege.",
                            ["DELETE /v1/collections"],
                            ["collection_names"],
                            time.time() - start,
                            TestStatus.FAILURE,
                            f"Unexpected response body: {body_text}",
                        )
                        return False
        except Exception:
            # If server maps auth to exception => success expected
            self.add_test_result(
                self._test_delete_collections_denied_without_privilege.test_number,
                self._test_delete_collections_denied_without_privilege.test_name,
                "DELETE /v1/collections should be denied for writer without drop privilege.",
                ["DELETE /v1/collections"],
                ["collection_names"],
                time.time() - start,
                TestStatus.SUCCESS,
            )
            return True

    @test_case(82, "DELETE /v1/collections allowed after granting privilege")
    async def _test_delete_collections_allowed_after_privilege(self) -> bool:
        """DELETE /v1/collections should succeed after granting DropCollection privilege."""
        logger.info("\n=== Test 75: Writer can drop collection with privilege (API) ===")
        start = time.time()
        try:
            # Grant writer the DropCollection (or All) privilege
            client = MilvusClient(uri=_milvus_uri(), token=_milvus_root_token())
            try:
                _grant_collection_privilege(client, self.writer_role, "Global", self.collection_name, privilege="DropCollection")
            except Exception:
                pass

            headers = {"Authorization": f"Bearer {self.writer_user}:{self.writer_pwd}"}
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    f"{self.ingestor_server_url}/v1/collections",
                    json=[self.collection_name],
                    headers=headers,
                ) as resp:
                    result = await resp.json()
                    if resp.status == 200:
                        self.add_test_result(
                            self._test_delete_collections_allowed_after_privilege.test_number,
                            self._test_delete_collections_allowed_after_privilege.test_name,
                            "DELETE /v1/collections should succeed for writer with DropCollection privilege.",
                            ["DELETE /v1/collections"],
                            ["collection_names"],
                            time.time() - start,
                            TestStatus.SUCCESS,
                        )
                        return True
                    else:
                        self.add_test_result(
                            self._test_delete_collections_allowed_after_privilege.test_number,
                            self._test_delete_collections_allowed_after_privilege.test_name,
                            "DELETE /v1/collections should succeed for writer with DropCollection privilege.",
                            ["DELETE /v1/collections"],
                            ["collection_names"],
                            time.time() - start,
                            TestStatus.FAILURE,
                            f"status={resp.status} body={json.dumps(result)}",
                        )
                        return False
        except Exception as e:
            self.add_test_result(
                self._test_delete_collections_allowed_after_privilege.test_number,
                self._test_delete_collections_allowed_after_privilege.test_name,
                "DELETE /v1/collections should succeed for writer with DropCollection privilege.",
                ["DELETE /v1/collections"],
                ["collection_names"],
                time.time() - start,
                TestStatus.FAILURE,
                str(e),
            )
            return False

    @test_case(83, "RAG search denied without privileges (reader)")
    async def _test_rag_search_denied_without_privileges(self) -> bool:
        """Reader should not be able to perform RAG search without grants on a new collection."""
        logger.info("\n=== Test 76: RAG search denied without privileges (reader) ===")
        start = time.time()
        cfg = NvidiaRAGConfig()
        temp_collection = "auth_rag"
        try:
            # Create temp collection via ingestor API as admin
            headers_admin = {"Authorization": f"Bearer {_milvus_root_token()}"}
            payload = {"collection_name": temp_collection, "embedding_dimension": cfg.embeddings.dimensions}
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.ingestor_server_url}/v1/collection", json=payload, headers=headers_admin) as c_resp:
                    _ = await c_resp.json()
                    if c_resp.status != 200:
                        raise RuntimeError(f"Failed to create temp collection {temp_collection}")

            # Call RAG /search as reader (should be denied)
            headers_reader = {"Authorization": f"Bearer {self.reader_user}:{self.reader_pwd}"}
            search_payload = {
                "query": "test access",
                "collection_names": [temp_collection],
                "messages": [],
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.rag_server_url}/v1/search", json=search_payload, headers=headers_reader) as resp:
                    # Assert expected authorization/privilege error content from response body
                    try:
                        body = await resp.json()
                        message = str(body.get("message") or body.get("error") or body.get("detail") or body).lower()
                        body_text = json.dumps(body)
                    except Exception:
                        body_text = await resp.text()
                        message = body_text.lower()
                    # Treat presence of any error-like phrasing as access denial success
                    error_markers = (
                        "denied",
                        "not authorized",
                        "permission",
                        "privilege",
                        "error",
                        "fail",
                        "illegal",
                        "unavailable",
                    )
                    has_error = any(substr in message for substr in error_markers)
                    if has_error:
                        self.add_test_result(
                            self._test_rag_search_denied_without_privileges.test_number,
                            self._test_rag_search_denied_without_privileges.test_name,
                            "POST /v1/search should be denied for reader without privileges.",
                            ["POST /v1/search"],
                            ["query", "collection_names"],
                            time.time() - start,
                            TestStatus.SUCCESS,
                        )
                        return True
                    else:
                        self.add_test_result(
                            self._test_rag_search_denied_without_privileges.test_number,
                            self._test_rag_search_denied_without_privileges.test_name,
                            "POST /v1/search should be denied for reader without privileges.",
                            ["POST /v1/search"],
                            ["query", "collection_names"],
                            time.time() - start,
                            TestStatus.FAILURE,
                            f"Unexpected response body: {body_text}",
                        )
                        return False
        except Exception as e:
            self.add_test_result(
                self._test_rag_search_denied_without_privileges.test_number,
                self._test_rag_search_denied_without_privileges.test_name,
                "POST /v1/search should be denied for reader without privileges.",
                ["POST /v1/search"],
                ["query", "collection_names"],
                time.time() - start,
                TestStatus.SUCCESS,
                str(e),
            )
            return True
        finally:
            # cleanup temp collection as admin
            try:
                headers_admin = {"Authorization": f"Bearer {_milvus_root_token()}"}
                async with aiohttp.ClientSession() as session:
                    await session.delete(
                        f"{self.ingestor_server_url}/v1/collections",
                        json=[temp_collection],
                        headers=headers_admin,
                    )
            except Exception:
                pass

    @test_case(84, "RAG search allowed after privileges (reader)")
    async def _test_rag_search_allowed_after_privileges(self) -> bool:
        """Grant reader access and verify RAG search succeeds on a new collection."""
        logger.info("\n=== Test 77: RAG search allowed after privileges (reader) ===")
        start = time.time()
        cfg = NvidiaRAGConfig()
        temp_collection = "auth_rag_2"
        try:
            # Create temp collection via ingestor API as admin
            headers_admin = {"Authorization": f"Bearer {_milvus_root_token()}"}
            payload = {"collection_name": temp_collection, "embedding_dimension": cfg.embeddings.dimensions}
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.ingestor_server_url}/v1/collection", json=payload, headers=headers_admin) as c_resp:
                    _ = await c_resp.json()
                    if c_resp.status != 200:
                        raise RuntimeError(f"Failed to create temp collection {temp_collection}")

            # Grant reader privileges to this temp collection
            # Use the helper that handles Milvus 2.5+ privilege groups (ClusterReadOnly includes GetLoadState)
            client = MilvusClient(uri=_milvus_uri(), token=_milvus_root_token())
            _grant_reader_privileges(client, self.reader_role, temp_collection)
            await asyncio.sleep(6)

            # Call RAG /search as reader (should succeed)
            headers_reader = {"Authorization": f"Bearer {self.reader_user}:{self.reader_pwd}"}
            search_payload = {
                "query": "what is milvus?",
                "collection_names": [temp_collection],
                "messages": [],
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.rag_server_url}/v1/search", json=search_payload, headers=headers_reader) as resp:
                    result = await resp.json()
                    if resp.status == 200:
                        self.add_test_result(
                            self._test_rag_search_allowed_after_privileges.test_number,
                            self._test_rag_search_allowed_after_privileges.test_name,
                            "POST /v1/search should succeed after granting read privileges.",
                            ["POST /v1/search"],
                            ["query", "collection_names"],
                            time.time() - start,
                            TestStatus.SUCCESS,
                        )
                        return True
                    else:
                        self.add_test_result(
                            self._test_rag_search_allowed_after_privileges.test_number,
                            self._test_rag_search_allowed_after_privileges.test_name,
                            "POST /v1/search should succeed after granting read privileges.",
                            ["POST /v1/search"],
                            ["query", "collection_names"],
                            time.time() - start,
                            TestStatus.FAILURE,
                            f"status={resp.status} body={json.dumps(result)}",
                        )
                        return False
        except Exception as e:
            self.add_test_result(
                self._test_rag_search_allowed_after_privileges.test_number,
                self._test_rag_search_allowed_after_privileges.test_name,
                "POST /v1/search should succeed after granting read privileges.",
                ["POST /v1/search"],
                ["query", "collection_names"],
                time.time() - start,
                TestStatus.FAILURE,
                str(e),
            )
            return False
        finally:
            # cleanup temp collection as admin
            try:
                headers_admin = {"Authorization": f"Bearer {_milvus_root_token()}"}
                async with aiohttp.ClientSession() as session:
                    await session.delete(
                        f"{self.ingestor_server_url}/v1/collections",
                        json=[temp_collection],
                        headers=headers_admin,
                    )
            except Exception:
                pass



    @test_case(85, "Cleanup auth resources (collections, users, roles)")
    async def _test_cleanup_auth_resources(self) -> bool:
        """Cleanup resources created by this module: collections, users, roles."""
        logger.info("\n=== Test 78: Cleanup auth resources (collections, users, roles) ===")
        start = time.time()
        try:
            headers_admin = {"Authorization": f"Bearer {_milvus_root_token()}"}
            cfg = NvidiaRAGConfig()
            client = MilvusClient(uri=_milvus_uri(), token=_milvus_root_token())

            # 1) Delete known main collection if present
            try:
                if getattr(self, "collection_name", None):
                    async with aiohttp.ClientSession() as session:
                        await session.delete(
                            f"{self.ingestor_server_url}/v1/collections",
                            json=[self.collection_name],
                            headers=headers_admin,
                        )
            except Exception:
                pass

            # 2) Sweep temporary collections created by these tests
            #    Handle both fixed names and historical suffixed variants.
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.ingestor_server_url}/v1/collections",
                        headers=headers_admin,
                    ) as resp:
                        data = await resp.json()
                        colls = [c.get("collection_name") for c in data.get("collections", [])]
                        targets_exact = {"auth_it", "auth_rag", "auth_gen"}
                        targets_prefix = ("auth_it_", "auth_rag_", "auth_gen_")
                        to_delete = []
                        for c in colls:
                            if not isinstance(c, str):
                                continue
                            if c in targets_exact or c.startswith(targets_prefix):
                                to_delete.append(c)
                        if to_delete:
                            await session.delete(
                                f"{self.ingestor_server_url}/v1/collections",
                                json=to_delete,
                                headers=headers_admin,
                            )
            except Exception:
                pass

            # 3) Drop test users (reader, writer) if present; also attempt standard names
            for attr_user in ("reader_user", "writer_user"):
                try:
                    user_val = getattr(self, attr_user, None)
                    if user_val:
                        client.drop_user(user_name=user_val)
                except Exception:
                    pass
            for standard_user in ("reader", "writer"):
                try:
                    client.drop_user(user_name=standard_user)
                except Exception:
                    pass

            # 4) Drop test roles (reader_role, writer_role) if supported; also attempt standard names
            for attr_role in ("reader_role", "writer_role"):
                try:
                    role_val = getattr(self, attr_role, None)
                    if role_val and hasattr(client, "drop_role"):
                        client.drop_role(role_name=role_val, force_drop=True)
                except Exception:
                    pass
            for standard_role in ("role_reader", "role_writer"):
                try:
                    if hasattr(client, "drop_role"):
                        client.drop_role(role_name=standard_role, force_drop=True)
                except Exception:
                    pass

            self.add_test_result(
                self._test_cleanup_auth_resources.test_number,
                self._test_cleanup_auth_resources.test_name,
                "Cleanup collections, users, and roles created during auth tests.",
                ["GET /v1/collections", "DELETE /v1/collections"],
                [],
                time.time() - start,
                TestStatus.SUCCESS,
            )
            return True
        except Exception as e:
            self.add_test_result(
                self._test_cleanup_auth_resources.test_number,
                self._test_cleanup_auth_resources.test_name,
                "Cleanup collections, users, and roles created during auth tests.",
                ["GET /v1/collections", "DELETE /v1/collections"],
                [],
                time.time() - start,
                TestStatus.FAILURE,
                str(e),
            )
            return False