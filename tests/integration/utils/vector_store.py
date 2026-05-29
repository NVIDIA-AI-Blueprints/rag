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

"""Vector store backend detection for integration tests (matches RAG server APP_VECTORSTORE_NAME)."""

import os


def normalized_vector_store_name() -> str:
    raw = os.environ.get("APP_VECTORSTORE_NAME", "elasticsearch")
    return raw.strip().strip('"').strip("'").lower()


def is_elasticsearch_vector_store() -> bool:
    return normalized_vector_store_name() == "elasticsearch"
