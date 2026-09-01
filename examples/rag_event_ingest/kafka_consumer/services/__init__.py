# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# services/__init__.py
"""External service clients."""

from .storage import ObjectStorage
from .document_indexer import DocumentIndexer

__all__ = ['ObjectStorage', 'DocumentIndexer']
