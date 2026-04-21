# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sanity checks for nemo_retriever package exports."""

from nvidia_rag.ingestor_server import nemo_retriever as pkg


def test_public_exports() -> None:
    assert set(pkg.__all__) == {"NemoRetrieverHandler", "IngestSchemaManager"}
    assert pkg.NemoRetrieverHandler.__name__ == "NemoRetrieverHandler"
    assert pkg.IngestSchemaManager.__name__ == "IngestSchemaManager"
