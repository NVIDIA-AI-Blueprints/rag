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

from unittest.mock import MagicMock, patch

import pytest

from nvidia_rag.utils.batch_utils import (
    calculate_dynamic_batch_parameters,
    calculate_text_like_batch_params,
)


class TestCalculateDynamicBatchParameters:
    """Test calculate_dynamic_batch_parameters function"""

    def test_dynamic_batching_disabled(self, caplog):
        """Test when dynamic batching is disabled - should return default config values"""
        mock_config = MagicMock()
        mock_config.nv_ingest.enable_dynamic_batching = False
        mock_config.nv_ingest.files_per_batch = 50
        mock_config.nv_ingest.concurrent_batches = 3

        filepaths = ["file1.pdf", "file2.pdf", "file3.txt"]

        with caplog.at_level("INFO"):
            files_per_batch, concurrent_batches = calculate_dynamic_batch_parameters(
                filepaths, mock_config
            )

        assert files_per_batch == 50
        assert concurrent_batches == 3
        assert any(
            "Dynamic batching is disabled" in record.message
            for record in caplog.records
        )

    def test_empty_filepaths_list(self, caplog):
        """Test with empty filepaths list - should return default values and log warning"""
        mock_config = MagicMock()
        mock_config.nv_ingest.enable_dynamic_batching = True
        mock_config.nv_ingest.files_per_batch = 50
        mock_config.nv_ingest.concurrent_batches = 3

        filepaths = []

        with caplog.at_level("WARNING"):
            files_per_batch, concurrent_batches = calculate_dynamic_batch_parameters(
                filepaths, mock_config
            )

        assert files_per_batch == 50
        assert concurrent_batches == 3
        assert any(
            "Empty filepaths list provided" in record.message
            for record in caplog.records
        )

    @patch("nvidia_rag.utils.batch_utils.calculate_text_like_batch_params", return_value=(64, 4))
    def test_all_text_like_files(self, mock_text_batch_params, caplog):
        """Test with 100% text-like files - should optimize for text processing"""
        mock_config = MagicMock()
        mock_config.nv_ingest.enable_dynamic_batching = True
        mock_config.nv_ingest.files_per_batch = 50
        mock_config.nv_ingest.concurrent_batches = 3

        filepaths = [
            "file1.txt",
            "file2.md",
            "file3.json",
            "file4.html",
            "file5.sh",
        ]

        with caplog.at_level("INFO"):
            files_per_batch, concurrent_batches = calculate_dynamic_batch_parameters(
                filepaths, mock_config
            )

        assert files_per_batch == 64
        assert concurrent_batches == 4
        assert any(
            "100.0% text-like files" in record.message for record in caplog.records
        )
        assert any(
            "optimized parameters for text processing" in record.message
            for record in caplog.records
        )

    @patch("nvidia_rag.utils.batch_utils.calculate_text_like_batch_params", return_value=(64, 4))
    def test_majority_text_like_files(self, mock_text_batch_params, caplog):
        """Test with >50% text-like files - should optimize for text processing"""
        mock_config = MagicMock()
        mock_config.nv_ingest.enable_dynamic_batching = True
        mock_config.nv_ingest.files_per_batch = 50
        mock_config.nv_ingest.concurrent_batches = 3

        # 6 text files out of 10 = 60%
        filepaths = [
            "file1.txt",
            "file2.txt",
            "file3.md",
            "file4.json",
            "file5.html",
            "file6.sh",
            "file7.pdf",
            "file8.docx",
            "file9.pptx",
            "file10.jpg",
        ]

        with caplog.at_level("INFO"):
            files_per_batch, concurrent_batches = calculate_dynamic_batch_parameters(
                filepaths, mock_config
            )

        assert files_per_batch == 64
        assert concurrent_batches == 4
        assert any("60.0% text-like files" in record.message for record in caplog.records)

    def test_exactly_fifty_percent_text_files(self, caplog):
        """Test with exactly 50% text-like files - should use default config"""
        mock_config = MagicMock()
        mock_config.nv_ingest.enable_dynamic_batching = True
        mock_config.nv_ingest.files_per_batch = 50
        mock_config.nv_ingest.concurrent_batches = 3

        # 5 text files out of 10 = 50%
        filepaths = [
            "file1.txt",
            "file2.txt",
            "file3.md",
            "file4.json",
            "file5.html",
            "file6.pdf",
            "file7.docx",
            "file8.pptx",
            "file9.jpg",
            "file10.png",
        ]

        with caplog.at_level("INFO"):
            files_per_batch, concurrent_batches = calculate_dynamic_batch_parameters(
                filepaths, mock_config
            )

        assert files_per_batch == 50
        assert concurrent_batches == 3
        assert any("50.0% text-like files" in record.message for record in caplog.records)
        assert any(
            "default configuration parameters" in record.message
            for record in caplog.records
        )

    def test_minority_text_like_files(self, caplog):
        """Test with <50% text-like files - should use default config"""
        mock_config = MagicMock()
        mock_config.nv_ingest.enable_dynamic_batching = True
        mock_config.nv_ingest.files_per_batch = 50
        mock_config.nv_ingest.concurrent_batches = 3

        # 2 text files out of 10 = 20%
        filepaths = [
            "file1.txt",
            "file2.md",
            "file3.pdf",
            "file4.pdf",
            "file5.docx",
            "file6.pptx",
            "file7.jpg",
            "file8.png",
            "file9.mp4",
            "file10.xlsx",
        ]

        with caplog.at_level("INFO"):
            files_per_batch, concurrent_batches = calculate_dynamic_batch_parameters(
                filepaths, mock_config
            )

        assert files_per_batch == 50
        assert concurrent_batches == 3
        assert any("20.0% text-like files" in record.message for record in caplog.records)

    def test_no_text_like_files(self, caplog):
        """Test with 0% text-like files - should use default config"""
        mock_config = MagicMock()
        mock_config.nv_ingest.enable_dynamic_batching = True
        mock_config.nv_ingest.files_per_batch = 50
        mock_config.nv_ingest.concurrent_batches = 3

        filepaths = [
            "file1.pdf",
            "file2.docx",
            "file3.pptx",
            "file4.jpg",
            "file5.png",
        ]

        with caplog.at_level("INFO"):
            files_per_batch, concurrent_batches = calculate_dynamic_batch_parameters(
                filepaths, mock_config
            )

        assert files_per_batch == 50
        assert concurrent_batches == 3
        assert any("0.0% text-like files" in record.message for record in caplog.records)

    @patch("nvidia_rag.utils.batch_utils.calculate_text_like_batch_params", return_value=(64, 4))
    def test_case_insensitive_extension_handling(self, mock_text_batch_params):
        """Test that file extensions are handled case-insensitively"""
        mock_config = MagicMock()
        mock_config.nv_ingest.enable_dynamic_batching = True
        mock_config.nv_ingest.files_per_batch = 50
        mock_config.nv_ingest.concurrent_batches = 3

        # All uppercase and mixed case text extensions
        filepaths = [
            "file1.TXT",
            "file2.MD",
            "file3.JSON",
            "file4.Html",
            "file5.Sh",
        ]

        files_per_batch, concurrent_batches = calculate_dynamic_batch_parameters(
            filepaths, mock_config
        )

        # Should recognize these as text-like files
        assert files_per_batch == 64
        assert concurrent_batches == 4

    def test_files_without_extensions(self, caplog):
        """Test handling of files without extensions"""
        mock_config = MagicMock()
        mock_config.nv_ingest.enable_dynamic_batching = True
        mock_config.nv_ingest.files_per_batch = 50
        mock_config.nv_ingest.concurrent_batches = 3

        filepaths = [
            "file1",
            "file2",
            "file3",
            "file4.txt",
            "file5.txt",
        ]

        with caplog.at_level("INFO"):
            files_per_batch, concurrent_batches = calculate_dynamic_batch_parameters(
                filepaths, mock_config
            )

        # 2 text files out of 5 = 40%
        assert files_per_batch == 50
        assert concurrent_batches == 3

    @patch("nvidia_rag.utils.batch_utils.calculate_text_like_batch_params", return_value=(64, 4))
    def test_files_with_multiple_dots(self, mock_text_batch_params):
        """Test handling of files with multiple dots in filename"""
        mock_config = MagicMock()
        mock_config.nv_ingest.enable_dynamic_batching = True
        mock_config.nv_ingest.files_per_batch = 50
        mock_config.nv_ingest.concurrent_batches = 3

        filepaths = [
            "file.name.with.dots.txt",
            "another.file.md",
            "yet.another.json",
            "some.file.html",
        ]

        files_per_batch, concurrent_batches = calculate_dynamic_batch_parameters(
            filepaths, mock_config
        )

        # All text-like files
        assert files_per_batch == 64
        assert concurrent_batches == 4

    @patch("nvidia_rag.utils.batch_utils.calculate_text_like_batch_params", return_value=(16, 4))
    def test_single_file_text(self, mock_text_batch_params):
        """Test with single text-like file"""
        mock_config = MagicMock()
        mock_config.nv_ingest.enable_dynamic_batching = True
        mock_config.nv_ingest.files_per_batch = 50
        mock_config.nv_ingest.concurrent_batches = 3

        filepaths = ["file1.txt"]

        files_per_batch, concurrent_batches = calculate_dynamic_batch_parameters(
            filepaths, mock_config
        )

        # 100% text-like
        assert files_per_batch == 16
        assert concurrent_batches == 4

    def test_single_file_non_text(self):
        """Test with single non-text file"""
        mock_config = MagicMock()
        mock_config.nv_ingest.enable_dynamic_batching = True
        mock_config.nv_ingest.files_per_batch = 50
        mock_config.nv_ingest.concurrent_batches = 3

        filepaths = ["file1.pdf"]

        files_per_batch, concurrent_batches = calculate_dynamic_batch_parameters(
            filepaths, mock_config
        )

        # 0% text-like
        assert files_per_batch == 50
        assert concurrent_batches == 3

    @patch("nvidia_rag.utils.batch_utils.calculate_text_like_batch_params", return_value=(64, 4))
    def test_full_file_paths(self, mock_text_batch_params):
        """Test with full file paths including directories"""
        mock_config = MagicMock()
        mock_config.nv_ingest.enable_dynamic_batching = True
        mock_config.nv_ingest.files_per_batch = 50
        mock_config.nv_ingest.concurrent_batches = 3

        filepaths = [
            "/path/to/documents/file1.txt",
            "/another/path/file2.md",
            "/yet/another/path/file3.json",
            "/some/directory/file4.pdf",
        ]

        files_per_batch, concurrent_batches = calculate_dynamic_batch_parameters(
            filepaths, mock_config
        )

        # 3 text files out of 4 = 75%
        assert files_per_batch == 64
        assert concurrent_batches == 4

    @patch("nvidia_rag.utils.batch_utils.calculate_text_like_batch_params", return_value=(64, 4))
    def test_relative_file_paths(self, mock_text_batch_params):
        """Test with relative file paths"""
        mock_config = MagicMock()
        mock_config.nv_ingest.enable_dynamic_batching = True
        mock_config.nv_ingest.files_per_batch = 50
        mock_config.nv_ingest.concurrent_batches = 3

        filepaths = [
            "./documents/file1.txt",
            "../other/file2.md",
            "relative/path/file3.html",
            "file4.pdf",
        ]

        files_per_batch, concurrent_batches = calculate_dynamic_batch_parameters(
            filepaths, mock_config
        )

        # 3 text files out of 4 = 75%
        assert files_per_batch == 64
        assert concurrent_batches == 4

    def test_mixed_extension_types(self):
        """Test with various extension types to verify text-like detection"""
        mock_config = MagicMock()
        mock_config.nv_ingest.enable_dynamic_batching = True
        mock_config.nv_ingest.files_per_batch = 50
        mock_config.nv_ingest.concurrent_batches = 3

        filepaths = [
            # Text-like (5 files)
            "file1.txt",
            "file2.md",
            "file3.json",
            "file4.html",
            "file5.sh",
            # Complex files (5 files)
            "file6.pdf",
            "file7.docx",
            "file8.pptx",
            "file9.jpg",
            "file10.png",
        ]

        files_per_batch, concurrent_batches = calculate_dynamic_batch_parameters(
            filepaths, mock_config
        )

        # Exactly 50% - should use default config
        assert files_per_batch == 50
        assert concurrent_batches == 3

    def test_extension_counts_logging(self, caplog):
        """Test that extension distribution is logged for debugging"""
        mock_config = MagicMock()
        mock_config.nv_ingest.enable_dynamic_batching = True
        mock_config.nv_ingest.files_per_batch = 50
        mock_config.nv_ingest.concurrent_batches = 3

        filepaths = [
            "file1.txt",
            "file2.txt",
            "file3.pdf",
            "file4.pdf",
            "file5.docx",
        ]

        with caplog.at_level("DEBUG"):
            calculate_dynamic_batch_parameters(filepaths, mock_config)

        # Check that extension distribution was logged
        assert any(
            "File distribution analysis" in record.message for record in caplog.records
        )
        assert any("Total=5" in record.message for record in caplog.records)

    def test_unknown_extensions_not_treated_as_text(self):
        """Test that unknown extensions are not treated as text-like"""
        mock_config = MagicMock()
        mock_config.nv_ingest.enable_dynamic_batching = True
        mock_config.nv_ingest.files_per_batch = 50
        mock_config.nv_ingest.concurrent_batches = 3

        filepaths = [
            "file1.xyz",
            "file2.abc",
            "file3.unknown",
            "file4.txt",  # Only this one is text-like
        ]

        files_per_batch, concurrent_batches = calculate_dynamic_batch_parameters(
            filepaths, mock_config
        )

        # 1 text file out of 4 = 25%
        assert files_per_batch == 50
        assert concurrent_batches == 3

    @patch("nvidia_rag.utils.batch_utils.calculate_text_like_batch_params", return_value=(64, 4))
    def test_large_batch_of_text_files(self, mock_text_batch_params):
        """Test with a large number of text-like files"""
        mock_config = MagicMock()
        mock_config.nv_ingest.enable_dynamic_batching = True
        mock_config.nv_ingest.files_per_batch = 50
        mock_config.nv_ingest.concurrent_batches = 3

        # Create 100 text files
        filepaths = [f"file{i}.txt" for i in range(100)]

        files_per_batch, concurrent_batches = calculate_dynamic_batch_parameters(
            filepaths, mock_config
        )

        assert files_per_batch == 64
        assert concurrent_batches == 4

    def test_large_batch_of_non_text_files(self):
        """Test with a large number of non-text files"""
        mock_config = MagicMock()
        mock_config.nv_ingest.enable_dynamic_batching = True
        mock_config.nv_ingest.files_per_batch = 50
        mock_config.nv_ingest.concurrent_batches = 3

        # Create 100 PDF files
        filepaths = [f"file{i}.pdf" for i in range(100)]

        files_per_batch, concurrent_batches = calculate_dynamic_batch_parameters(
            filepaths, mock_config
        )

        assert files_per_batch == 50
        assert concurrent_batches == 3

    @patch("nvidia_rag.utils.batch_utils.calculate_text_like_batch_params", return_value=(64, 4))
    def test_boundary_case_51_percent_text(self, mock_text_batch_params):
        """Test boundary case with just over 50% text files"""
        mock_config = MagicMock()
        mock_config.nv_ingest.enable_dynamic_batching = True
        mock_config.nv_ingest.files_per_batch = 50
        mock_config.nv_ingest.concurrent_batches = 3

        # 51 text files out of 100 = 51%
        filepaths = [f"file{i}.txt" for i in range(51)]
        filepaths.extend([f"file{i}.pdf" for i in range(51, 100)])

        files_per_batch, concurrent_batches = calculate_dynamic_batch_parameters(
            filepaths, mock_config
        )

        # Should optimize for text since > 50%
        assert files_per_batch == 64
        assert concurrent_batches == 4

    def test_boundary_case_49_percent_text(self):
        """Test boundary case with just under 50% text files"""
        mock_config = MagicMock()
        mock_config.nv_ingest.enable_dynamic_batching = True
        mock_config.nv_ingest.files_per_batch = 50
        mock_config.nv_ingest.concurrent_batches = 3

        # 49 text files out of 100 = 49%
        filepaths = [f"file{i}.txt" for i in range(49)]
        filepaths.extend([f"file{i}.pdf" for i in range(49, 100)])

        files_per_batch, concurrent_batches = calculate_dynamic_batch_parameters(
            filepaths, mock_config
        )

        # Should use default config since < 50%
        assert files_per_batch == 50
        assert concurrent_batches == 3

    @patch("nvidia_rag.utils.batch_utils.calculate_text_like_batch_params", return_value=(64, 4))
    def test_all_text_like_extension_types(self, mock_text_batch_params):
        """Test that all defined text-like extensions are recognized"""
        mock_config = MagicMock()
        mock_config.nv_ingest.enable_dynamic_batching = True
        mock_config.nv_ingest.files_per_batch = 50
        mock_config.nv_ingest.concurrent_batches = 3

        # All the text-like extensions defined in the function
        filepaths = [
            "file1.txt",
            "file2.md",
            "file3.json",
            "file4.sh",
            "file5.html",
        ]

        files_per_batch, concurrent_batches = calculate_dynamic_batch_parameters(
            filepaths, mock_config
        )

        # All should be recognized as text-like
        assert files_per_batch == 64
        assert concurrent_batches == 4

    def test_config_values_preserved_when_disabled(self):
        """Test that config values are returned unchanged when dynamic batching is disabled"""
        mock_config = MagicMock()
        mock_config.nv_ingest.enable_dynamic_batching = False
        mock_config.nv_ingest.files_per_batch = 123
        mock_config.nv_ingest.concurrent_batches = 456

        filepaths = ["file1.txt", "file2.txt"]

        files_per_batch, concurrent_batches = calculate_dynamic_batch_parameters(
            filepaths, mock_config
        )

        # Should return exactly the config values
        assert files_per_batch == 123
        assert concurrent_batches == 456

    def test_config_values_preserved_for_non_text_files(self):
        """Test that config values are preserved for non-text files"""
        mock_config = MagicMock()
        mock_config.nv_ingest.enable_dynamic_batching = True
        mock_config.nv_ingest.files_per_batch = 123
        mock_config.nv_ingest.concurrent_batches = 456

        filepaths = ["file1.pdf", "file2.docx"]

        files_per_batch, concurrent_batches = calculate_dynamic_batch_parameters(
            filepaths, mock_config
        )

        # Should return exactly the config values
        assert files_per_batch == 123
        assert concurrent_batches == 456


class TestCalculateTextLikeBatchParams:
    """Test calculate_text_like_batch_params function"""

    def _make_config(
        self,
        chunk_size=512,
        chunk_overlap=50,
        embedding_dimensions=1024,
        max_memory_budget_mb=4096,
    ):
        """Create a mock config with ingest and embeddings attributes."""
        mock_config = MagicMock()
        mock_config.nv_ingest.chunk_size = chunk_size
        mock_config.nv_ingest.chunk_overlap = chunk_overlap
        mock_config.nv_ingest.max_memory_budget_mb = max_memory_budget_mb
        mock_config.embeddings.dimensions = embedding_dimensions
        return mock_config

    @patch("nvidia_rag.utils.batch_utils.calculate_average_embedding_size")
    @patch("nvidia_rag.utils.batch_utils.calculate_average_file_size")
    def test_returns_largest_batch_size_when_memory_and_file_count_allow(
        self, mock_avg_file_size, mock_avg_embedding_size
    ):
        """When memory budget and file count are high, should return (250, 4)."""
        mock_avg_file_size.return_value = 1000
        mock_avg_embedding_size.return_value = 2000
        config = self._make_config(max_memory_budget_mb=8192)
        filepaths = [f"file{i}.txt" for i in range(1000)]

        files_per_batch, concurrent_batches = calculate_text_like_batch_params(
            filepaths, config
        )

        assert files_per_batch == 250
        assert concurrent_batches == 4

    @patch("nvidia_rag.utils.batch_utils.calculate_average_embedding_size")
    @patch("nvidia_rag.utils.batch_utils.calculate_average_file_size")
    def test_returns_smallest_batch_size_when_memory_constrained(
        self, mock_avg_file_size, mock_avg_embedding_size
    ):
        """When memory per file is high, max_concurrent_files is low; should return (16, 4)."""
        # memory_per_file = (1M + 1M) * 2 = 4M; max_concurrent = 1024*1024*1024 // 4M = 256
        # 250*4=1000 > 256, 128*4=512 > 256, 64*4=256 OK -> would return (64,4) if 256.
        # To get (16,4) we need max_concurrent_files < 64. So memory_per_file > 16*1024*1024.
        mock_avg_file_size.return_value = 20 * 1024 * 1024  # 20 MB
        mock_avg_embedding_size.return_value = 20 * 1024 * 1024
        config = self._make_config(max_memory_budget_mb=1024)
        filepaths = [f"file{i}.txt" for i in range(500)]

        files_per_batch, concurrent_batches = calculate_text_like_batch_params(
            filepaths, config
        )

        assert files_per_batch == 16
        assert concurrent_batches == 4

    @patch("nvidia_rag.utils.batch_utils.calculate_average_embedding_size")
    @patch("nvidia_rag.utils.batch_utils.calculate_average_file_size")
    def test_returns_middle_batch_size_when_memory_limits_concurrency(
        self, mock_avg_file_size, mock_avg_embedding_size
    ):
        """When max_concurrent_files is between 64 and 128, should return (32, 4) or (64, 4)."""
        # Target: max_concurrent_files ~ 200 so 64*4=256 > 200, 32*4=128 <= 200 -> (32, 4)
        mock_avg_file_size.return_value = 50000
        mock_avg_embedding_size.return_value = 50000
        # memory_per_file = 100000 * 2 = 200000; max_concurrent = 1024*1024*1024 // 200000 = 5242
        # That's too high. We need max_concurrent ~ 200: memory_per_file = 1024*1024*1024/200 ~ 5.2M
        mock_avg_file_size.return_value = 2 * 1024 * 1024
        mock_avg_embedding_size.return_value = 2 * 1024 * 1024
        # memory_per_file = 8M; max_concurrent = 1024*1024*1024 // 8M = 128
        config = self._make_config(max_memory_budget_mb=1024)
        filepaths = [f"file{i}.txt" for i in range(500)]

        files_per_batch, concurrent_batches = calculate_text_like_batch_params(
            filepaths, config
        )

        # 250*4=1000>128, 128*4=512>128, 64*4=256>128, 32*4=128<=128 -> (32, 4)
        assert files_per_batch == 32
        assert concurrent_batches == 4

    @patch("nvidia_rag.utils.batch_utils.calculate_average_embedding_size")
    @patch("nvidia_rag.utils.batch_utils.calculate_average_file_size")
    def test_file_count_limits_batch_size(
        self, mock_avg_file_size, mock_avg_embedding_size
    ):
        """When file count is small, batch_size * 4 must be <= len(filepaths); else fallback to (16, 4)."""
        mock_avg_file_size.return_value = 100
        mock_avg_embedding_size.return_value = 100
        config = self._make_config(max_memory_budget_mb=8192)
        filepaths = [f"file{i}.txt" for i in range(50)]  # 50 files

        # 250*4=1000>50, 128*4=512>50, 64*4=256>50, 32*4=128>50, 16*4=64>50 -> none pass len check
        files_per_batch, concurrent_batches = calculate_text_like_batch_params(
            filepaths, config
        )

        assert files_per_batch == 16
        assert concurrent_batches == 4

    @patch("nvidia_rag.utils.batch_utils.calculate_average_embedding_size")
    @patch("nvidia_rag.utils.batch_utils.calculate_average_file_size")
    def test_returns_64_when_file_count_and_memory_allow_64(
        self, mock_avg_file_size, mock_avg_embedding_size
    ):
        """Exactly 256 files and enough memory: 64*4=256 <= 256, should return (64, 4)."""
        mock_avg_file_size.return_value = 1000
        mock_avg_embedding_size.return_value = 1000
        config = self._make_config(max_memory_budget_mb=4096)
        filepaths = [f"file{i}.txt" for i in range(256)]

        files_per_batch, concurrent_batches = calculate_text_like_batch_params(
            filepaths, config
        )

        assert files_per_batch == 64
        assert concurrent_batches == 4

    @patch("nvidia_rag.utils.batch_utils.calculate_average_embedding_size")
    @patch("nvidia_rag.utils.batch_utils.calculate_average_file_size")
    def test_single_file_returns_minimum_batch_size(
        self, mock_avg_file_size, mock_avg_embedding_size
    ):
        """With one file, no batch size satisfies batch_size * 4 <= 1; returns (16, 4)."""
        mock_avg_file_size.return_value = 100
        mock_avg_embedding_size.return_value = 100
        config = self._make_config()
        filepaths = ["single.txt"]

        files_per_batch, concurrent_batches = calculate_text_like_batch_params(
            filepaths, config
        )

        assert files_per_batch == 16
        assert concurrent_batches == 4

    def test_empty_filepaths_raises_zero_division_error(self):
        """Empty filepaths causes divide-by-zero in average file size calculation."""
        config = self._make_config()
        filepaths = []

        with pytest.raises(ZeroDivisionError):
            calculate_text_like_batch_params(filepaths, config)

    @patch("nvidia_rag.utils.batch_utils.calculate_average_embedding_size")
    @patch("nvidia_rag.utils.batch_utils.calculate_average_file_size")
    def test_config_values_passed_to_embedding_size_calculation(
        self, mock_avg_file_size, mock_avg_embedding_size
    ):
        """Config chunk_size, chunk_overlap, and embeddings.dimensions are passed to helper."""
        mock_avg_file_size.return_value = 2000
        mock_avg_embedding_size.return_value = 3000
        config = self._make_config(
            chunk_size=256,
            chunk_overlap=25,
            embedding_dimensions=768,
        )
        filepaths = [f"file{i}.txt" for i in range(100)]

        calculate_text_like_batch_params(filepaths, config)

        mock_avg_embedding_size.assert_called_once()
        call_kwargs = mock_avg_embedding_size.call_args[1]
        assert call_kwargs["chunk_size"] == 256
        assert call_kwargs["chunk_overlap"] == 25
        assert call_kwargs["embedding_dimensions"] == 768
        assert call_kwargs["avg_file_size_bytes"] == 2000
        assert call_kwargs["num_files"] == 100

    @patch("nvidia_rag.utils.batch_utils.calculate_average_embedding_size")
    @patch("nvidia_rag.utils.batch_utils.calculate_average_file_size")
    def test_debug_logging_when_calculating_params(
        self, mock_avg_file_size, mock_avg_embedding_size, caplog
    ):
        """Debug log contains batch parameter details."""
        mock_avg_file_size.return_value = 1000
        mock_avg_embedding_size.return_value = 2000
        config = self._make_config()
        filepaths = [f"file{i}.txt" for i in range(100)]

        with caplog.at_level("DEBUG"):
            calculate_text_like_batch_params(filepaths, config)

        assert any(
            "Text-like file batching parameters" in record.message
            for record in caplog.records
        )
        assert any(
            "avg_file_size_bytes" in record.message for record in caplog.records
        )
        assert any(
            "max_concurrent_files" in record.message for record in caplog.records
        )

    @patch("nvidia_rag.utils.batch_utils.calculate_average_embedding_size")
    @patch("nvidia_rag.utils.batch_utils.calculate_average_file_size")
    def test_batch_size_128_returned_when_memory_allows(
        self, mock_avg_file_size, mock_avg_embedding_size
    ):
        """Verify batch size 128 is chosen when max_concurrent_files is in [512, 1000)."""
        mock_avg_file_size.return_value = 10000
        mock_avg_embedding_size.return_value = 10000
        # memory_per_file = 40000; 1024 MB -> 1024*1024*1024/40000 = 26214. 250*4=1000, 128*4=512.
        # Need max_concurrent between 512 and 1000. 700: memory = 1024*1024*1024/700 ~ 1.5M
        mock_avg_file_size.return_value = 400000
        mock_avg_embedding_size.return_value = 400000
        # memory_per_file = 1.6M; max_concurrent = 1024*1024*1024/1.6e6 = 655
        config = self._make_config(max_memory_budget_mb=1024)
        filepaths = [f"file{i}.txt" for i in range(1000)]

        files_per_batch, concurrent_batches = calculate_text_like_batch_params(
            filepaths, config
        )

        # 250*4=1000>655, 128*4=512<=655 -> (128, 4)
        assert files_per_batch == 128
        assert concurrent_batches == 4

    @patch("nvidia_rag.utils.batch_utils.os.path.getsize")
    def test_integration_with_real_file_size_and_embedding_calculation(
        self, mock_getsize
    ):
        """Integration: real average file size and embedding size formulas with mocked getsize."""
        mock_getsize.return_value = 4096  # 4 KB per file
        config = self._make_config(
            chunk_size=512,
            chunk_overlap=50,
            embedding_dimensions=1024,
            max_memory_budget_mb=1024,
        )
        filepaths = [f"file{i}.txt" for i in range(500)]

        files_per_batch, concurrent_batches = calculate_text_like_batch_params(
            filepaths, config
        )

        assert files_per_batch in (16, 32, 64, 128, 250)
        assert concurrent_batches == 4
