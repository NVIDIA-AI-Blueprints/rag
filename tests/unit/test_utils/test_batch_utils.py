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

from unittest.mock import MagicMock

import pytest

from nvidia_rag.utils.batch_utils import calculate_dynamic_batch_parameters


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

    def test_all_text_like_files(self, caplog):
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

        assert files_per_batch == 200
        assert concurrent_batches == 1
        assert any(
            "100.0% text-like files" in record.message for record in caplog.records
        )
        assert any(
            "optimized parameters for text processing" in record.message
            for record in caplog.records
        )

    def test_majority_text_like_files(self, caplog):
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

        assert files_per_batch == 200
        assert concurrent_batches == 1
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

    def test_case_insensitive_extension_handling(self):
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
        assert files_per_batch == 200
        assert concurrent_batches == 1

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

    def test_files_with_multiple_dots(self):
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
        assert files_per_batch == 200
        assert concurrent_batches == 1

    def test_single_file_text(self):
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
        assert files_per_batch == 200
        assert concurrent_batches == 1

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

    def test_full_file_paths(self):
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
        assert files_per_batch == 200
        assert concurrent_batches == 1

    def test_relative_file_paths(self):
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
        assert files_per_batch == 200
        assert concurrent_batches == 1

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

    def test_large_batch_of_text_files(self):
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

        assert files_per_batch == 200
        assert concurrent_batches == 1

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

    def test_boundary_case_51_percent_text(self):
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
        assert files_per_batch == 200
        assert concurrent_batches == 1

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

    def test_all_text_like_extension_types(self):
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
        assert files_per_batch == 200
        assert concurrent_batches == 1

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
