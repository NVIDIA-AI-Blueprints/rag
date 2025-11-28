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

import base64
import io
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from PIL import Image as PILImage

from nvidia_rag.rag_server.vlm import VLM


class TestVLM:
    """Unit tests for the updated VLM helper."""

    def setup_method(self):
        self.vlm_model = "test-model"
        self.vlm_endpoint = "http://test-endpoint.com"
        self.mock_config = Mock()
        self.prompts_patcher = patch(
            "nvidia_rag.rag_server.vlm.get_prompts",
            return_value={
                "vlm_template": {
                    "system": "You are a helpful assistant.",
                    "human": "{context}\n\n{question}",
                }
            },
        )
        self.prompts_patcher.start()
        self.vlm = VLM(self.vlm_model, self.vlm_endpoint, config=self.mock_config)

    def teardown_method(self):
        patch.stopall()

    @staticmethod
    def create_test_image_b64(color: str = "red") -> str:
        img = PILImage.new("RGB", (32, 32), color=color)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def test_vlm_init_success(self):
        assert self.vlm.model_name == self.vlm_model
        assert self.vlm.invoke_url == self.vlm_endpoint
        assert self.vlm.vlm_template["system"] == "You are a helpful assistant."

    def test_vlm_init_missing_url(self):
        with pytest.raises(
            OSError, match="VLM server URL and model name must be set in the environment"
        ):
            VLM(self.vlm_model, "", config=self.mock_config)

    def test_vlm_init_missing_model(self):
        with pytest.raises(
            OSError, match="VLM server URL and model name must be set in the environment"
        ):
            VLM("", self.vlm_endpoint, config=self.mock_config)

    def test_normalize_messages_converts_images_and_accumulates_system_text(self):
        with patch.object(VLM, "_convert_image_url_to_png_b64", return_value="converted"):
            messages, last_idx, system_text = VLM._normalize_messages(
                [
                    {"role": "system", "content": "sys notice"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "hello"},
                            {"type": "image_url", "image_url": {"url": "http://img"}},
                        ],
                    },
                    {"role": "assistant", "content": "hi"},
                ]
            )

        assert isinstance(messages[0], HumanMessage)
        assert last_idx == 0
        assert system_text == "sys notice"
        image_part = messages[0].content[1]
        assert image_part["type"] == "image_url"
        assert image_part["image_url"]["url"] == "data:image/png;base64,converted"

    def test_extract_and_process_messages_attaches_doc_images(self):
        mock_minio = MagicMock()
        mock_minio.get_payload.return_value = {"content": self.create_test_image_b64()}
        doc = SimpleNamespace(
            metadata={
                "content_metadata": {
                    "type": "image",
                    "page_number": 1,
                    "location": [0, 0, 1, 1],
                },
                "collection_name": "demo",
                "source": {"source_id": "sample.pdf"},
            },
            page_content="ignored",
        )
        with (
            patch("nvidia_rag.rag_server.vlm.get_minio_operator", return_value=mock_minio),
            patch(
                "nvidia_rag.rag_server.vlm.get_unique_thumbnail_id",
                return_value="thumb-id",
            ),
        ):
            system_msg, user_msg, history = self.vlm.extract_and_process_messages(
                self.vlm.vlm_template,
                [doc],
                [{"role": "user", "content": "Hi"}],
                context_text=None,
                question_text="Question?",
                max_total_images=4,
            )

        assert isinstance(system_msg, SystemMessage)
        assert history
        assert len(user_msg.content) == 2
        assert user_msg.content[1]["type"] == "image_url"

    def test_extract_and_process_messages_respects_image_budget(self):
        existing_image = {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{self.create_test_image_b64()}"},
                },
            ],
        }
        system_msg, user_msg, _ = self.vlm.extract_and_process_messages(
            self.vlm.vlm_template,
            docs=[],
            incoming_messages=[existing_image],
            context_text="ctx",
            question_text="question",
            max_total_images=1,
        )

        assert isinstance(system_msg, SystemMessage)
        assert len(user_msg.content) == 1  # only text; no room for doc images

    def test_analyze_with_messages_invokes_model(self):
        system_message = SystemMessage(content="sys")
        user_message = HumanMessage(content=[{"type": "text", "text": "ctx"}])
        history = [HumanMessage(content=[{"type": "text", "text": "prev"}])]

        with (
            patch.object(VLM, "init_model") as mock_init_model,
            patch.object(
                VLM,
                "extract_and_process_messages",
                return_value=(system_message, user_message, history),
            ),
            patch.object(
                VLM, "assemble_messages", return_value=[system_message, user_message]
            ) as mock_assemble,
            patch.object(VLM, "_redact_messages_for_logging"),
            patch.object(VLM, "invoke_model", return_value="final-response") as mock_invoke,
        ):
            response = self.vlm.analyze_with_messages(
                docs=[],
                messages=[{"role": "user", "content": "question"}],
                temperature=0.2,
                top_p=0.9,
                max_tokens=128,
            )

        assert response == "final-response"
        mock_init_model.assert_called_once()
        mock_assemble.assert_called_once()
        mock_invoke.assert_called_once_with(
            mock_init_model.return_value,
            [system_message, user_message],
            temperature=0.2,
            top_p=0.9,
            max_tokens=128,
        )

    def test_analyze_with_messages_returns_empty_without_messages(self):
        with patch.object(VLM, "init_model") as mock_init_model:
            response = self.vlm.analyze_with_messages(docs=[], messages=[])

        assert response == ""
        mock_init_model.assert_not_called()

    def test_analyze_with_messages_logs_exception_and_returns_empty(self):
        system_message = SystemMessage(content="sys")
        user_message = HumanMessage(content=[{"type": "text", "text": "ctx"}])

        with (
            patch.object(VLM, "init_model"),
            patch.object(
                VLM,
                "extract_and_process_messages",
                return_value=(system_message, user_message, []),
            ),
            patch.object(VLM, "assemble_messages", return_value=[system_message, user_message]),
            patch.object(VLM, "_redact_messages_for_logging"),
            patch.object(VLM, "invoke_model", side_effect=RuntimeError("boom")),
        ):
            response = self.vlm.analyze_with_messages(
                docs=[], messages=[{"role": "user", "content": "hi"}]
            )

        assert response == ""

    def test_stream_with_messages_yields_chunks(self):
        system_message = SystemMessage(content="sys")
        user_message = HumanMessage(content=[{"type": "text", "text": "ctx"}])
        mock_model = Mock()
        mock_model.stream.return_value = [
            SimpleNamespace(content="Hello"),
            SimpleNamespace(content=""),
            SimpleNamespace(content="World"),
        ]

        with (
            patch.object(VLM, "init_model", return_value=mock_model),
            patch.object(
                VLM,
                "extract_and_process_messages",
                return_value=(system_message, user_message, []),
            ),
            patch.object(VLM, "assemble_messages", return_value=[system_message, user_message]),
            patch.object(VLM, "_redact_messages_for_logging"),
        ):
            chunks = list(
                self.vlm.stream_with_messages(
                    docs=[], messages=[{"role": "user", "content": "hi"}], temperature=0.1
                )
            )

        assert chunks == ["Hello", "World"]

    def test_convert_image_url_to_png_b64_data_url(self):
        test_image = self.create_test_image_b64()
        data_url = f"data:image/jpeg;base64,{test_image}"
        result = self.vlm._convert_image_url_to_png_b64(data_url)
        assert isinstance(result, str)
        assert result.startswith("iVBOR")

    def test_convert_image_url_to_png_b64_invalid_input_returns_original(self):
        invalid = "data:image/jpeg;invalid,"
        result = self.vlm._convert_image_url_to_png_b64(invalid)
        assert result == invalid

    def test_redact_messages_for_logging_masks_base64(self):
        messages = [
            SystemMessage(content="sys"),
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{self.create_test_image_b64()}"
                        },
                    }
                ]
            ),
            AIMessage(content="done"),
        ]

        redacted = self.vlm._redact_messages_for_logging(messages)
        assert redacted[1]["content"][0]["image_url"]["url"].endswith("[REDACTED]")

    def test_format_docs_text_includes_filename_and_content(self):
        doc = SimpleNamespace(
            metadata={
                "content_metadata": {"type": "text"},
                "source": {"source_name": "/tmp/foo.txt"},
            },
            page_content="Important text",
        )
        formatted = self.vlm._format_docs_text([doc])
        assert "File: foo" in formatted
        assert "Important text" in formatted
