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
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from PIL import Image as PILImage

from nvidia_rag.rag_server.vlm import VLM


class TestVLM:
    """Unit tests for the VLM helper (native OpenAI SDK implementation)."""

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
            OSError,
            match="VLM server URL and model name must be set in the environment",
        ):
            VLM(self.vlm_model, "", config=self.mock_config)

    def test_vlm_init_missing_model(self):
        with pytest.raises(
            OSError,
            match="VLM server URL and model name must be set in the environment",
        ):
            VLM("", self.vlm_endpoint, config=self.mock_config)

    def test_build_extra_body_includes_thinking_budget_when_enabled(self):
        body = VLM._build_extra_body(enable_thinking=True, thinking_token_budget=128)
        assert body == {
            "chat_template_kwargs": {"enable_thinking": True},
            "thinking_token_budget": 128,
        }

    def test_build_extra_body_omits_thinking_budget_when_disabled(self):
        body = VLM._build_extra_body(enable_thinking=False, thinking_token_budget=512)
        assert body == {"chat_template_kwargs": {"enable_thinking": False}}

    def test_build_extra_body_omits_zero_budget(self):
        body = VLM._build_extra_body(enable_thinking=True, thinking_token_budget=0)
        assert body == {"chat_template_kwargs": {"enable_thinking": True}}

    def test_normalize_messages_converts_images_and_accumulates_system_text(self):
        with patch.object(
            VLM, "_convert_image_url_to_png_b64", return_value="converted"
        ):
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

        assert messages[0]["role"] == "user"
        assert last_idx == 0
        assert system_text == "sys notice"
        image_part = messages[0]["content"][1]
        assert image_part["type"] == "image_url"
        assert image_part["image_url"]["url"] == "data:image/png;base64,converted"
        assert messages[1] == {"role": "assistant", "content": "hi"}

    def test_extract_and_process_messages_attaches_doc_images(self):
        mock_object_store = MagicMock()
        b64_img = self.create_test_image_b64()
        mock_object_store.get_object_from_uri.return_value = base64.b64decode(b64_img)
        doc = SimpleNamespace(
            metadata={
                "content_metadata": {
                    "type": "image",
                    "page_number": 1,
                    "location": [0, 0, 1, 1],
                },
                "collection_name": "demo",
                "source": {
                    "source_id": "sample.pdf",
                    "source_location": "s3://default-bucket/demo/artifacts/page.png",
                },
            },
            page_content="ignored",
        )
        with patch(
            "nvidia_rag.rag_server.vlm.get_object_store_operator",
            return_value=mock_object_store,
        ):
            system_msg, user_msg, history = self.vlm.extract_and_process_messages(
                self.vlm.vlm_template,
                [doc],
                [{"role": "user", "content": "Hi"}],
                context_text=None,
                question_text="Question?",
                max_total_images=4,
            )

        assert system_msg["role"] == "system"
        assert history
        assert user_msg["role"] == "user"
        assert len(user_msg["content"]) == 2
        assert user_msg["content"][1]["type"] == "image_url"

    def test_extract_and_process_messages_respects_image_budget(self):
        existing_image = {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{self.create_test_image_b64()}"
                    },
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

        assert system_msg["role"] == "system"
        assert len(user_msg["content"]) == 1  # only text; no room for doc images

    @pytest.mark.asyncio
    async def test_analyze_with_messages_invokes_model(self):
        system_message = {"role": "system", "content": "sys"}
        user_message = {"role": "user", "content": [{"type": "text", "text": "ctx"}]}
        history = [{"role": "user", "content": [{"type": "text", "text": "prev"}]}]

        with (
            patch.object(VLM, "_create_async_client") as mock_create_client,
            patch.object(
                VLM,
                "extract_and_process_messages",
                return_value=(system_message, user_message, history),
            ),
            patch.object(
                VLM,
                "assemble_messages",
                return_value=[system_message, user_message],
            ) as mock_assemble,
            patch.object(VLM, "_redact_messages_for_logging"),
            patch.object(
                VLM,
                "invoke_model_async",
                new_callable=AsyncMock,
                return_value="final-response",
            ) as mock_invoke,
        ):
            self.mock_config.vlm.enable_thinking = True
            self.mock_config.vlm.thinking_token_budget = 0
            self.mock_config.vlm.get_api_key.return_value = None

            response = await self.vlm.analyze_with_messages(
                docs=[],
                messages=[{"role": "user", "content": "question"}],
                temperature=0.2,
                top_p=0.9,
                max_tokens=128,
            )

        assert response == "final-response"
        mock_create_client.assert_called_once()
        mock_assemble.assert_called_once()
        mock_invoke.assert_called_once_with(
            mock_create_client.return_value,
            self.vlm_model,
            [system_message, user_message],
            temperature=0.2,
            top_p=0.9,
            max_tokens=128,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )

    @pytest.mark.asyncio
    async def test_analyze_with_messages_returns_empty_without_messages(self):
        with patch.object(VLM, "_create_async_client") as mock_create_client:
            response = await self.vlm.analyze_with_messages(docs=[], messages=[])

        assert response == ""
        mock_create_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_analyze_with_messages_logs_exception_and_returns_empty(self):
        system_message = {"role": "system", "content": "sys"}
        user_message = {"role": "user", "content": [{"type": "text", "text": "ctx"}]}

        with (
            patch.object(VLM, "_create_async_client"),
            patch.object(
                VLM,
                "extract_and_process_messages",
                return_value=(system_message, user_message, []),
            ),
            patch.object(
                VLM,
                "assemble_messages",
                return_value=[system_message, user_message],
            ),
            patch.object(VLM, "_redact_messages_for_logging"),
            patch.object(
                VLM,
                "invoke_model_async",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            ),
        ):
            self.mock_config.vlm.enable_thinking = False
            self.mock_config.vlm.thinking_token_budget = 0
            self.mock_config.vlm.get_api_key.return_value = None

            response = await self.vlm.analyze_with_messages(
                docs=[], messages=[{"role": "user", "content": "hi"}]
            )

        assert response == ""

    @pytest.mark.asyncio
    async def test_invoke_model_async_returns_stripped_content(self):
        mock_client = MagicMock()
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="  hello  "))]
        )
        mock_client.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(return_value=completion),
            )
        )

        result = await VLM.invoke_model_async(
            mock_client,
            "test-model",
            [{"role": "user", "content": "hi"}],
            temperature=0.1,
            top_p=0.9,
            max_tokens=64,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )

        assert result == "hello"
        mock_client.chat.completions.create.assert_awaited_once_with(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.1,
            top_p=0.9,
            max_tokens=64,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )

    @pytest.mark.asyncio
    async def test_stream_with_messages_preserves_reasoning_when_filter_enabled(
        self,
    ):
        system_message = {"role": "system", "content": "sys"}
        user_message = {"role": "user", "content": [{"type": "text", "text": "ctx"}]}

        async def fake_stream():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=None, reasoning="thinking…")
                    )
                ]
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="Hello", reasoning=None)
                    )
                ]
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content="", reasoning=None))
                ]
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="World", reasoning=None)
                    )
                ]
            )

        mock_client = MagicMock()
        mock_client.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(return_value=fake_stream()),
            )
        )

        with (
            patch.object(VLM, "_create_async_client", return_value=mock_client),
            patch.object(
                VLM,
                "extract_and_process_messages",
                return_value=(system_message, user_message, []),
            ),
            patch.object(
                VLM,
                "assemble_messages",
                return_value=[system_message, user_message],
            ),
            patch.object(VLM, "_redact_messages_for_logging"),
        ):
            self.mock_config.vlm.enable_thinking = True
            self.mock_config.vlm.thinking_token_budget = 0
            self.mock_config.vlm.filter_think_tokens = True
            self.mock_config.vlm.get_api_key.return_value = None

            chunks = []
            async for chunk in self.vlm.stream_with_messages(
                docs=[],
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.1,
            ):
                chunks.append(chunk)

        assert [
            (
                chunk.content,
                chunk.additional_kwargs.get("reasoning_content"),
            )
            for chunk in chunks
        ] == [("", "thinking…"), ("Hello", None), ("World", None)]

    @pytest.mark.asyncio
    async def test_stream_with_messages_uses_reasoning_content_field(
        self,
    ):
        system_message = {"role": "system", "content": "sys"}
        user_message = {"role": "user", "content": [{"type": "text", "text": "ctx"}]}

        async def fake_stream():
            # Use reasoning_content (alternate field name) to confirm fallback.
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=None, reasoning_content="step1")
                    )
                ]
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="answer", reasoning_content=None)
                    )
                ]
            )

        mock_client = MagicMock()
        mock_client.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=AsyncMock(return_value=fake_stream()),
            )
        )

        with (
            patch.object(VLM, "_create_async_client", return_value=mock_client),
            patch.object(
                VLM,
                "extract_and_process_messages",
                return_value=(system_message, user_message, []),
            ),
            patch.object(
                VLM,
                "assemble_messages",
                return_value=[system_message, user_message],
            ),
            patch.object(VLM, "_redact_messages_for_logging"),
        ):
            self.mock_config.vlm.enable_thinking = True
            self.mock_config.vlm.thinking_token_budget = 0
            self.mock_config.vlm.filter_think_tokens = False
            self.mock_config.vlm.get_api_key.return_value = None

            chunks = []
            async for chunk in self.vlm.stream_with_messages(
                docs=[],
                messages=[{"role": "user", "content": "hi"}],
            ):
                chunks.append(chunk)

        assert [
            (
                chunk.content,
                chunk.additional_kwargs.get("reasoning_content"),
            )
            for chunk in chunks
        ] == [("", "step1"), ("answer", None)]

    @pytest.mark.asyncio
    async def test_stream_with_messages_returns_early_without_messages(self):
        with patch.object(VLM, "_create_async_client") as mock_create_client:
            chunks = []
            async for chunk in self.vlm.stream_with_messages(docs=[], messages=[]):
                chunks.append(chunk)

        assert chunks == []
        mock_create_client.assert_not_called()

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
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{self.create_test_image_b64()}"
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "done"},
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

    def test_extract_images_from_docs_nrl_fetches_stored_uri_and_returns_png_parts(
        self,
    ):
        mock_object_store = MagicMock()
        b64_img = self.create_test_image_b64()
        mock_object_store.get_object_from_uri.return_value = base64.b64decode(b64_img)
        doc = SimpleNamespace(
            metadata={
                "stored_image_uri": "s3://default-bucket/collection/page.png",
            },
            page_content="chunk text",
        )
        with patch(
            "nvidia_rag.rag_server.vlm.get_object_store_operator",
            return_value=mock_object_store,
        ):
            parts = self.vlm._extract_images_from_docs_nrl(
                [doc], remaining_image_budget=None
            )

        mock_object_store.get_object_from_uri.assert_called_once_with(
            "s3://default-bucket/collection/page.png"
        )
        assert len(parts) == 1
        assert parts[0]["type"] == "image_url"
        assert parts[0]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_extract_images_from_docs_nrl_skips_without_stored_image_uri(self):
        doc = SimpleNamespace(metadata={}, page_content="text only")
        with patch(
            "nvidia_rag.rag_server.vlm.get_object_store_operator"
        ) as mock_get_object_store:
            parts = self.vlm._extract_images_from_docs_nrl(
                [doc], remaining_image_budget=5
            )

        assert parts == []
        mock_get_object_store.return_value.get_object_from_uri.assert_not_called()

    def test_extract_images_from_docs_nrl_respects_remaining_image_budget(self):
        mock_object_store = MagicMock()
        b64_img = self.create_test_image_b64()
        raw_png = base64.b64decode(b64_img)
        mock_object_store.get_object_from_uri.return_value = raw_png
        docs = [
            SimpleNamespace(
                metadata={"stored_image_uri": "s3://b/a/1.png"},
                page_content="a",
            ),
            SimpleNamespace(
                metadata={"stored_image_uri": "s3://b/a/2.png"},
                page_content="b",
            ),
        ]
        with patch(
            "nvidia_rag.rag_server.vlm.get_object_store_operator",
            return_value=mock_object_store,
        ):
            parts = self.vlm._extract_images_from_docs_nrl(
                docs, remaining_image_budget=1
            )

        assert len(parts) == 1
        mock_object_store.get_object_from_uri.assert_called_once_with("s3://b/a/1.png")

    def test_extract_images_from_docs_nrl_continues_on_object_store_error(self):
        mock_object_store = MagicMock()
        b64_img = self.create_test_image_b64()
        good_raw = base64.b64decode(b64_img)
        docs = [
            SimpleNamespace(
                metadata={"stored_image_uri": "s3://b/bad.png"},
                page_content="x",
            ),
            SimpleNamespace(
                metadata={"stored_image_uri": "s3://b/good.png"},
                page_content="y",
            ),
        ]
        mock_object_store.get_object_from_uri.side_effect = [
            RuntimeError("unavailable"),
            good_raw,
        ]
        with patch(
            "nvidia_rag.rag_server.vlm.get_object_store_operator",
            return_value=mock_object_store,
        ):
            parts = self.vlm._extract_images_from_docs_nrl(
                docs, remaining_image_budget=None
            )

        assert len(parts) == 1
        assert parts[0]["type"] == "image_url"
