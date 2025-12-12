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

"""Opentelemetery Metrics"""

import logging

from opentelemetry import metrics


class OtelMetrics:
    """Encapsulates OpenTelemetry Metrics for API tracking."""

    def __init__(self, service_name: str = "rag"):
        self.service_name = service_name
        self.meter = metrics.get_meter(service_name)
        self._otlp_meter = None
        self._setup_metrics()

    def _create_instruments(self, meter):
        """Create all OpenTelemetry instruments for a given meter."""
        instruments = {}

        # API request counter
        instruments["api_request_counter"] = meter.create_counter(
            "api_requests_total", description="Total API requests"
        )

        # Token gauges
        instruments["input_token_gauge"] = meter.create_gauge(
            "input_tokens", description="Number of input tokens processed"
        )
        instruments["output_token_gauge"] = meter.create_gauge(
            "output_tokens", description="Number of output tokens generated"
        )
        instruments["total_token_gauge"] = meter.create_gauge(
            "total_tokens", description="Total tokens (input + output)"
        )
        instruments["avg_words_per_chunk_gauge"] = meter.create_gauge(
            "avg_words_per_chunk", description="Avg words per chunk in context"
        )

        # Token usage histogram
        instruments["token_usage_histogram"] = meter.create_histogram(
            "token_usage_distribution",
            description="Token usage distribution per request",
        )

        # Latency histograms
        instruments["latency_hists"] = {
            "rag_ttft_ms": meter.create_histogram(
                "rag_ttft_ms", description="RAG time-to-first-token latency"
            ),
            "llm_ttft_ms": meter.create_histogram(
                "llm_ttft_ms", description="LLM time-to-first-token latency"
            ),
            "context_reranker_time_ms": meter.create_histogram(
                "context_reranker_time_ms", description="Context reranker latency"
            ),
            "retrieval_time_ms": meter.create_histogram(
                "retrieval_time_ms", description="Document retrieval latency"
            ),
            "llm_generation_time_ms": meter.create_histogram(
                "llm_generation_time_ms", description="LLM generation latency"
            ),
        }

        return instruments

    def _setup_metrics(self):
        """Initializes the OpenTelemetry metrics. Avoid duplicate instrument creation."""
        # Guard against re-initialization in the same process
        if hasattr(self, "api_request_counter"):
            logging.info(
                "OpenTelemetry Metrics already initialized; skipping re-creation"
            )
            return

        # Create instruments for the main meter
        instruments = self._create_instruments(self.meter)

        # Assign to instance variables
        self.api_request_counter = instruments["api_request_counter"]
        self.input_token_gauge = instruments["input_token_gauge"]
        self.output_token_gauge = instruments["output_token_gauge"]
        self.total_token_gauge = instruments["total_token_gauge"]
        self.avg_words_per_chunk_gauge = instruments["avg_words_per_chunk_gauge"]
        self.token_usage_histogram = instruments["token_usage_histogram"]
        self.latency_hists = instruments["latency_hists"]

        logging.info("OpenTelemetry Metrics Initialized")

    def setup_otlp_meter(self, otlp_provider):
        """Set up OTLP meter for dual export"""
        if otlp_provider is None:
            logging.warning("OTLP provider is None, skipping OTLP meter setup")
            return

        try:
            self._otlp_meter = otlp_provider.get_meter(self.service_name)

            # Create OTLP instruments using the same unified function
            otlp_instruments = self._create_instruments(self._otlp_meter)

            # Cache OTLP instruments for better performance
            self._otlp_api_request_counter = otlp_instruments["api_request_counter"]
            self._otlp_input_token_gauge = otlp_instruments["input_token_gauge"]
            self._otlp_output_token_gauge = otlp_instruments["output_token_gauge"]
            self._otlp_total_token_gauge = otlp_instruments["total_token_gauge"]
            self._otlp_token_usage_histogram = otlp_instruments["token_usage_histogram"]
            self._otlp_avg_words_per_chunk_gauge = otlp_instruments[
                "avg_words_per_chunk_gauge"
            ]
            self._otlp_latency_hists = otlp_instruments["latency_hists"]

            logging.info(
                "OTLP meter configured for dual export with cached instruments"
            )
        except Exception as e:
            logging.error(f"Failed to setup OTLP meter: {e}")
            self._otlp_meter = None

    def update_api_requests(self, method: str = None, endpoint: str = None):
        """Updates the API request counter."""
        if method and endpoint:
            self.api_request_counter.add(1, {"method": method, "endpoint": endpoint})
            logging.info(f"API Request Tracked: {method} {endpoint}")

            # Also update OTLP meter if available (using cached counter)
            if hasattr(self, "_otlp_api_request_counter"):
                self._otlp_api_request_counter.add(
                    1, {"method": method, "endpoint": endpoint}
                )

    def update_llm_tokens(self, input_t: int = None, output_t: int = None):
        """Updates the token-related metrics."""
        if input_t is not None and output_t is not None:
            total_t = input_t + output_t
            self.input_token_gauge.set(input_t)
            self.output_token_gauge.set(output_t)
            self.total_token_gauge.set(total_t)
            self.token_usage_histogram.record(total_t)
            logging.info(
                f"Token Usage - Input: {input_t}, Output: {output_t}, Total: {total_t}"
            )

            # Also update OTLP meter if available (using cached instruments)
            if hasattr(self, "_otlp_input_token_gauge"):
                self._otlp_input_token_gauge.set(input_t)
                self._otlp_output_token_gauge.set(output_t)
                self._otlp_total_token_gauge.set(total_t)
                self._otlp_token_usage_histogram.record(total_t)

    def update_avg_words_per_chunk(self, avg_words_per_chunk: int = None):
        """Updates chunk related metrics"""
        if avg_words_per_chunk is not None:
            self.avg_words_per_chunk_gauge.set(avg_words_per_chunk)
            logging.info(f"Avg words per chunk: {avg_words_per_chunk}")

            # Also update OTLP meter if available (using cached gauge)
            if hasattr(self, "_otlp_avg_words_per_chunk_gauge"):
                self._otlp_avg_words_per_chunk_gauge.set(avg_words_per_chunk)

    # ------------------- Latency metrics -------------------

    def update_latency_metrics(self, metrics: dict[str, float]):
        """Record latency metrics into histograms (ms)."""
        for name, value in metrics.items():
            hist = self.latency_hists.get(name)
            if hist and value is not None:
                hist.record(value)

                # Also update OTLP meter if available (using cached histogram)
                if (
                    hasattr(self, "_otlp_latency_hists")
                    and name in self._otlp_latency_hists
                ):
                    self._otlp_latency_hists[name].record(value)


# Singleton factory to reuse a single OtelMetrics instance per process
_singleton_metrics: OtelMetrics | None = None


def get_otel_metrics(service_name: str = "rag") -> OtelMetrics:
    global _singleton_metrics
    if _singleton_metrics is None:
        _singleton_metrics = OtelMetrics(service_name)
    return _singleton_metrics
