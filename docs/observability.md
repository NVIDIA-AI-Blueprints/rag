<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
# Observability Setup for NVIDIA RAG Blueprint

This guide provides step-by-step instructions to enable tracing and observability for the [NVIDIA RAG Blueprint](readme.md) using OpenTelemetry (OTel) Collector and Zipkin.


The observability stack consists of:
- **OTel Collector** - Collects, processes, and exports telemetry data.
- **Zipkin** - Used for **visualizing traces**.


## Enable Observability with Docker

Use the following procedure to enable observability with Docker.

1. Set the required environment variable for the OTel Collector Config by running the following code from the root directory of the repo.

    ```sh
    export OPENTELEMETRY_CONFIG_FILE=$(pwd)/deploy/config/otel-collector-config.yaml
    ```

2. Start the OTel Collector and Zipkin observability services by running the following code.

    ```sh
    docker compose -f deploy/compose/observability.yaml up -d
    ```

3. Enable tracing in the RAG server by setting `APP_TRACING_ENABLED` is set to `"True"` in `docker-compose-rag-server.yaml`.

    ```yaml
    services:
    rag-server:
        environment:
        # Tracing
        APP_TRACING_ENABLED: "True"
    ```

4. Start the RAG Server by following the instructions in the appropriate [deployment guide](readme.md#deployment-options-for-rag-blueprint).


## View Traces in Zipkin

After tracing is enabled and the system is running, you can **view the traces** in **Zipkin** by opening:

<p align="center">
<img src="assets/zipkin_ui.png" width="750">
</p>

Open the Zipkin UI at: **http://localhost:9411**



## View Metrics in Grafana

As part of the tracing, the RAG service also exports metrics like API request counts, LLM prompt and completion token count and words per chunk.

These metrics are exposed on the metrics endpoint exposed by Otel collector at **http://localhost:8889/metrics**

You can open Grafana UI and visualize these metrics on a dashboard by selecting data source as Prometheus and putting prometheus URL as **http://prometheus:9090**

Open the Grafana UI at **http://localhost:3000**


### Create a Dashboard in Grafana

To create a dashboard in [Grafana](https://grafana.com/) use the following procedure.

1. Navigate to the Grafana UI at `http://localhost:3000`.

2. Log in with the default credentials (`admin`/`admin`).

3. Go to the **Dashboards** section and click **Import**.

4. Upload the JSON file located in the `deploy/config` directory.

5. Select the data source for the dashboard. Ensure that the data source is correctly configured to pull metrics from your Prometheus instance.

6. Save the dashboard.

7. View your metrics and traces.


## Query-to-Answer Pipeline and Studying Time Spent

For a complete description of the flow from query to answer, and guidance on measuring time spent in each stage of the pipeline, refer to [Query-to-Answer Pipeline](query-to-answer-pipeline.md). 

The documentation explains the following:

- **Pipeline stages** – Describes the sequence of stages (query rewriter → retriever → context reranker → LLM generation) and the role of each one.
- **Studying time** – Shows how to use Zipkin span durations and Prometheus/Grafana metrics (`retrieval_time_ms`, `context_reranker_time_ms`, `llm_generation_time_ms`, `rag_ttft_ms`, etc.) to identify where latency occurs.

Each request trace in Zipkin includes spans such as `query-rewriter`, `retriever`, `context-reranker`, and `llm-stream`. Each span duration is the time spent in that stage. Metrics are exposed at **http://localhost:8889/metrics** and can be visualized in Grafana.


## Viewing Inputs / Outputs of each stage of the RAG pipeline using Zipkin

After tracing is enabled and running, you can view inputs and outputs of different stages of the RAG pipeline in [Zipkin](https://zipkin.io/).

1. Click on any of the workflows out of `query-rewriter`, `retriver`, `context-reranker` or `llm-stream`. Details appear in the details pane.

2. In the details, find the `traceloop.entity.input` and `traceloop.entity.ouput` rows. These rows show the input and output of that particular workflow.

3. Similarly, you can view inputs and outputs for sub stages within the workflows by clicking on a substage and finding the `traceloop.entity.input` and `traceloop.entity.ouput` rows.

  <p align="center">
  <img src="./assets/zipkin_ui_labelled.png" width="750">
  </p>



## Enable Observability with Helm

Use the following procedure to enable observability with Helm.

### Prerequisites: Install Prometheus Operator CRDs

Before enabling the observability stack, install the Prometheus Operator CRDs:

```bash
# Add the Prometheus Helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus Operator CRDs
helm upgrade --install prometheus-crds prometheus-community/prometheus-operator-crds \
  --version 26.0.1 \
  --namespace rag \
  --create-namespace
```

:::{note}
The Prometheus Operator CRDs must be installed before deploying the RAG blueprint with observability enabled. These CRDs are required for ServiceMonitor and other Prometheus resources.
:::

### Enable OpenTelemetry Collector, Zipkin and Prometheus stack

The observability stack is **disabled by default** to minimize resource usage. To enable it:

1. **Install Prometheus Operator CRDs** (if not already installed - see Prerequisites above)

2. Modify [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml) to enable the observability components:

   ```yaml
   # Enable observability stack
   serviceMonitor:
     enabled: true

   opentelemetry-collector:
     enabled: true

   zipkin:
     enabled: true

   kube-prometheus-stack:
     enabled: true

   # Enable tracing in rag-server
   envVars:
     # === Tracing ===
     APP_TRACING_ENABLED: "True"  # Change from "False" to "True"
     APP_TRACING_OTLPHTTPENDPOINT: "http://rag-opentelemetry-collector:4318/v1/traces"
     APP_TRACING_OTLPGRPCENDPOINT: "grpc://rag-opentelemetry-collector:4317"
   ```

3. Deploy the changes:

   After modifying [`values.yaml`](../deploy/helm/nvidia-blueprint-rag/values.yaml), apply the changes as described in [Change a Deployment](deploy-helm.md#change-a-deployment).

   For detailed HELM deployment instructions, see [Helm Deployment Guide](deploy-helm.md).

### Port-forwarding Zipkin and Grafana dashboards

For Helm deployments, to port-forward services to your local computer, use the following instructions:

- [Zipkin UI](https://zipkin.io/) – Run the following code to port-forward the Zipkin service to your local computer. Then access the Zipkin UI at `http://localhost:9411`.

  ```sh
  kubectl port-forward -n rag service/rag-zipkin 9411:9411 --address 0.0.0.0
  ```

- [Grafana UI](https://grafana.com/) – Run the following code to port-forward the Grafana service to your local computer. Then access the Grafana UI at `http://localhost:3001` and use the default credentials (`admin`/`admin`).

  ```sh
  kubectl port-forward -n rag service/rag-grafana 3001:80 --address 0.0.0.0
  ```

For detailed information on tracing, refer to [Viewing Traces in Zipkin](#view-traces-in-zipkin) and [Viewing Metrics in Grafana Dashboard](#view-metrics-in-grafana).



## Related Topics

- [Query-to-Answer Pipeline](query-to-answer-pipeline.md) – What happens from query to answer and how to study time per stage.
- [NVIDIA RAG Blueprint Documentation](readme.md)
- [Best Practices for Common Settings](accuracy_perf.md).
- [RAG Pipeline Debugging Guide](debugging.md)
- [Troubleshoot](troubleshooting.md)
- [Notebooks](notebooks.md)
