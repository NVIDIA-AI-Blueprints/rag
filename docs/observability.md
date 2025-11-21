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

```{image} assets/zipkin_ui.png
:width: 750px
```

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

### Enable OpenTelemetry Collector, Zipkin and Prometheus stack

1. Modify `values.yaml`:

   Update the `values.yaml` file to enable the OpenTelemetry Collector and Zipkin:

   ```yaml
   env:
   # ... existing code ...
   APP_TRACING_ENABLED: "True"

   # ... existing code ...
   serviceMonitor:
   enabled: true
   opentelemetry-collector:
   enabled: true
   # ... existing code ...

   zipkin:
   enabled: true
   kube-prometheus-stack:
   enabled: true
   ```

### Deploy the Changes

Redeploy the Helm chart to apply these changes:

```sh
helm uninstall rag -n rag
helm install rag -n rag https://helm.ngc.nvidia.com/0648981100760671/charts/nvidia-blueprint-rag-v2.4.0-dev.tgz \
--username '$oauthtoken' \
--password "${NGC_API_KEY}" \
--set imagePullSecret.password=$NGC_API_KEY \
--set ngcApiSecret.password=$NGC_API_KEY \
-f deploy/helm/nvidia-blueprint-rag/values.yaml
```

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

- [NVIDIA RAG Blueprint Documentation](readme.md)
- [Best Practices for Common Settings](accuracy_perf.md).
- [RAG Pipeline Debugging Guide](debugging.md)
- [Troubleshoot](troubleshooting.md)
- [Notebooks](notebooks.md)
