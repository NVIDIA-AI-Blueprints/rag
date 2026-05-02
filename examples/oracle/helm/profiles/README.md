# Oracle Helm Model Profiles

The Oracle wrapper chart uses the stock NVIDIA RAG chart as a dependency named
`rag`. To change models, use the same stock RAG values but nest them under
`rag:`.

## Stock-to-Wrapper Translation

Stock RAG values:

```yaml
envVars:
  APP_LLM_MODELNAME: "<model-name>"
nimOperator:
  nim-llm:
    image:
      repository: nvcr.io/nim/<image>
      tag: "<tag>"
```

Oracle wrapper values:

```yaml
rag:
  envVars:
    APP_LLM_MODELNAME: "<model-name>"
  nimOperator:
    nim-llm:
      image:
        repository: nvcr.io/nim/<image>
        tag: "<tag>"
```

## Use Nemotron 3 Super 120B

```bash
helm install rag examples/oracle/helm \
  -f examples/oracle/helm/values.create-adb.yaml \
  -f examples/oracle/helm/profiles/nemotron3-super-values.yaml \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY \
  --timeout 90m
```

`nemotron3-super-values.yaml` is generated from the stock NVIDIA RAG
`deploy/helm/nvidia-blueprint-rag/nemotron3-super-values.yaml` file by nesting
the same values under `rag:`. It does not add Oracle-specific model tuning.

Large models require sufficient GPU capacity. If the selected model cannot run
on the cluster hardware, the NIM service will fail profile selection and the
user should choose a smaller model or a larger GPU shape.
