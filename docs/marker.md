# Marker

Install the Marker parser in the RAGit environment:

```bash
pip install -e '.[marker]'
```

Marker parsing without picture descriptions needs no additional service.

## Picture descriptions

With `describe_pictures=True`, RAGit asks a local vLLM vision model to replace
detected pictures with textual descriptions in Marker's Markdown output. The
default model is the small open-source
`Qwen/Qwen2.5-VL-3B-Instruct` model.

The `marker` extra includes the supported vLLM version. Start the local model
server in a separate terminal before running a Marker configuration with
picture descriptions:

```bash
vllm serve Qwen/Qwen2.5-VL-3B-Instruct
```

The server listens at `http://127.0.0.1:8000/v1` by default. You can select a
different vLLM-compatible vision model or server before running RAGit:

```bash
export RAGIT_MARKER_VLLM_MODEL='organization/model-name'
export RAGIT_MARKER_VLLM_BASE_URL='http://127.0.0.1:8000/v1'
```

Serve the selected model under the same name supplied through
`RAGIT_MARKER_VLLM_MODEL`. These environment variables configure the model
deployment, while Marker's low-level parser options remain internal to RAGit.
