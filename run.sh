# python -m vllm.entrypoints.openai.api_server \
#     --model Qwen/Qwen2.5-14B-Instruct \
#     --host 127.0.0.1 \
#     --port 8000 \
#     --enable-auto-tool-choice \
#     --tool-call-parser hermes

# python -m vllm.entrypoints.openai.api_server \
#     --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
#     --host 127.0.0.1 \
#     --port 8000 \
#     --enable-auto-tool-choice \
#     --tool-call-parser hermes

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Thinking-2507 \
    --host 127.0.0.1 \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-model-len 131072

# python -m vllm.entrypoints.openai.api_server \
#     --model Qwen/Qwen3-14B \
#     --host 127.0.0.1 \
#     --port 8000 \
#     --enable-auto-tool-choice \
#     --tool-call-parser hermes \