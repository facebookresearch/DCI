#!/bin/bash

/private/home/jju/miniconda3/envs/serve_llama/bin/python -m fastchat.serve.vllm_worker --model-path meta-llama/Llama-2-70b-chat-hf --num-gpus 8 --no-register --download-dir ~/.cache/huggingface/hub/ > server.out 2>&1 &
SERVER_PID=$!

curl --output /dev/null --silent --head --fail http://localhost:21002
until [ "$?" == "22" ]; do
    echo '.'
    sleep 15
    curl --output /dev/null --silent --head --fail http://localhost:21002
done

# /private/home/jju/miniconda3/envs/long-caps/bin/python /private/home/jju/diffusion_with_feedback/long_captions/prepare/gen_summaries.py > sum_gen.out 2>&1
# /private/home/jju/miniconda3/envs/long-caps/bin/python /private/home/jju/diffusion_with_feedback/long_captions/prepare/gen_negatives.py > neg_gen.out 2>&1
/private/home/jju/miniconda3/envs/long-caps/bin/python /private/home/jju/diffusion_with_feedback/long_captions/prepare/gen_alternate_summaries.py > add_sums.out 2>&1

kill "$SERVER_PID"