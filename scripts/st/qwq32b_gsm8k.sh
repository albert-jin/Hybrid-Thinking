# # jwq：xxx-sk-76661068bc9c45c7b364e1b1c965a355
# # gj：xxx-sk-cef80e6978ef43248d99177af92fdc97
export PATH="/root/.conda/envs/st/bin:${PATH}"
PART1="sk-76661068"
PART2="bc9c45c7b36"
PART3="4e1b1c965a355"
export OPENAI_API_KEY="sk-f5aff073f1da401c98180a7a9c8a50f9"
python run_sglang_softthinking.py \
    --dataset "gsm8k" \
    --model_name "/root/shared-nvme/gj/Hybrid-Thinking/models/QwQ-32B/Qwen/QwQ-32B" \
    --model_id_scope "qwen/QwQ-32B" \
    --max_topk 10 \
    --max_generated_tokens 32768 \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 30 \
    --min_p 0.001 \
    --after_thinking_temperature 0.6 \
    --after_thinking_top_p 0.95 \
    --after_thinking_top_k 30 \
    --after_thinking_min_p 0.0 \
    --early_stopping_entropy_threshold 0.01 \
    --early_stopping_length_threshold 256 \
    --mem_fraction_static 0.8 \
    --start_idx 0 \
    --end_idx 100000 \
    --num_gpus 4 \
    --num_samples 1 \
    --enable_soft_thinking \
    --use_llm_judge \
    --api_base "https://api.deepseek.com/v1" \
    --judge_model_name "deepseek-chat"