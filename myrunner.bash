# OPENAI_API_KEY=$OPENROUTER_API_KEY lm_eval --model openrouter-chat \
#     --model_args model=meta-llama/llama-3.3-70b-instruct,num_concurrent=64 \
#     --tasks gsm8k_cot_zeroshot \
#     --apply_chat_template 
OPENAI_API_KEY=$OPENROUTER_API_KEY lm_eval --model openrouter-chat \
    --model_args model=meta-llama/llama-3.3-70b-instruct,num_concurrent=1 \
    --tasks strawberry \
    --apply_chat_template 

