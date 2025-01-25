#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export HF_HUB_REPO_ID="TheFinAI/lm-eval-results"
echo $CUDA_VISIBLE_DEVICES
#export WRITE_OUT_PATH="results/{model_name}/{tasks_name}/"
export OPENAI_API_SECRET_KEY="xxxx"
export HF_MODELS_CACHE='YOUR_LOCAL_PATH_TO_SAVE_MODELS'
export HF_DATASETS_CACHE='YOUR_LOCAL_PATH_TO_SAVE_DATASETS'
export HF_HOME='YOUR_LOCAL_PATH_TO_SAVE_MODELS'
export HF_TOKEN='YOUT_HF_TOKEN'
export CUDA_LAUNCH_BLOCKING=1


MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
#"meta-llama/Llama-3.1-8B-Instruct"
#"THUDM/glm-4-9b-chat-1m"
#LargeWorldModel/LWM-Text-Chat-1M
#aws-prototyping/MegaBeam-Mistral-7B-512k
#01-ai/Yi-6B-200K
#mistralai/Ministral-8B-Instruct-2410
#CohereForAI/c4ai-command-r-v01
#THUDM/glm-4-9b-chat
#THUDM/chatglm3-6b-128k
#meta-llama/Llama-3.1-8B-Instruct
#Qwen/Qwen2.5-7B-Instruct
#NeoZ123/LongReward-llama3.1-8b-SFT
#Qwen/Qwen2-7B-Instruct


MODEL="vllm" #CHANGE TO vllm/hf
MODEL_ARGS="pretrained=$MODEL_NAME,,tensor_parallel_size=1,gpu_memory_utilization=0.95,trust_remote_code=True,max_model_len=40000"  # CHANGE THE ARGS BASED ON VLLM, YOU NEED TO VERIFY IT 
#,tensor_parallel_size=1,gpu_memory_utilization=0.95,trust_remote_code=True,max_model_len=40000" # ARGS FOR VLLM
#,trust_remote_code=True,parallelize=True,max_length=40000 # ARGS FOR HF


TASKS="ACL_Longcontext_8k"
NUM_FEWSHOT=0
BATCH_SIZE=1
OUTPUT_PATH="results"
CACHE="./cache"
HF_HUB_LOG_ARGS="hub_results_org=TheFinAI,details_repo_name=lm-eval-results-longcontext,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False"
INCLUDE_PATH="./tasks"

lm_eval \
  --model "$MODEL" \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASKS" \
  --device cuda \
  --num_fewshot "$NUM_FEWSHOT" \
  --batch_size "$BATCH_SIZE" \
  --output_path "$OUTPUT_PATH" \
  --hf_hub_log_args "$HF_HUB_LOG_ARGS" \
  --log_samples \
  --include_path "$INCLUDE_PATH" \
  --apply_chat_template \ 
