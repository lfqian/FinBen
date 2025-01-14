# github repo clone
git clone git@github.com:xueqingpeng/FinBen.git --recursive #change to your repo

# set environment
cd FinBen/finlm_eval/
conda create -n finben python=3.12
conda activate finben
pip install -e .
pip install -e .[vllm]

# log in to your huggingface
HF_TOKEN="your_hf_token"

# model evaluation
cd FinBen/
#small model
lm_eval --model hf --model_args "pretrained=meta-llama/Llama-3.2-1B-Instruct" --tasks GRQA --device cuda:0 --batch_size 8 --output_path results --hf_hub_log_args "hub_results_org=TheFinAI,results_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" --log_samples --apply_chat_template --include_path ./tasks
#large model
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
lm_eval --model vllm --model_args "pretrained=google/gemma-2-27b-it,tensor_parallel_size=4,gpu_memory_utilization=0.8,max_model_len=1024" --tasks GRMultifin --batch_size auto --output_path results --hf_hub_log_args "hub_results_org=TheFinAI,hub_repo_name=lm-eval-results,push_results_to_hub=True,push_samples_to_hub=True,public_repo=False" --log_samples --apply_chat_template --include_path ./tasks

# new task
cd FinBen/tasks/your_folder