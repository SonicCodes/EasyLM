#! /bin/bash

# This is the example script to pretrain a 3B LLaMA model on a TPU v4-64 pod.
# To use this on TPU pod, you need to run this
# script on every hosts in a TPU pod.



umask 000
LEV_ROOT=$(dirname "$(readlink -f $0)")/..

source ~/venv/bin/activate

PYTHONPATH=${LEV_ROOT}:${LEV_ROOT}/src:${LEV_ROOT}/examples:$PYTHONPATH "$@"

# Put your WANDB API key here to enable logging to wandb.
# export WANDB_API_KEY='<your wandb api key here>'

# or just export it before...

# kinda unclear on mesh_dim might need to be mesh_dim='1,8,2' 

# TPU specific flags to improve training throughput
export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_enable_async_all_gather=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'


python -m EasyLM.models.llama.llama_train \
    --mesh_dim='-1,8,1' \
    --dtype='bfloat16' \
    --total_steps=200000 \
    --log_freq=500 \
    --save_model_freq=5000 \
    --save_milestone_freq=25000 \
    --eval_steps 10\
    --load_llama_config='7b' \
    --update_llama_config='' \
    --load_dataset_state='gs://caltech-bucket/LLM/easyLM/7bSMD1/experiment_output/llama3-log/80607f3a98894eb3b8e0621959ca4255/dataset_100000.pkl'  \
    --load_checkpoint='trainstate::gs://caltech-bucket/LLM/easyLM/7bSMD1/experiment_output/llama3-log/80607f3a98894eb3b8e0621959ca4255/streaming_train_state_100000' \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.1 \
    --optimizer.adamw_optimizer.lr=3e-4 \
    --optimizer.adamw_optimizer.end_lr=3e-12 \
    --optimizer.adamw_optimizer.lr_warmup_steps=0 \
    --optimizer.adamw_optimizer.lr_decay_steps=250000 \
    --train_dataset.type='huggingface' \
    --train_dataset.text_processor.fields='text' \
    --train_dataset.huggingface_dataset.path='HuggingFaceFW/fineweb-edu' \
    --train_dataset.huggingface_dataset.streaming=True \
    --train_dataset.huggingface_dataset.seq_length=2048 \
    --train_dataset.huggingface_dataset.batch_size=128 \
    --train_dataset.huggingface_dataset.split='train' \
    --train_dataset.huggingface_dataset.name='sample-100BT' \
    --eval_dataset.type='huggingface' \
    --eval_dataset.text_processor.fields='text' \
    --eval_dataset.huggingface_dataset.path='HuggingFaceFW/fineweb-edu' \
    --eval_dataset.huggingface_dataset.streaming=True \
    --eval_dataset.huggingface_dataset.seq_length=2048 \
    --eval_dataset.huggingface_dataset.batch_size=128 \
    --eval_dataset.huggingface_dataset.split='train' \
    --eval_dataset.huggingface_dataset.name='sample-10BT' \
    --checkpointer.save_optimizer_state=True \
    --logger.online=True \
    --logger.prefix='EasyLM' \
    --logger.project="open_llama_7b" \
    --logger.output_dir="gs://caltech-bucket/LLM/easyLM/7bBaseRunInitFix/experiment_output/llama3-log" \
    --logger.wandb_dir="/dev/shm/experiment_output/open_llama_7b_l2_base" \
    --q=2 \
    --starting_q=None \
    --q_decay_steps=10000 \
|& tee $HOME/output.txt
