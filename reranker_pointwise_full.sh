#user/bin/env bash
pip install accelerate -U
pip install datasets
torchrun --nproc_per_node=4 --master_port=54122 train_cls_bsln.py \
    --model_name_or_path /mnt/ceph_home/pretrained_model/llama-7b-hf \
    --data_path /mnt/ceph_home/data/chunked_tart_data_scored_high \
    --eval_data_path /mnt/ceph_home/data/chunked_dev_data_scored_high \
    --bf16 True \
    --output_dir /mnt/ceph_home/output/reranker-pointwise-full-berri \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "steps" \
    --eval_steps 5 \
    --eval_accumulation_steps 16 \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True