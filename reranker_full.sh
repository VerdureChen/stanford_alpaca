#user/bin/env bash
pip install accelerate -U
torchrun --nproc_per_node=4 --master_port=54122 train.py \
    --model_name_or_path /mnt/ceph_home/pretrained_model/llama-7b-hf \
    --data_path /mnt/ceph_home/data/reranker_100k_bm25_2_gpt_ot.json \
    --bf16 True \
    --output_dir /mnt/ceph_home/output/reranker-full-alpaca \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True