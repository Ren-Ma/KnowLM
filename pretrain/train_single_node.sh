ROOT="/data/renma/unigpt/"
MODEL_PATH=/data/caihua/huggingfaceModels/llama/llama-13B
OUTPUT_PATH=$ROOT/KnowLM/pretrain/output
dataPath=$ROOT/law_data/wenshu5w/民事案件data
deepspeed_PATH=$ROOT/KnowLM/pretrain/configs/config.json

nohup deepspeed train.py \
    --model_name_or_path $MODEL_PATH \
    --model_max_length 1024 \
    --data_path $dataPath \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 1.5e-5 \
    --warmup_steps 300 \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed $deepspeed_PATH \
    --fp16 True \
    --log_on_each_node False \
    --lr_scheduler_type "cosine" \
    --adam_beta1 0.9 --adam_beta2 0.95 --weight_decay 0.1 \
    > $OUTPUT_PATH/training0721.log &