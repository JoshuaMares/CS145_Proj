# DATA_DIR="datasets/com2sense"
#MODEL_TYPE="bert-base-cased"
#MODEL_TYPE="roberta-base"
MODEL_TYPE="microsoft/deberta-base"
# TASK_NAME="com2sense"
OUTPUT_DIR=kevin

GRADIENT_ACCU_STEPS=4
TRAIN_BATCH_SIZE=16
LEARNING_RATE=5e-5
WARMUP_STEPS=500
WEIGHT_DECAY=0.01
NUM_TRAIN_EPOCHS=5


CUDA_VISIBLE_DEVICES=0 python3 -m trainers.train \
  --model_name_or_path ${MODEL_TYPE} \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --eval_all_checkpoints \
  --gradient_accumulation_steps ${GRADIENT_ACCU_STEPS} \
  --per_gpu_train_batch_size ${TRAIN_BATCH_SIZE} \
  --weight_decay ${WEIGHT_DECAY} \
  --num_train_epochs ${NUM_TRAIN_EPOCHS} \
  --per_gpu_eval_batch_size 1 \
  --learning_rate ${LEARNING_RATE} \
  --output_dir "${OUTPUT_DIR}" \
  --task_name "${TASK_NAME}" \
  --save_steps 50 \
  --logging_steps 50 \
  --warmup_steps ${WARMUP_STEPS} \
  --eval_split "dev" \
  --score_average_method "micro" \
  # --max_seq_length ${MAX_SEQ_LENGTH} \
  # --do_not_load_optimizer \
  # --overwrite_output_dir \
#   --data_dir "${DATA_DIR}" \