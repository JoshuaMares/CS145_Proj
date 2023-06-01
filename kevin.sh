# DATA_DIR="datasets/com2sense"
#MODEL_TYPE="bert-base-cased"
#MODEL_TYPE="roberta-base"
MODEL_TYPE="microsoft/deberta-v3-base"
# TASK_NAME="com2sense"
OUTPUT_DIR=kevin

GRADIENT_ACCU_STEPS=4
TRAIN_BATCH_SIZE=48
LEARNING_RATE=1e-6
MAX_STEPS=2000
MAX_SEQ_LENGTH=128
WARMUP_STEPS=100


CUDA_VISIBLE_DEVICES=0 python3 -m trainers.train \
  --model_name_or_path ${MODEL_TYPE} \
  --do_train \
  --do_eval \
  --eval_all_checkpoints \
  --gradient_accumulation_steps ${GRADIENT_ACCU_STEPS} \
  --per_gpu_train_batch_size ${TRAIN_BATCH_SIZE} \
  --per_gpu_eval_batch_size 1 \
  --learning_rate ${LEARNING_RATE} \
  --max_steps ${MAX_STEPS} \
  --max_seq_length ${MAX_SEQ_LENGTH} \
  --output_dir "${OUTPUT_DIR}" \
  --task_name "${TASK_NAME}" \
  --save_steps 100 \
  --logging_steps 100 \
  --warmup_steps ${WARMUP_STEPS} \
  --eval_split "dev" \
  --score_average_method "binary" \
#   --evaluate_during_training \
  # --do_not_load_optimizer \
  # --overwrite_output_dir \
#   --data_dir "${DATA_DIR}" \