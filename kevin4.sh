MODEL_TYPE="microsoft/deberta-base"
OUTPUT_DIR=kevin4

GRADIENT_ACCU_STEPS=4
TRAIN_BATCH_SIZE=16
LEARNING_RATE=1e-5
WARMUP_STEPS=200
WEIGHT_DECAY=0.001
NUM_TRAIN_EPOCHS=12


CUDA_VISIBLE_DEVICES=0 python3 -m trainers.train \
  --model_name_or_path ${MODEL_TYPE} \
  --do_train \
  --evaluate_during_training \
  --gradient_accumulation_steps ${GRADIENT_ACCU_STEPS} \
  --per_gpu_train_batch_size ${TRAIN_BATCH_SIZE} \
  --weight_decay ${WEIGHT_DECAY} \
  --num_train_epochs ${NUM_TRAIN_EPOCHS} \
  --per_gpu_eval_batch_size 1 \
  --learning_rate ${LEARNING_RATE} \
  --output_dir "${OUTPUT_DIR}" \
  --logging_steps 50 \
  --warmup_steps ${WARMUP_STEPS} \
  --eval_split "dev" \
  --score_average_method "binary" \
