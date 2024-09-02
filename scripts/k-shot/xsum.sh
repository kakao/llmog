# Example command for running many k-shot evaluation
# Modify this script manually based on the required settings

# Number of output_file, num_k, dataset_name, subtask_name, train_path, valid_path must all be matched exactly!

OUTPUT_DIR="output"
mkdir -p $OUTPUT_DIR

# model args
MODEL_PATH=${1:-"google/t5-v1_1-xxl"}
MODEL_CACHE_DIR=${2:-"../cache"}

# env args
SEED=${3:-"42"} # iterate over number of seeds for all the matched settings
NUM_GPUS=${4:-"1"}
OUTPUT_FILE=${5:-"$OUTPUT_DIR/xsum.json"}

# task args
TYPE=${6:-"k-shot"}
NUM_K=${7:-"1 5"}

DATASET_NAME=${8:-"xsum"}
SUBTASK_NAME=${9:-"None"}

TRAIN_PATH=${10:-"None"}
VALID_PATH=${11:-"None"}


python run_many_eval.py \
    --model_path $MODEL_PATH \
    --model_cache_dir $MODEL_CACHE_DIR \
    --seed $SEED \
    --output_file $OUTPUT_FILE \
    --type $TYPE \
    --num_k $NUM_K \
    --dataset_name $DATASET_NAME \
    --subtask_name $SUBTASK_NAME \
    --train_path $TRAIN_PATH \
    --valid_path $VALID_PATH \
    --use_sentinel \
    --template_names "DOC_write_summary_of_above" \
    --num_valid_samples 1000 \
    --generation_hp_path "./configs/generation_config.yaml"
