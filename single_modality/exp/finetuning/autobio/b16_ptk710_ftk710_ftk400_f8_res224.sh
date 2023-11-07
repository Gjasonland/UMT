export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

JOB_NAME='b16_ptk710_ftk710_ftautobio_f8_res224'
OUTPUT_DIR="$(dirname $0)/run6_$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
PREFIX='/home/weijie/autobio/kinect/video_all'
DATA_PATH='/home/weijie/autobio/kinect'
MODEL_PATH="/home/weijie/autobio/b16_ptk710_f8_res224.pth"

PARTITION='video'
GPUS=1
GPUS_PER_NODE=1
CPUS_PER_TASK=14

python run_class_finetuning.py \
        --model vit_base_patch16_224 \
        --data_path ${DATA_PATH} \
        --prefix ${PREFIX} \
        --data_set 'Kinetics_sparse' \
        --split ',' \
        --nb_classes 79 \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 4 \
        --num_sample 1 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 10 \
        --num_frames 8 \
        --num_workers 1 \
        --warmup_epochs 5 \
        --tubelet_size 1 \
        --epochs 200 \
        --lr 1e-3 \
        --drop_path 0.1 \
        --opt adamw \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --test_num_segment 4 \
        --test_num_crop 3 \
        --dist_eval \
        --test_best \
        --eval
