# !/bin/bash
DATASET_SRC="${DATASET_SRC:-./datasets}"

python train_stl10.py \
    --nesterov --cuda \
    --model "wrn_disco" \
    --save_model_path "./saved_models/stl10/wrn_disco.pt" \
    --data_dir="$DATASET_SRC/stl10" \
    --basis_save_dir "./precalculated_basis" \

