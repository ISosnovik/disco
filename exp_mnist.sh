# !/bin/bash
DATASET_SRC="${DATASET_SRC:-./datasets}"

function train_scale_mnist() {
    # 1 model_name 
    # 2 extra_scaling
    # 3 basis_save_dir 
    # 4 basis_min_scale
    # 5 basis_mult
    # 6 basis_max_order
    # 7 basis_num_rotations
    # 8 basis_sigma

    for seed in {0..5}
    do
        data_dir="$DATASET_SRC/MNIST_scale/seed_$seed/scale_0.3_1.0"
        python train_scale_mnist.py \
            --model $1 \
            --save_model_path "./saved_models/mnist/$1_$2.pt" \
            --cuda \
            --extra_scaling $2 \
            --data_dir="$data_dir" \
            --basis_save_dir $3 \
            --basis_min_scale $4 \
            --basis_mult $5 \
            --basis_max_order $6 \
            --basis_num_rotations $7 \
            --basis_sigma $8 \

    done             
}

# Scale-MNIST + 
train_scale_mnist "mnist_hermite" 0.5 0 1.7 1.5 0 0 0
train_scale_mnist "mnist_lr_harmonics" 0.5 0 0.5 1.5 3 10 0.3
train_scale_mnist "mnist_bspline" 0.5 0 0.6 3 1 0 0
train_scale_mnist "mnist_fb" 0.5 0 0.6 0 0 0 0
train_scale_mnist "mnist_disco" 0.5 "./precalculated_basis" 1.9 1.9 0 0 0



# Scale-MNIST
train_scale_mnist "mnist_hermite" 1.0 0 1.5 1.4 0 0 0
train_scale_mnist "mnist_lr_harmonics" 1.0 0 0.1 1.5 3 11 0.2
train_scale_mnist "mnist_bspline" 1.0 0 0.6 2 2 0 0
train_scale_mnist "mnist_fb" 1.0 0 0.88 0 0 0 0
train_scale_mnist "mnist_disco" 1.0 "./precalculated_basis" 1.9 1.5 0 0 0



