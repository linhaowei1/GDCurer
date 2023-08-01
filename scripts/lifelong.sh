seed=(2023 4023 2026 3032 2032 2041 3017 2017 2025 2035)

for i in $(seq 0 9);
do
    python code/train.py \
        --wandb \
        --training \
        --seed ${seed[$i]} \
        --extra_info FD+ER \
        --distill \
        --buffer_size 50 \
        --lambda1 0.1 \
        --lambda2 0.15 \
        --lambda3 0.6 \
        --lambda4 0.05 \
        --lambda5 0.1
done
