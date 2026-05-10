# train
DATA_PATH="/home/lhg/work/ssd_new/visual_geolocalization/University-Release"
CKPT_PATH="/home/lhg/work/ssd_new/visual_geolocalization/LKGL/LKGL_11.24/checkpoints/university/convnext_base.fb_in22k_ft_in1k_384/05_02_21_33_48/weights_e3_0.9708.pth"

python train.py --backbone 'dino' --img_size 392 --multi_weather True --add_local_test True --itr 0 --batch_size 8 --lr 0.00001 --dataset U1652-D2S --dataset_name U1652 --data_folder "$DATA_PATH" --weight_D_fine_S_fine 0 --batch_size_eval 8 --epochs 3 --eval_every_n_epoch 3

# test
# D2S
python train.py --ckpt_path "$CKPT_PATH" --backbone 'dino' --only_test True --weather_condition 'Normal' --img_size 392 --multi_weather True  --add_local_test True --itr 0 --batch_size 8 --lr 0.00001 --dataset U1652-D2S --dataset_name U1652 --data_folder "$DATA_PATH" --weight_D_fine_S_fine 0 --batch_size_eval 8 --epochs 3 --eval_every_n_epoch 3

python train.py --ckpt_path "$CKPT_PATH" --backbone 'dino' --only_test True --weather_condition 'Fog' --img_size 392 --multi_weather True  --add_local_test True --itr 0 --batch_size 8 --lr 0.00001 --dataset U1652-D2S --dataset_name U1652 --data_folder "$DATA_PATH" --weight_D_fine_S_fine 0 --batch_size_eval 8 --epochs 3 --eval_every_n_epoch 3

python train.py --ckpt_path "$CKPT_PATH" --backbone 'dino' --only_test True --weather_condition 'Rain' --img_size 392 --multi_weather True  --add_local_test True --itr 0 --batch_size 8 --lr 0.00001 --dataset U1652-D2S --dataset_name U1652 --data_folder "$DATA_PATH" --weight_D_fine_S_fine 0 --batch_size_eval 8 --epochs 3 --eval_every_n_epoch 3

python train.py --ckpt_path "$CKPT_PATH" --backbone 'dino' --only_test True --weather_condition 'Snow' --img_size 392 --multi_weather True  --add_local_test True --itr 0 --batch_size 8 --lr 0.00001 --dataset U1652-D2S --dataset_name U1652 --data_folder "$DATA_PATH" --weight_D_fine_S_fine 0 --batch_size_eval 8 --epochs 3 --eval_every_n_epoch 3

python train.py --ckpt_path "$CKPT_PATH" --backbone 'dino' --only_test True --weather_condition 'Fog+Rain' --img_size 392 --multi_weather True  --add_local_test True --itr 0 --batch_size 8 --lr 0.00001 --dataset U1652-D2S --dataset_name U1652 --data_folder "$DATA_PATH" --weight_D_fine_S_fine 0 --batch_size_eval 8 --epochs 3 --eval_every_n_epoch 3

python train.py --ckpt_path "$CKPT_PATH" --backbone 'dino' --only_test True --weather_condition 'Fog+Snow' --img_size 392 --multi_weather True  --add_local_test True --itr 0 --batch_size 8 --lr 0.00001 --dataset U1652-D2S --dataset_name U1652 --data_folder "$DATA_PATH" --weight_D_fine_S_fine 0 --batch_size_eval 8 --epochs 3 --eval_every_n_epoch 3

python train.py --ckpt_path "$CKPT_PATH" --backbone 'dino' --only_test True --weather_condition 'Rain+Snow' --img_size 392 --multi_weather True  --add_local_test True --itr 0 --batch_size 8 --lr 0.00001 --dataset U1652-D2S --dataset_name U1652 --data_folder "$DATA_PATH" --weight_D_fine_S_fine 0 --batch_size_eval 8 --epochs 3 --eval_every_n_epoch 3

python train.py --ckpt_path "$CKPT_PATH" --backbone 'dino' --only_test True --weather_condition 'Dark' --img_size 392 --multi_weather True  --add_local_test True --itr 0 --batch_size 8 --lr 0.00001 --dataset U1652-D2S --dataset_name U1652 --data_folder "$DATA_PATH" --weight_D_fine_S_fine 0 --batch_size_eval 8 --epochs 3 --eval_every_n_epoch 3

python train.py --ckpt_path "$CKPT_PATH" --backbone 'dino' --only_test True --weather_condition 'Over-exposure' --img_size 392 --multi_weather True  --add_local_test True --itr 0 --batch_size 8 --lr 0.00001 --dataset U1652-D2S --dataset_name U1652 --data_folder "$DATA_PATH" --weight_D_fine_S_fine 0 --batch_size_eval 8 --epochs 3 --eval_every_n_epoch 3

python train.py --ckpt_path "$CKPT_PATH" --backbone 'dino' --only_test True --weather_condition 'Wind' --img_size 392 --multi_weather True  --add_local_test True --itr 0 --batch_size 8 --lr 0.00001 --dataset U1652-D2S --dataset_name U1652 --data_folder "$DATA_PATH" --weight_D_fine_S_fine 0 --batch_size_eval 8 --epochs 3 --eval_every_n_epoch 3

# S2D
python train.py --ckpt_path "$CKPT_PATH" --backbone 'dino' --only_test True --weather_condition 'Normal' --img_size 392 --multi_weather True  --add_local_test True --itr 0 --batch_size 8 --lr 0.00001 --dataset U1652-S2D --dataset_name U1652 --data_folder "$DATA_PATH" --weight_D_fine_S_fine 0 --batch_size_eval 8 --epochs 3 --eval_every_n_epoch 3

python train.py --ckpt_path "$CKPT_PATH" --backbone 'dino' --only_test True --weather_condition 'Fog' --img_size 392 --multi_weather True  --add_local_test True --itr 0 --batch_size 8 --lr 0.00001 --dataset U1652-S2D --dataset_name U1652 --data_folder "$DATA_PATH" --weight_D_fine_S_fine 0 --batch_size_eval 8 --epochs 3 --eval_every_n_epoch 3

python train.py --ckpt_path "$CKPT_PATH" --backbone 'dino' --only_test True --weather_condition 'Rain' --img_size 392 --multi_weather True  --add_local_test True --itr 0 --batch_size 8 --lr 0.00001 --dataset U1652-S2D --dataset_name U1652 --data_folder "$DATA_PATH" --weight_D_fine_S_fine 0 --batch_size_eval 8 --epochs 3 --eval_every_n_epoch 3

python train.py --ckpt_path "$CKPT_PATH" --backbone 'dino' --only_test True --weather_condition 'Snow' --img_size 392 --multi_weather True  --add_local_test True --itr 0 --batch_size 8 --lr 0.00001 --dataset U1652-S2D --dataset_name U1652 --data_folder "$DATA_PATH" --weight_D_fine_S_fine 0 --batch_size_eval 8 --epochs 3 --eval_every_n_epoch 3

python train.py --ckpt_path "$CKPT_PATH" --backbone 'dino' --only_test True --weather_condition 'Fog+Rain' --img_size 392 --multi_weather True  --add_local_test True --itr 0 --batch_size 8 --lr 0.00001 --dataset U1652-S2D --dataset_name U1652 --data_folder "$DATA_PATH" --weight_D_fine_S_fine 0 --batch_size_eval 8 --epochs 3 --eval_every_n_epoch 3

python train.py --ckpt_path "$CKPT_PATH" --backbone 'dino' --only_test True --weather_condition 'Fog+Snow' --img_size 392 --multi_weather True  --add_local_test True --itr 0 --batch_size 8 --lr 0.00001 --dataset U1652-S2D --dataset_name U1652 --data_folder "$DATA_PATH" --weight_D_fine_S_fine 0 --batch_size_eval 8 --epochs 3 --eval_every_n_epoch 3

python train.py --ckpt_path "$CKPT_PATH" --backbone 'dino' --only_test True --weather_condition 'Rain+Snow' --img_size 392 --multi_weather True  --add_local_test True --itr 0 --batch_size 8 --lr 0.00001 --dataset U1652-S2D --dataset_name U1652 --data_folder "$DATA_PATH" --weight_D_fine_S_fine 0 --batch_size_eval 8 --epochs 3 --eval_every_n_epoch 3

python train.py --ckpt_path "$CKPT_PATH" --backbone 'dino' --only_test True --weather_condition 'Dark' --img_size 392 --multi_weather True  --add_local_test True --itr 0 --batch_size 8 --lr 0.00001 --dataset U1652-S2D --dataset_name U1652 --data_folder "$DATA_PATH" --weight_D_fine_S_fine 0 --batch_size_eval 8 --epochs 3 --eval_every_n_epoch 3

python train.py --ckpt_path "$CKPT_PATH" --backbone 'dino' --only_test True --weather_condition 'Over-exposure' --img_size 392 --multi_weather True  --add_local_test True --itr 0 --batch_size 8 --lr 0.00001 --dataset U1652-S2D --dataset_name U1652 --data_folder "$DATA_PATH" --weight_D_fine_S_fine 0 --batch_size_eval 8 --epochs 3 --eval_every_n_epoch 3

python train.py --ckpt_path "$CKPT_PATH" --backbone 'dino' --only_test True --weather_condition 'Wind' --img_size 392 --multi_weather True  --add_local_test True --itr 0 --batch_size 8 --lr 0.00001 --dataset U1652-S2D --dataset_name U1652 --data_folder "$DATA_PATH" --weight_D_fine_S_fine 0 --batch_size_eval 8 --epochs 3 --eval_every_n_epoch 3
