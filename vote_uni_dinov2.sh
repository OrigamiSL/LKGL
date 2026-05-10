DATA_PATH="/home/lhg/work/ssd_new/visual_geolocalization/University-Release"
############## University D2S
for n in {1..10}
do
 python train.py --backbone 'dino' --img_size 392 --add_local_test True --itr $n --batch_size 8 --lr 0.00001 --dataset U1652-D2S --dataset_name U1652 --data_folder "$DATA_PATH" --weight_D_fine_S_fine 0 --batch_size_eval 8 --epochs 3 --eval_every_n_epoch 3
 sleep 2
done
python get_vote_results.py --dataset U1652-D2S  --dataset_name U1652

############## University S2D
for n in {1..10}
do
 python train.py --backbone 'dino' --img_size 392 --add_local_test True --itr $n --batch_size 8 --lr 0.00001 --dataset U1652-S2D --dataset_name U1652 --data_folder "$DATA_PATH" --weight_D_fine_S_fine 0 --batch_size_eval 8 --epochs 3 --eval_every_n_epoch 3
 sleep 2
done
python get_vote_results.py --dataset U1652-S2D  --dataset_name U1652