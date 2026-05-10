DATA_PATH="/home/lhg/work/ssd_new/visual_geolocalization/denseUAV/DenseUAV"
for n in {1..10}
do
 python train.py --backbone 'dino' --img_size 392 --add_local_test True --itr $n --batch_size 8 --lr 0.00002 --dataset U1652-D2S --dataset_name DenseUAV --data_folder "$DATA_PATH" --weight_D_fine_S_fine 0 --batch_size_eval 8 --epochs 5 --eval_every_n_epoch 5
 sleep 2
done
python get_vote_results.py --dataset U1652-D2S  --dataset_name DenseUAV