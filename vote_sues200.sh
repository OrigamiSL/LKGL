DATA_PATH="/home/lhg/work/ssd_new/visual_geolocalization/SUES-200-512x512-V2/SUES-200-512x512"
##############  D2S
for n in {1..10}
do
 python train.py --data_folder "$DATA_PATH" --add_local_test True --itr $n --batch_size 8 --lr 0.0001 --dataset U1652-D2S --altitude 150 --dataset_name SUES-200
 sleep 2
done
python get_vote_results.py --dataset U1652-D2S --altitude 150 --dataset_name SUES-200

for n in {1..10}
do
 python train.py --data_folder "$DATA_PATH" --add_local_test True --itr $n --batch_size 8 --lr 0.0001 --dataset U1652-D2S --altitude 200 --dataset_name SUES-200
 sleep 2
done
python get_vote_results.py --dataset U1652-D2S --altitude 200 --dataset_name SUES-200

for n in {1..10}
do
 python train.py --data_folder "$DATA_PATH" --add_local_test True --itr $n --batch_size 8 --lr 0.0001 --dataset U1652-D2S --altitude 250 --dataset_name SUES-200
 sleep 2
done
python get_vote_results.py --dataset U1652-D2S --altitude 250 --dataset_name SUES-200

for n in {1..10}
do
 python train.py --data_folder "$DATA_PATH" --add_local_test True --itr $n --batch_size 8 --lr 0.0001 --dataset U1652-D2S --altitude 300 --dataset_name SUES-200
 sleep 2
done
python get_vote_results.py --dataset U1652-D2S --altitude 300 --dataset_name SUES-200
##############  S2D
for n in {1..10}
do
 python train.py --data_folder "$DATA_PATH" --add_local_test True --itr $n --batch_size 8 --lr 0.0001 --dataset U1652-S2D --altitude 150 --dataset_name SUES-200
 sleep 2
done
python get_vote_results.py --dataset U1652-S2D --altitude 150 --dataset_name SUES-200

for n in {1..10}
do
 python train.py --data_folder "$DATA_PATH" --add_local_test True --itr $n --batch_size 8 --lr 0.0001 --dataset U1652-S2D --altitude 200 --dataset_name SUES-200
 sleep 2
done
python get_vote_results.py --dataset U1652-S2D --altitude 200 --dataset_name SUES-200

for n in {1..10}
do
 python train.py --data_folder "$DATA_PATH" --add_local_test True --itr $n --batch_size 8 --lr 0.0001 --dataset U1652-S2D --altitude 250 --dataset_name SUES-200
 sleep 2
done
python get_vote_results.py --dataset U1652-S2D --altitude 250 --dataset_name SUES-200

for n in {1..10}
do
 python train.py --data_folder "$DATA_PATH" --add_local_test True --itr $n --batch_size 8 --lr 0.0001 --dataset U1652-S2D --altitude 300 --dataset_name SUES-200
 sleep 2
done
python get_vote_results.py --dataset U1652-S2D --altitude 300 --dataset_name SUES-200