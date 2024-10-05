# !sh
#Convert the trained ANN to SNN, and test the SNN.

gpus=1
bs=128
l=8
data='cifar10'
model='vgg16'
id='vgg16-cifar10'
tau=4
t=32
device='cuda'
seed=42

python main.py test --gpus=$gpus --bs=$bs --l=$l --model=$model --data=$data --mode=ann  --id=$id --tau=$tau --t=$t --device=$device --seed=$seed
python main.py test --gpus=$gpus --bs=$bs --l=$l --model=$model --data=$data --mode=snn  --id=$id --sn_type=if --tau=$tau --t=$t --device=$device --seed=$seed
python main.py test --gpus=$gpus --bs=$bs --l=$l --model=$model --data=$data --mode=snn  --id=$id --sn_type=gn --tau=$tau --t=$t --device=$device --seed=$seed
python main.py test --gpus=$gpus --bs=$bs --l=$l --model=$model --data=$data --mode=snn  --id=$id --sn_type=pgn --tau=$tau --t=$t --device=$device --seed=$seed
