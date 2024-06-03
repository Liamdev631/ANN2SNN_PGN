# !sh
#Convert the trained ANN to SNN, and test the SNN.

gpus=1
bs=128
l=16
data='cifar100'
model='resnet20'
id='resnet20-cifar100'
mode='ann'
sn_type='gn'  #'gn' means group neuron; 'if' means IF neuron
tau=6
t=32
device='cuda'
seed=42

python main.py test --gpus=$gpus --bs=$bs --l=$l --model=$model --data=$data --mode=$mode --id=$id --sn_type=$sn_type --tau=$tau --t=$t --device=$device --seed=$seed