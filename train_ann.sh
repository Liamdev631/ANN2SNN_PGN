# !sh
#Train ANN with QCFS.

gpus=1
bs=128
lr=0.001
lr_min=0.0001
epochs=200
l=16
data='cifar100'
model='vgg16'
wd=0
id=${model}-${data}

python main.py train --gpus=$gpus --bs=$bs --lr=$lr --lr_min=$lr_min --epochs=$epochs --l=$l --model=$model --data=$data --id=$id --wd=$wd