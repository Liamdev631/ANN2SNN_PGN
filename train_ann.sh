# !sh
#Train ANN with QCFS.

gpus=1
bs=256
lr=0.01
lr_min=0.00001
epochs=200
l=16
data='cifar100'
model='vgg16'
wd=0
id=${model}-${data}

python main.py train --gpus=$gpus --bs=$bs --lr=$lr --lr_min=$lr_min --epochs=$epochs --l=$l --model=$model --data=$data --id=$id --wd=$wd