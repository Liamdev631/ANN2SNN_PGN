# !sh
#Train ANN with QCFS.

gpus=8
bs=160
lr=0.1
epochs=120
l=8
data='cifar100'
model='resnet20'
id=${model}-${data}

python main.py train --gpus=$gpus --bs=$bs --lr=$lr --epochs=$epochs --l=$l --model=$model --data=$data --id=$id