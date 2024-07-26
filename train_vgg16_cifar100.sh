# !sh
#Train ANN with QCFS.

gpus=8
bs=128
lr=0.1
epochs=200
l=8
data='cifar100'
model='vgg16'
id=${model}-${data}

python main.py train --gpus=$gpus --bs=$bs --lr=$lr --epochs=$epochs --l=$l --model=$model --data=$data --id=$id
