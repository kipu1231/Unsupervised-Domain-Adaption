VALEPOCH=1
LR=0.0004
python train_source.py --data_dir $1 --val_epoch $VALEPOCH --lr $LR
