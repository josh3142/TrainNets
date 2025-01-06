data=mnist
end=2
model=cnn_small
lr="4e-3"
warmup="0.1"
decay="0.3"

# testing
ckpt_v=0
ckpt_name="epoch19-step4700.ckpt"

seed=6 #"1,2,3,4,5"


CUDA_VISIBLE_DEVICES=0 python train.py -m \
    seed=$seed \
    data=$data epoch.end=$end \
    model=$model optim.lr=$lr \
    optim.scheduler.decay=$decay optim.scheduler.warmup=$warmup \

# testing
# ckpt_v="1,2,3,4"

python save_renamed_state_dict.py -m \
    seed=$seed \
    data=$data epoch.end=$end \
    model=$model optim.lr=$lr ckpt.version=$ckpt_v

python test.py -m \
    data=$data epoch.end=$end \
    seed=$seed \
    model=$model optim.lr=$lr\
    ckpt.version=$ckpt_v ckpt.name=$ckpt_name

# swa
# enable=True
# start=1
# n_save=20
# swa_lr="1e0,1e-1,1e-2,1e-3,1e-4"

# CUDA_VISIBLE_DEVICES=0 python train_continue.py -m \
#     seed=$seed \
#     data=$data epoch.end=$end \
#     model=$model optim.lr=$lr \
#     swa.enable=$enable swa.epoch.start=$start swa.n_save=$n_save swa.lr=$swa_lr\
#     ckpt.version=$ckpt_v ckpt.name=$ckpt_name