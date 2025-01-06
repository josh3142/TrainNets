data=enb
end="1500"
n_layer=2
n_hidden=128

batch_size=1000
lr="4e-3"
warmup="0.01"
decay="0.5"

# testing
ckpt_v=0
ckpt_name="epoch1499-step1500.ckpt"

seed=6


python train.py -m \
    seed=$seed \
    data=$data epoch.end=$end \
    optim.bs=$batch_size optim.lr=$lr\
    optim.scheduler.decay=$decay optim.scheduler.warmup=$warmup \
    model.param.n_layer=$n_layer model.param.n_hidden=$n_hidden 

# testing
# ckpt_v="1,2,3,4"

python save_renamed_state_dict.py -m \
    seed=$seed \
    data=$data epoch.end=$end \
    optim.bs=$batch_size optim.lr=$lr\
    model.param.n_layer=$n_layer model.param.n_hidden=$n_hidden \
    ckpt.version=$ckpt_v

python test.py -m \
    seed=$seed \
    data=$data epoch.end=$end \
    optim.bs=$batch_size optim.lr=$lr\
    model.param.n_layer=$n_layer model.param.n_hidden=$n_hidden \
    ckpt.version=$ckpt_v ckpt.name=$ckpt_name

python plot_correlation.py -m \
    seed=$seed \
    data=$data epoch.end=$end \
    optim.bs=$batch_size optim.lr=$lr

# swa
# enable=True
# start=1
# n_save=20
# swa_lr="1e0,1e-1,1e-2,1e-3,1e-4"

# python train_continue.py -m \
#     data=$data epoch.end=$end \
#     optim.bs=$batch_size optim.lr=$lr\
#     model.param.n_layer=$n_layer model.param.n_hidden=$n_hidden \
#     swa.enable=$enable swa.epoch.start=$start swa.n_save=$n_save swa.lr=$swa_lr\
#     ckpt.version=$ckpt_v ckpt.name=$ckpt_name