# CUDA_VISIBLE_DEVICES=1 python train.py config/train_shakespeare_char.py --opt_name=sgdm --learning_rate=1e-2 --decay_lr=False --hyper_lr=0.0 --wandb_log=True &

# CUDA_VISIBLE_DEVICES=7 python train.py config/train_shakespeare_char.py --opt_name=adam_hd --decay_lr=False --hyper_lr=0.00001 --wandb_log=True &

# CUDA_VISIBLE_DEVICES=6 python train.py config/train_shakespeare_char.py --opt_name=adam_hd --decay_lr=False --hyper_lr=0.0 --wandb_log=True &

# CUDA_VISIBLE_DEVICES=7 python train.py config/train_shakespeare_char.py --opt_name=adam_hd --learning_rate=1e-4 --decay_lr=False --hyper_lr=1e-5 --wandb_log=True &
CUDA_VISIBLE_DEVICES=1 python train.py config/train_shakespeare_char.py --opt_name=adam_hdn --learning_rate=1e-4 --decay_lr=False --hyper_lr=1e-6 --wandb_log=True &
CUDA_VISIBLE_DEVICES=2 python train.py config/train_shakespeare_char.py --opt_name=adam_hdn --learning_rate=1e-4 --decay_lr=False --hyper_lr=1e-4 --wandb_log=True &
CUDA_VISIBLE_DEVICES=3 python train.py config/train_shakespeare_char.py --opt_name=adam_hdn --learning_rate=1e-4 --decay_lr=False --hyper_lr=1e-5 --wandb_log=True &
# CUDA_VISIBLE_DEVICES=5 python train.py config/train_shakespeare_char.py --opt_name=adamw --learning_rate=1e-4 --decay_lr=False --hyper_lr=0.0 --wandb_log=True &

# CUDA_VISIBLE_DEVICES=4 python train.py config/train_shakespeare_char.py --opt_name=adam --hyper_lr=0.0 --wandb_log=True &

# CUDA_VISIBLE_DEVICES=7 python train.py config/train_shakespeare_char.py --opt_name=sgd_hdn --learning_rate=1e-2 --decay_lr=False --hyper_lr=1e-3 --wandb_log=True &

# CUDA_VISIBLE_DEVICES=6 python train.py config/train_shakespeare_char.py --opt_name=sgd_hdn --learning_rate=1e-2 --decay_lr=False --hyper_lr=1e-4 --wandb_log=True &
wait