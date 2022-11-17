
# For different augmentation levels

## Train original models
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  main_level.py --model-arch resnet50 --batch-size 64 --aug-level 6 --tag level_6

## Train constraint models with scaling schedulers
### Constant
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  main_constraint_level.py --model-arch resnet50 --batch-size 64 --aug-level 6 --tag constant_level_6

### Linear
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  main_constraint_level.py --cfg configs/scaling_linear.yaml --model-arch resnet50 --batch-size 64 --aug-level 1 --tag linear_dot2_level_1

### Cosine
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  main_constraint_level.py --cfg configs/scaling_cosine.yaml --model-arch resnet50 --batch-size 64 --aug-level 1 --tag cosine_level_1


# BYOL
## Pretrain
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  main_SSL.py \
--batch-size 64 --model-arch resnet50 --aug-level 1 --tag level_1_1x

## Linear Evaluation
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  main_SSL_linear_eval.py \
--batch-size 256 --model-arch resnet50 --aug-level 1 --tag level_1


## Resume from pretrained models
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  main.py \
--batch-size 64 --model-arch resnet50 --aug-level 1 --tag level_1_BYOL_pretrain \
--resume output/SSL/resnet50/level_1/ckpt_resnet50.pth




## For resvision
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  main_constraint_level.py --model-arch resnet18 --batch-size 64 --aug-level 4 --tag test