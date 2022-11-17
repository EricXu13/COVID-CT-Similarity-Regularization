# COVID-CT-Similarity-Regularization

## Train SR models with scaling schedulers 
### Constant Scheduler
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  main_constraint_level.py --model-arch densenet121 --batch-size 64 --aug-level 4 --tag constant_l4

### Linear Scheduler
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  main_constraint_level.py --cfg configs/scaling_linear.yaml --model-arch densenet121 --batch-size 64 --aug-level 4 --tag linear_l4

### Cosine Scheduler
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  main_constraint_level.py --cfg configs/scaling_cosine.yaml --model-arch densenet121 --batch-size 64 --aug-level 4 --tag cosine_l4

## Train original models
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  main_level.py --model-arch densenet121 --batch-size 64 --aug-level 4 --tag original_l4

## Conventional BYOL pretrain + finetunning
### Pretrain
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  main_SSL.py \
--batch-size 64 --model-arch densenet121 --aug-level 4 --tag byol_l4

### Linear Evaluation
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  main_SSL_linear_eval.py \
--batch-size 256 --model-arch densenet121 --aug-level 4 --tag linear_BYOL_4


### Resume from pretrained models
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  main.py \
--batch-size 64 --model-arch densenet121 --aug-level 4 --tag l4_BYOL_pretrain \
--resume output/SSL/densenet121/byol_l4/ckpt_densenet121.pth

