
# Set the other parameters needed for training the model according to your specific requirements, such as learning rate, num of gpu, etc.
nohup python -m torch.distributed.launch --nproc_per_node=8 --master_port 10086 main_bert.py > details_discrete_full.log 2>&1 &
