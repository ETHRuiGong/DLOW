# DLOW
This code is the implementation of the paper "https://arxiv.org/abs/1812.05418".

The code is originally based on https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix and https://github.com/aalmah/augmented_cyclegan.git.

Update： code open source；

Instruction on how to use the code:

Training:
python train.py --dataroot ./datasets/datasetname --name projectname --model cycle_gan --checkpoints_dir ./checkpoints --gpu_ids '0' --lambda_identity 0 --lambda_GA 0 --lambda_GB 0 --display_id -1 --save_epoch_freq 1

Testing:
python test.py --dataroot ./datasets/datasetname --name projectname --model test --dataset_mode single --phase test --gpu_id 0 --checkpoints_dir ./checkpoints/ --loadSize 256 --which_epoch 70 --how_many 750 --label_intensity_styletransfer 0 1 0 0
