python train.py \
--name nyuv2_VGGdeeplab_depthconv \
--dataset_mode nyuv2 \
--flip --scale --crop --colorjitter \
--depthconv \
--list ./lists/train.lst \
--vallist ./lists/val.lst \
--continue_train
