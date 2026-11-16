#  --dataset_max_samples DATASET_MAX_SAMPLES
#                        Limits the number of dataset samples. -1 means all data
#  --dataset_path DATASET_PATH #                        dataset basedir
#  --seed SEED           random seed in [0,inf]. -1 means non-deterministic
#  --conv_filter_size_x CONV_FILTER_SIZE_X
#                        convolution filter size in x direction
#  --conv_filter_size_z CONV_FILTER_SIZE_Z
#                        convolution filter size in z direction (depth)
#  --use_transformers USE_TRANSFORMERS
#                        whether using transformers in UNet latent space
#  --max_epochs MAX_EPOCHS
#                        maximum number of training epochs

#MAX_EPOCHS=10
#MAX_SAMPLES=20
MAX_EPOCHS=150
MAX_SAMPLES=-1
SEED=100
MLURI="http://gr14sm02.ep.petrobras.com.br:31947/"
MLNAME="deep_tomography_2.0"
LOGDIR="./logs"


SRCPATH=../
SCRIPT=$SRCPATH/src/tests/ray_test.py

PYTHONPATH=$SRCPATH  python  $SCRIPT 

# UNET 3x3
#CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$SRCPATH  python  $SCRIPT --execution_id="UNET_3X3" --seed=$SEED \
#                                                                       --experiment_tracking_uri=$MLURI --experiment_tracking_name=$MLNAME \
#                                                                       --dataset_max_samples=$MAX_SAMPLES --max_epochs=$MAX_EPOCHS \
#                                                                       --no-use_transformers  \
#                                                                       --conv_filter_size_x=3 --conv_filter_size_z=3 1>"$LOGDIR/UNET_3X3.out" 2>&1 &



