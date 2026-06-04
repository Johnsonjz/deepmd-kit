DEEPMD_ROOT=/data/home/public/jiangzhen/dp/deepmd-kit/build/sog_lmp_stable
TENSORFLOW_INCLUDE_DIRS="/data/home/public/.conda/envs/mlip/lib/python3.10/site-packages/torch/include;/data/home/public/.conda/envs/mlip/lib/python3.10/site-packages/torch/include/torch/csrc/api/include"
TENSORFLOW_LIBRARY_PATH="/data/home/public/.conda/envs/mlip/lib/python3.10/site-packages/torch/lib"

TF_INCLUDE_DIRS=`echo $TENSORFLOW_INCLUDE_DIRS | sed "s/;/ -I/g"`
TF_LIBRARY_PATH=`echo $TENSORFLOW_LIBRARY_PATH | sed "s/;/ -L/g"`
TF_RPATH=`echo $TENSORFLOW_LIBRARY_PATH | sed "s/;/ -Wl,-rpath=/g"`

NNP_INC=" -DLAMMPS_VERSION_NUMBER=$(./lmp_version.sh) -I$DEEPMD_ROOT/include/ "
NNP_PATH=" -L$TF_LIBRARY_PATH -L$DEEPMD_ROOT/lib"
NNP_LIB=" -Wl,--no-as-needed -ldeepmd_c -Wl,-rpath=$TF_RPATH -Wl,-rpath=$DEEPMD_ROOT/lib"
