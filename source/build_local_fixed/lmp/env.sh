DEEPMD_ROOT=/usr/local
TENSORFLOW_INCLUDE_DIRS="/root/code/dp_dev/lib/python3.14/site-packages/torch/include;/root/code/dp_dev/lib/python3.14/site-packages/torch/include/torch/csrc/api/include"
TENSORFLOW_LIBRARY_PATH="/root/code/dp_dev/lib/python3.14/site-packages/torch/lib"

TF_INCLUDE_DIRS=`echo $TENSORFLOW_INCLUDE_DIRS | sed "s/;/ -I/g"`
TF_LIBRARY_PATH=`echo $TENSORFLOW_LIBRARY_PATH | sed "s/;/ -L/g"`
TF_RPATH=`echo $TENSORFLOW_LIBRARY_PATH | sed "s/;/ -Wl,-rpath=/g"`

NNP_INC=" -D_GLIBCXX_USE_CXX11_ABI=1 -std=c++17 -DLAMMPS_VERSION_NUMBER=$(./lmp_version.sh) -I$DEEPMD_ROOT/include/ "
NNP_PATH=" -L$TF_LIBRARY_PATH -L$DEEPMD_ROOT/lib"
NNP_LIB=" -Wl,--no-as-needed -ldeepmd_cc -Wl,-rpath=$TF_RPATH -Wl,-rpath=$DEEPMD_ROOT/lib"
