CUDA_ROOT=/usr/local/cuda-9.0
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

echo $CUDA_ROOT
echo $TF_INC
echo $TF_LIB

#$CUDA_ROOT/bin/nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF>=1.2.0
g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I$TF_INC/ -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -I$CUDA_ROOT/include -lcudart -L$CUDA_ROOT/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I$TF_INC/ -I$TF_INC/external/nsync/public -L$TF_LIB -I$CUDA_ROOT/include -lcudart -L$CUDA_ROOT/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0



# TF1.2
#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /usr/local/lib/python3.6/site-packages/tensorflow/include -I /usr/local/cuda-9.0/include -lcudart -L /usr/local/cuda-9.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF1.4
#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -I /usr/local/lib/python2.7/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-8.0/lib64/ -L/usr/local/lib/python2.7/dist-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
