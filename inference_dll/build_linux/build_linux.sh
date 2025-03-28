g++ -o inference ../inference_main.cpp \
    -I/usr/local/cuda-12.6/include \
    -I/usr/local/tensorrt-10.5/include \
    -L/usr/local/cuda-12.6/lib64 \
    -L/usr/local/tensorrt-10.5/lib \
    -lnvinfer \
    -lnvonnxparser \
    -lcudart \
    -lpthread \
    -std=c++11
