export CUDA_HOME='/usr/local/cuda-11.4'
cd detectron2/ && python3 setup.py build
pip3 install cloudpickle
pip3 install omegaconf
cd ../fvcore/ && python3 setup.py install
pip3 install cython
cd ../cocoapi/PythonAPI/ && python3 setup.py install
cd ../../
#mv detectron2/build/lib.linux-x86_64-3.8/detectron2 ./build/...
