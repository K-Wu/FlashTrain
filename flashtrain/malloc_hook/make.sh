g++ -I/usr/local/cuda/include -fPIC -shared -o hook.so hook.cpp -ldl -L/usr/local/cuda/lib64 -lcudart -lcufile