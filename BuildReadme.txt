To setup the build directory from the root of the project:

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

To build a new gzl4.cpp file into a new executable:

cd build 
cmake --build . -j$(nproc)
