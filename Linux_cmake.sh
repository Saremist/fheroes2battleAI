cd your_project
mkdir -p build && cd build

cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DENABLE_TOOLS=ON \
         -DENABLE_IMAGE=OFF \
         -DCMAKE_PREFIX_PATH="$(pwd)/../thirdparty/libtorch"

cmake --build . --parallel
