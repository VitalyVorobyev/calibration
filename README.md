# Calibration library

## Ubuntu setup

```bash
sudo apt install libeigen3-dev libceres-dev nlohmann-json3-dev
```

## Build

```bash
mkdir build; cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="../install" ..
cmake --build . -j4
```