Here you find instructions how to install and build the BayesNet library and run Unit tests or the examples.

# Linux
## Get BayesNet
Clone BayesNet and GoogleTest
```
cd <your-install-dir>
git clone https://github.com/fkeidel/BayesNet.git
cd BayesNet
git clone https://github.com/google/googletest.git
```
## Build and run
### Command line
build
```
cd <your-install-dir>
mkdir build
cd build
cmake ..
make
```
run

In the build folder go to subfolder tests or examples and run the executables

### IDE (Clion, VS Code ...)
Open the folder <your-install-dir> to open the cmake project, then build with your IDE
# Windows

