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
configure and build
```
cd <your-install-dir>
mkdir build
cd build
cmake ..
make
```
run

In the build folder go to the subfolder "tests" or "examples" and run the executables

### IDE (Clion, VS Code ...)
Open the folder \<your-install-dir\> to open the cmake project, then build with your IDE.
You might want to add build and run configurations
- Clion: Go to File->Settings->Build,Execution,Deployment->Cmake and create the configuration.
- VS Code: If not installed, install extension "CMake Tools". When opening the project, you will be asked, if you want to configure the project. Click yes. 
To configure the project, you can also select the file CMakeLists.txt, right-click on it and then select "Configure all projects", then again, right-click on it and select "Build all projects". Then go to Run->"Add configuration" and configure your launch configuration.

# Windows
## Prerequisites
- [Git](https://git-scm.com/download/): Download and install for Windows
- [cmake](https://cmake.org/download/): Download and run Windows x64 Installer
- [Visual Studio](https://visualstudio.microsoft.com/de/downloads/): Download and install Visual Studio "Community" edition. During installation select "Desktop development with C++"
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
configure and build
```
cd <your-install-dir>
mkdir build
cd build
cmake ..
cmake --build .
```
run

In the build folder go to the subfolder tests or examples. The executables are in the subfolder for your build configuration, e.g. debug. Then run the executables.

### Visual Studio
You have two possibilities:
1. Configure with cmake, then open Visual Studio solution (.sln)<br>
Configure with the following commands. This will create Visual Studio projects and a solution. You can then open the solution.
```
cd <your-install-dir>
mkdir build
cd build
cmake ..
```

2. Open cmake project directly in Visual Studio<br>
Select "Open local folder" in the starting screen, or when already opened, select "File->Open->Folder...", then select the folder \<your-install-dir\>. 
You might want to add a build configuration (e.g. release) to the project. Go to Project->"CMake Settings for BayesNet" and add a configuratiion.
Then build and run a target (e.g. an example or the unit tests).





