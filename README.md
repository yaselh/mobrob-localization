# mobrob-localization
## Setup
To set up the python environment simply type `pip install -r requirements.txt`

Additionally you need to install scikit-image from wheel file. pip install failed with unicode decode error. Download current version from https://www.lfd.uci.edu/~gohlke/pythonlibs/ then just install with `pip install [name].whl`

Another dependency that cannot be installed via pip ins python-pcl which offers python bindings for Pointcloud Library. To setup python pcl go to https://github.com/strawlab/python-pcl and follow the installation instruction for your OS.
For Windows follow roughly those steps:
1. Download Microsoft Visual C++ Compiler Tools from http://landinghub.visualstudio.com/visual-cpp-build-tools
2. Download PCL 1.8.1 for your system from https://github.com/PointCloudLibrary/pcl/releases/
3. Download GTK+ from http://win32builder.gnome.org/. Paste contents of bin folder in pkg-config
4. Add some things to the PATH variable. On my machine the following worked:
    1. C:\Program Files\PCL 1.8.1\bin
    2. C:\Program Files\OpenNI2\Redist
    3. C:\Program Files (x86)\MSBuild\14.0\bin
    4. C:\Program Files (x86)\Windows Kits\10\bin\10.0.16299.0\x64
5. If not set by installer add system variable PCL_ROOT with value C:\Program Files\PCL 1.8.1
6. Install numpy and cython
7. Execute python setup.py build_ext -i
8. Execute python setup.py install
## MS Coco API
If you want to use the MS Coco API move to `3rdparty/cocoapi/PythonAPI`. Then execute make. In order to install make on windows install MinGW. If compilation still does fail on windows with "vcvarsall.bat not found" install Visual Studio Build Tools. Then try again. Note: setup.py was modified in order to work on windows! Original mscoco API has no windows support.

## References
Parts of the project structure are inspired by [this arcticle](http://drivendata.github.io/cookiecutter-data-science/).
