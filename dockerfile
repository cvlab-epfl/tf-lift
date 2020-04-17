# docker build . -t tflift

# function dockerpythontflift() { 
# 	xhost +;
# 	docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=1 --rm -it \
# 	--env HOST_PERMS="$(id -u):$(id -g)" \
# 	--user=$(id -u) \
# 	--env="DISPLAY" \
# 	--env QT_X11_NO_MITSHM=1 \
# 	--workdir=/app \
# 	--volume="$PWD":/app \
# 	--volume="/home/tsael/":/home/tsael \
# 	--volume="/etc/group:/etc/group:ro" \
# 	--volume="/etc/passwd:/etc/passwd:ro" \
# 	--volume="/etc/shadow:/etc/shadow:ro" \
# 	--volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
# 	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
# 	--volume="/dev/input:/dev/input" \
# 	tflift $*;
# 	xhost -;}
# alias pylift="dockerpythontflift"

FROM tensorflow/tensorflow:1.4.1-gpu-py3
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES},display

SHELL ["/bin/bash", "-c"]
RUN apt-get update && pip install numpy pillow tqdm imutils &&\
DEBIAN_FRONTEND="noninteractive" apt-get install python3-tk -y && ln -fs /usr/share/zoneinfo/America/Chicago /etc/localtime && dpkg-reconfigure --frontend noninteractive tzdata\
&& apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
        libgtk2.0-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /
ENV OPENCV_VERSION="4.1.0"
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
&& wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip \
&& unzip opencv.zip \
&& unzip opencv_contrib.zip \
&& mv opencv-${OPENCV_VERSION} opencv \
&& mv opencv_contrib-${OPENCV_VERSION} opencv_contrib \
&& mkdir /opencv/build \
&& cd /opencv/build \
# && mkdir /opencv-${OPENCV_VERSION}/cmake_binary \
# && cd /opencv-${OPENCV_VERSION}/cmake_binary \
&& cmake -DBUILD_TIFF=ON \
  -DBUILD_opencv_java=OFF \
  -DWITH_CUDA=OFF \
  -DWITH_OPENGL=ON \
  -DWITH_OPENCL=ON \
  -DWITH_IPP=ON \
  -DWITH_TBB=ON \
  -DWITH_EIGEN=ON \
  -DWITH_V4L=ON \
  -DBUILD_TESTS=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DOPENCV_ENABLE_NONFREE=ON \
  -DOPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DCMAKE_INSTALL_PREFIX=$(python3.5 -c "import sys; print(sys.prefix)") \
  -DPYTHON_EXECUTABLE=$(which python3) \
  -DPYTHON_INCLUDE_DIR=$(python3.5 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
  -DPYTHON_PACKAGES_PATH=$(python3.5 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
  .. \
&& make install -j4 \
&& rm /opencv.zip \
&& rm /opencv_contrib.zip \
&& rm -r /opencv \
&& rm -r /opencv_contrib
RUN ln -s \
  /usr/local/python/cv2/python-3.5/cv2.cpython-35m-x86_64-linux-gnu.so \
  /usr/local/lib/python3.5/dist-packages/cv2.so

ENTRYPOINT ["/bin/bash"]

