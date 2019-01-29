FROM tensorflow/tensorflow

RUN apt-get -y update

# Dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
# protobuf-compiler \
python-pil \
python-lxml \
python-tk \
wget \
git \
libgstreamer1.0-0 \
gstreamer1.0-plugins-base \
gstreamer1.0-plugins-good \
gstreamer1.0-plugins-bad \
gstreamer1.0-plugins-ugly \
gstreamer1.0-libav \
gstreamer1.0-doc \
gstreamer1.0-tools \
libgstreamer1.0-dev \
libgstreamer-plugins-base1.0-dev \
build-essential \
cmake \
libgtk2.0-dev \
pkg-config \
libavcodec-dev \
libavformat-dev \
libswscale-dev \
kafkacat

# Libraries
RUN pip install --upgrade pip 
RUN pip install --user Cython
RUN pip install --user contextlib2
RUN pip install --user jupyter
RUN pip install --user matplotlib

# Object detection
RUN cd / && \
#git clone --depth 1 https://gitlab.com/esrl/jetson
git clone --depth 1 https://github.com/SDU-Embedded/jetson.git

# Install protoc
RUN cd / && ls
RUN echo BREAK
RUN cd /jetson && \
mkdir protoc/ && \
cd /jetson/protoc/ && \
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.6.0/protoc-3.6.0-linux-x86_64.zip && \
unzip protoc-3.6.0-linux-x86_64.zip && \
chmod +x /jetson/protoc/bin/protoc

# Install opencv
RUN cd / && \
wget https://github.com/opencv/opencv/archive/3.4.0.zip && \
ls && \
unzip 3.4.0.zip && \
cd opencv-3.4.0 && \
mkdir build && \
cd build && \
cmake -D WITH_LIBV4L=ON -D WITH_FFMPEG=ON WITH_GSTREAMER=ON -D ENABLE_NEON=ON WITH_OPENGL=OFF -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local .. && \
make -j2 && \
make install && \
rm -rf /opencv-3.4.0 && \
rm /3.4.0.zip

# Coco
RUN cd / && \
git clone https://github.com/cocodataset/cocoapi.git && \
cd cocoapi/PythonAPI/ && \
make && \
cp -r pycocotools /jetson/models/research/

# Protobuf compilation
# RUN cd /jetson/models/research/ && \
# /jetson/protoc/bin/protoc object_detection/protos/*.proto --python_out=. && \
# export PYTHONPATH=$PYTHONPATH:/models/research:models/research/slim

RUN chmod +x /jetson/start.sh

# Add files
COPY files/ /

CMD ["/scripts/do_run"]
