FROM tensorflow/tensorflow

RUN apt-get -y update

# Dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    protobuf-compiler \
    python-pil \
    python-lxml \
    python-tk \
    git

# Libraries
RUN pip install --upgrade pip 
RUN pip install --user Cython
RUN pip install --user contextlib2
RUN pip install --user jupyter
RUN pip install --user matplotlib

# Object detection
RUN cd / && \
   git clone --depth 1 https://gitlab.com/esrl/jetson 

# Coco
RUN cd / && \
git clone https://github.com/cocodataset/cocoapi.git && \
cd cocoapi/PythonAPI/ && \
make && \
cp -r pycocotools /jetson/models/research/

# Add files
COPY files/mongoose_detect.py /jetson/models/research/object_detection
COPY files/frozen_inference_graph.pb /jetson/models/research/object_detection/Mangoose_Outdoor_inference_graph/


# Protobuf compilation
#RUN cd /models/research/ && \
#protoc object_detection/protos/*.proto --python_out=. && \
#export PYTHONPATH=$PYTHONPATH:/models/research:models/research/slim

CMD ["python /jetson/models/research/object_detection/mongoose_detect.py"]
