#!/bin/sh

/jetson/protoc/bin/protoc /jetson/models/research/object_detection/protos/*.proto --python_out=. 
export PYTHONPATH=$PYTHONPATH:/jetson/models/research:/jetson/models/research/slim
python /jetson/models/research/object_detection/mongoose_detect.py $_camera_ip $_iface $_weight_path $_topic 
