sudo docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$1 -p $2:$2  -it --rm --ulimit memlock=-1 --ulimit stack=67108864  -v $(pwd):/workspace nvcr.io/nvidia/pytorch:20.11-py3  
