#!/bin/bash -e

PROJECT_FOLDER=$PWD
#echo $PROJECT_FOLDER
VIDEO_FOLDER=$PWD/yolov5/data/videos
#echo $VIDEO_FOLDER

echo "#################### Downloading Yolo V5 Repo "
git clone https://github.com/ultralytics/yolov5

mv './app.py' './yolov5'

if [[ ! -d "$VIDEO_FOLDER" ]]
then
    mkdir -p "$VIDEO_FOLDER"
    echo "Video Folder Created"
    mv './sampleVideo0.mp4' $VIDEO_FOLDER
fi

cd './yolov5'

echo "##### Installing Requirements.txt #####"
pip3 install -r requirements.txt
pip3 install streamlit

python3 - <<END
import torch
print("#################### Dependencies Check")
#from IPython.display import Image, clear_output  # to display images
#from utils.google_utils import gdrive_download  # to download models/datasets
#clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
END

echo "__________________ Script END __________________"
