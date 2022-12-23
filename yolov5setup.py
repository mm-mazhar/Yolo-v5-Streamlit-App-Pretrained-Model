import subprocess
import os, shutil

files = ['app.py', 'sampleVideo0.mp4']

# cmd = "pip install -r requirements.txt"
# subprocess.run(cmd, shell = True)
        
print("#################### Downloading Yolo V5 Model Repo ")
cmd = "git clone https://github.com/ultralytics/yolov5"
#subprocess.run(cmd, shell = True)
#cmd = "git reset --hard 68211f72c99915a15855f7b99bf5d93f5631330f"
subprocess.run(cmd, shell = True)

#os.replace(os.path.join('.', 'app.py'), os.path.join('.', 'yolov5'))
shutil.move(os.path.join('.', files[0]), os.path.join('.', 'yolov5'))
print("app.py is copied")

isExist = os.path.exists(os.path.join('.', 'yolov5', 'data', 'videos'))
if not isExist:
   os.makedirs(os.path.join('.', 'yolov5', 'data', 'videos'))
   print("The New Directory is Created!")
   shutil.move(os.path.join('.', files[1]), os.path.join('.', 'yolov5', 'data', 'videos'))
   print("Sample Video file is copied")

os.chdir(os.path.join('.', 'yolov5'))

print("#################### Installing Dependencies ")
cmd = "pip install -r requirements.txt"
subprocess.run(cmd, shell = True)
cmd = "pip install streamlit"
subprocess.run(cmd, shell = True)

import torch

print("#################### Dependencies Check")
#from IPython.display import Image, clear_output  # to display images
#from utils.google_utils import gdrive_download  # to download models/datasets

# clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
