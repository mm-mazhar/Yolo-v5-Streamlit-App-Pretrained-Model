from io import StringIO
from pathlib import Path
import streamlit as st
import time
from detect import *
import os
import sys
import argparse
from PIL import Image
import cv2
import time

#st.set_page_config(layout = "wide")
st.set_page_config(page_title = "Yolo V5 Multiple Object Detection on Pretrained Model", page_icon="🤖")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 340px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 340px;
        margin-left: -340px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

#################### Title #####################################################
#st.title('Yolo V5 Multiple Object Detection on Pretrained Model')
#st.subheader('Multiple Object Detection on Pretrained Model')
st.markdown("<h3 style='text-align: center; color: red; font-family: font of choice, fallback font no1, sans-serif;'>Yolo V5</h3>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: black; font-family: font of choice, fallback font no1, sans-serif;'>Multiple Object Detection on Pretrained Model</h2>", unsafe_allow_html=True)
#st.markdown('---') # inserts underline
#st.markdown("<hr/>", unsafe_allow_html=True) # inserts underline
st.markdown('#') # inserts empty space

#--------------------------------------------------------------------------------

DEMO_VIDEO = os.path.join('data', 'videos', 'sampleVideo0.mp4')
DEMO_PIC = os.path.join('data', 'images', 'bus.jpg')

def get_subdirs(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result


def get_detection_folder():
    '''
        Returns the latest folder in a runs\detect
    '''
    return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)

#---------------------------Main Function for Execution--------------------------

def main():

    source = ("Detect From Image", "Detect From Video", "Detect From Live Feed")
    source_index = st.sidebar.selectbox("Select Activity", range(
        len(source)), format_func = lambda x: source[x])
    
    cocoClassesLst = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat", \
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat",\
    "baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog",\
    "pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink",\
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush", "All"]
    
    classes_index = st.sidebar.multiselect("Select Classes", range(
        len(cocoClassesLst)), format_func = lambda x: cocoClassesLst[x])
    
    isAllinList = 80 in classes_index
    if isAllinList == True:
        classes_index = classes_index.clear()
        
    print("Selected Classes: ", classes_index)
    
    #################### Parameters to setup ########################################
    # MAX_BOXES_TO_DRAW = st.sidebar.number_input('Maximum Boxes To Draw', value = 5, min_value = 1, max_value = 5)
    deviceLst = ['cpu', '0', '1', '2', '3']
    DEVICES = st.sidebar.selectbox("Select Devices", deviceLst, index = 0)
    print("Devices: ", DEVICES)
    MIN_SCORE_THRES = st.sidebar.slider('Min Confidence Score Threshold', min_value = 0.0, max_value = 1.0, value = 0.4)
    #################### /Parameters to setup ########################################
    
    weights = os.path.join("weights", "yolov5s.pt")

    if source_index == 0:
        
        uploaded_file = st.sidebar.file_uploader(
            "Upload Image", type = ['png', 'jpeg', 'jpg'])
        
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text = 'Resource Loading...'):
                st.sidebar.text("Uploaded Pic")
                st.sidebar.image(uploaded_file)
                picture = Image.open(uploaded_file)
                picture.save(os.path.join('data', 'images', uploaded_file.name))
                data_source = os.path.join('data', 'images', uploaded_file.name)
        
        elif uploaded_file is None:
            is_valid = True
            st.sidebar.text("DEMO Pic")
            st.sidebar.image(DEMO_PIC)
            data_source = DEMO_PIC
        
        else:
            is_valid = False
    
    elif source_index == 1:
        
        uploaded_file = st.sidebar.file_uploader("Upload Video", type = ['mp4'])
        
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text = 'Resource Loading...'):
                st.sidebar.text("Uploaded Video")
                st.sidebar.video(uploaded_file)
                with open(os.path.join("data", "videos", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                data_source = os.path.join("data", "videos", uploaded_file.name)
        
        elif uploaded_file is None:
            is_valid = True
            st.sidebar.text("DEMO Video")
            st.sidebar.video(DEMO_VIDEO)
            data_source = DEMO_VIDEO
        
        else:
            is_valid = False
    
    else:
        ######### Select and capture Camera #################
        
        selectedCam = st.sidebar.selectbox("Select Camera", ("Use WebCam", "Use Other Camera"), index = 0)
        if selectedCam:
            if selectedCam == "Use Other Camera":
                data_source = int(1)
                is_valid = True
            else:
                data_source = int(0)
                is_valid = True
        else:
            is_valid = False
        
        st.sidebar.markdown("<strong>Press 'q' multiple times on camera window and 'Ctrl + C' on CMD to clear camera window/exit</strong>", unsafe_allow_html=True)
        
    if is_valid:
        print('valid')
        if st.button('Detect'):
            if classes_index:
                with st.spinner(text = 'Inferencing, Please Wait.....'):
                    run(weights = weights, 
                        source = data_source,  
                        #source = 0,  #for webcam
                        conf_thres = MIN_SCORE_THRES,
                        #max_det = MAX_BOXES_TO_DRAW,
                        device = DEVICES,
                        save_txt = True,
                        save_conf = True,
                        classes = classes_index,
                        nosave = False, 
                        )
                        
            else:
                with st.spinner(text = 'Inferencing, Please Wait.....'):
                    run(weights = weights, 
                        source = data_source,  
                        #source = 0,  #for webcam
                        conf_thres = MIN_SCORE_THRES,
                        #max_det = MAX_BOXES_TO_DRAW,
                        device = DEVICES,
                        save_txt = True,
                        save_conf = True,
                    nosave = False, 
                    )

            if source_index == 0:
                with st.spinner(text = 'Preparing Images'):
                    for img in os.listdir(get_detection_folder()):
                        if img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith(".png"):
                            pathImg = os.path.join(get_detection_folder(), img)
                            st.image(pathImg)
                    
                    st.markdown("### Output")
                    st.write("Path of Saved Images: ", pathImg)    
                    st.write("Path of TXT File: ", os.path.join(get_detection_folder(), 'labels'))  
                    st.balloons()
                    
            elif source_index == 1:
                with st.spinner(text = 'Preparing Video'):
                    for vid in os.listdir(get_detection_folder()):
                        if vid.endswith(".mp4"):
                            #st.video(os.path.join(get_detection_folder(), vid))
                            #video_file = open(os.path.join(get_detection_folder(), vid), 'rb')
                            #video_bytes = video_file.read()
                            #st.video(video_bytes)
                            video_file = os.path.join(get_detection_folder(), vid)
                            
                stframe = st.empty()
                cap = cv2.VideoCapture(video_file)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                print("Width: ", width, "\n")
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print("Height: ", height, "\n")

                while cap.isOpened():
                    ret, img = cap.read()
                    if ret:
                        stframe.image(cv2.resize(img, (width, height)), channels = 'BGR', use_column_width = True)
                    else:
                        break
                
                cap.release()
                st.markdown("### Output")
                st.write("Path of Saved Video: ", video_file)    
                st.write("Path of TXT File: ", os.path.join(get_detection_folder(), 'labels'))    
                st.balloons()
            
            else:
                with st.spinner(text = 'Preparing Video'):
                    for vid in os.listdir(get_detection_folder()):
                        if vid.endswith(".mp4"):
                            liveFeedvideoFile = os.path.join(get_detection_folder(), vid)
                    
                    st.markdown("### Output")
                    st.write("Path of Live Feed Saved Video: ", liveFeedvideoFile)    
                    st.write("Path of TXT File: ", os.path.join(get_detection_folder(), 'labels'))    
                    st.balloons()
                


# --------------------MAIN FUNCTION CODE------------------------                                                                    
if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
# ------------------------------------------------------------------


