import streamlit as st
import cv2 
from ultralytics import YOLO
import pandas as pd

def click_detect_btn():
        return 0

def video_params(cap):
    # get videoclip's parameters
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    params_dict = {"width": width,
                   "height": height,
                   "fps": fps,
                   "length": length
                   }
    return params_dict

def create_writer(video_file, cap):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_path = video_file.split('.')[0] + '_out.avi'
    print(output_path)
    out_video = cv2.VideoWriter(
        output_path, 
        cv2.VideoWriter.fourcc(*'XVID'), 
        fps, (width, height))
    return out_video


st.set_page_config(layout="wide")
st.header("UAV detect")
with st.sidebar:
    
    with st.form("forma"):
        uploaded_video = st.file_uploader(
            "Choose a video...", type=["mp4"])
        detect_thr = st.slider("Detection threshold",
                   0., 100., 50.)
        st.form_submit_button("Submit")

col1, col2 = st.columns(2)

    

# I convert bytestream to file, openCV likes it
if uploaded_video is not None:
    with col1:
        model = YOLO('temp.pt')
        st.video(uploaded_video)
        vid_name = uploaded_video.name
        with open(vid_name, mode='wb') as f:
            f.write(uploaded_video.read())
        
    
        vid_cap = cv2.VideoCapture(vid_name)
        vid_params = video_params(vid_cap)
        out_video = create_writer(video_file=vid_name, cap=vid_cap)


    with st.form("detect_form"):
        #st.table(vid_params)  
        percent = st.slider("Percent of video to process", 0., 100., 5.)   
        frame_skip = st.slider("Frame skip", 0, 10, 3)  
        show_chkbox = st.checkbox("Show preview", value=True) 
        detect_btn = st.form_submit_button("Detect")  
        frames_to_include = int(vid_params["length"] * percent/100.)
    with col2:
        stframe = st.empty()
        if detect_btn:
            progress_text = "Detection in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)
            for i in range(frames_to_include):
                percent_completed = int(100*i/float(frames_to_include))
                my_bar.progress(percent_completed+1,text=progress_text)
                ret, frame = vid_cap.read()
                if i%(frame_skip+1) == 0:
                    result = model(frame)
                for detection in result[0].boxes.data:
                    x0, y0 = int(detection[0]), int(detection[1])
                    x1, y1 = int(detection[2]), int(detection[3])
                    score = round(float(detection[4]), 2)
                    if (score > detect_thr/100.):
                        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR->RGB, if needed
                if show_chkbox:
                    stframe.image(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out_video.write(frame)
            my_bar.empty()
            output_path = vid_name.split('.')[0] + '_out.avi'
            st_video = open(output_path,'rb')
            video_bytes = st_video.read()
            st.download_button("Download", data=video_bytes, file_name="out.avi", mime="video/avi")


    


    
