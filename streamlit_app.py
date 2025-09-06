import streamlit as st
import torch
import numpy as np
import os, cv2, time, torch, random, pytorchvideo, warnings, argparse, math
from PIL import Image
import tempfile
from collections import deque
warnings.filterwarnings("ignore", category=UserWarning)

from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,)
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slowfast_r50_detection
from deep_sort.deep_sort import DeepSort

# Import functions from yolo_slowfast.py
from yolo_slowfast import (
    MyVideoCapture, 
    tensor_to_numpy, 
    ava_inference_transform, 
    plot_one_box, 
    deepsort_update, 
    save_yolopreds_tovideo
)

class StreamlitVideoCapture(MyVideoCapture):
    """Extends MyVideoCapture to work with Streamlit"""
    def __init__(self, source):
        super().__init__(source)
        self.frame_buffer = deque(maxlen=25)  # Buffer to store frames for processing
    
    def read_for_display(self):
        """Read a frame for display purposes"""
        ret, img = self.cap.read()
        if ret:
            self.frame_buffer.append(img.copy())
            if len(self.frame_buffer) > 25:
                self.frame_buffer.popleft()
        else:
            self.end = True
        return ret, img
    
    def get_buffer_for_processing(self):
        """Get the current buffer for processing"""
        if len(self.frame_buffer) == 25:
            stack_copy = list(self.frame_buffer)
            self.stack = stack_copy
            return True
        return False

def process_frame(frame, model, video_model, deepsort_tracker, id_to_ava_labels, coco_color_map, imsize, device):
    """Process a single frame with YOLO and SlowFast"""
    # Run YOLO detection
    yolo_preds = model([frame], size=imsize)
    yolo_preds.files = ["img.jpg"]
    
    # Run DeepSORT tracking
    deepsort_outputs = []
    for j in range(len(yolo_preds.pred)):
        temp = deepsort_update(deepsort_tracker, yolo_preds.pred[j].cpu(), yolo_preds.xywh[j][:,0:4].cpu(), yolo_preds.ims[j])
        if len(temp) == 0:
            temp = np.ones((0, 8))
        deepsort_outputs.append(temp.astype(np.float32))
    
    yolo_preds.pred = deepsort_outputs
    
    return yolo_preds

def run_slowfast(cap, yolo_preds, video_model, id_to_ava_labels, ava_labelnames, imsize, device):
    """Run SlowFast model on the video clip"""
    if cap.get_buffer_for_processing() and yolo_preds.pred[0].shape[0]:
        clip = cap.get_video_clip()
        inputs, inp_boxes, _ = ava_inference_transform(clip, yolo_preds.pred[0][:,0:4], crop_size=imsize)
        inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0],1), inp_boxes], dim=1)
        
        if isinstance(inputs, list):
            inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
        else:
            inputs = inputs.unsqueeze(0).to(device)
        
        with torch.no_grad():
            slowfaster_preds = video_model(inputs, inp_boxes.to(device))
            slowfaster_preds = slowfaster_preds.cpu()
        
        for tid, avalabel in zip(yolo_preds.pred[0][:,5].tolist(), np.argmax(slowfaster_preds, axis=1).tolist()):
            id_to_ava_labels[tid] = ava_labelnames[avalabel+1]
    
    return id_to_ava_labels

def visualize_results(frame, yolo_preds, id_to_ava_labels, coco_color_map):
    """Visualize detection and tracking results on frame"""
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    if yolo_preds.pred[0].shape[0]:
        for j, (*box, cls, trackid, vx, vy) in enumerate(yolo_preds.pred[0]):
            if int(cls) != 0:
                ava_label = ''
            elif trackid in id_to_ava_labels.keys():
                ava_label = id_to_ava_labels[trackid].split(' ')[0]
            else:
                ava_label = 'Unknown'
            
            text = '{} {} {}'.format(int(trackid), yolo_preds.names[int(cls)], ava_label)
            color = coco_color_map[int(cls)]
            im = plot_one_box(box, im, color, text)
    
    return im

def main():
    st.set_page_config(page_title="YOLO-SlowFast Abnormal Behavior Detection", layout="wide")
    st.title("YOLO-SlowFast Abnormal Behavior Detection")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Model configuration
    device = st.sidebar.selectbox("Device", ["cuda", "cpu"], index=1 if not torch.cuda.is_available() else 0)
    imsize = st.sidebar.slider("Image Size", 320, 1280, 640, 32)
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)
    iou_threshold = st.sidebar.slider("IOU Threshold", 0.1, 1.0, 0.4, 0.05)
    
    # Camera selection
    camera_id = st.sidebar.number_input("Camera ID", 0, 10, 0, 1)
    
    # Initialize models when the user clicks the button
    start_button = st.sidebar.button("Start Detection")
    stop_button = st.sidebar.button("Stop Detection")
    
    # Create a placeholder for the video feed
    video_placeholder = st.empty()
    
    # Session state to track if the app is running
    if 'running' not in st.session_state:
        st.session_state.running = False
    
    if start_button:
        st.session_state.running = True
    
    if stop_button:
        st.session_state.running = False
    
    if st.session_state.running:
        # Load models
        with st.spinner("Loading models..."):
            # Load YOLO model
            model = torch.hub.load('ultralytics/yolov5', 'yolov5l6').to(device)
            model.conf = conf_threshold
            model.iou = iou_threshold
            model.max_det = 100
            
            # Load SlowFast model
            video_model = slowfast_r50_detection(True).eval().to(device)
            
            # Initialize DeepSORT tracker
            deepsort_tracker = DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
            
            # Load AVA labels
            ava_labelnames, _ = AvaLabeledVideoFramePaths.read_label_map("selfutils/temp.pbtxt")
            
            # Generate random colors for COCO classes
            coco_color_map = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]
        
        # Initialize video capture
        cap = StreamlitVideoCapture(camera_id)
        id_to_ava_labels = {}
        
        # Main detection loop
        st.info("Detection started. Click 'Stop Detection' to stop.")
        
        while st.session_state.running:
            ret, frame = cap.read_for_display()
            
            if not ret:
                st.error("Failed to capture frame from camera")
                st.session_state.running = False
                break
            
            # Process frame with YOLO and DeepSORT
            yolo_preds = process_frame(frame, model, video_model, deepsort_tracker, 
                                      id_to_ava_labels, coco_color_map, imsize, device)
            
            # Run SlowFast on accumulated frames
            id_to_ava_labels = run_slowfast(cap, yolo_preds, video_model, id_to_ava_labels, 
                                           ava_labelnames, imsize, device)
            
            # Visualize results
            result_frame = visualize_results(frame, yolo_preds, id_to_ava_labels, coco_color_map)
            
            # Display the result
            video_placeholder.image(result_frame, channels="RGB", use_column_width=True)
            
            # Add a small delay to reduce CPU usage
            time.sleep(0.01)
        
        # Release resources
        cap.release()
        st.success("Detection stopped")
    else:
        # Display instructions when not running
        st.info("Click 'Start Detection' to begin real-time detection using your camera.")
        st.warning("Note: This application requires access to your camera.")

if __name__ == "__main__":
    main()
