import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
from PIL import Image
import numpy as np
from collections import Counter
import time

#  PAGE CONFIG 
st.set_page_config(page_title="Helmet Detection App", layout="wide", initial_sidebar_state="expanded")

# CUSTOM CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 50%, #45B7D1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .model-card h3 {
        margin: 0;
        color: white;
    }
    .class-badge {
        display: inline-block;
        padding: 0.2rem 0.4rem;
        margin: 0.15rem;
        border-radius: 12px;
        font-weight: normal;
        font-size: 0.75rem;
        background: #f0f0f0;
        color: #333;
        border: 1px solid #ddd;
    }
    section[data-testid="stFileUploadDropzone"] {
        padding: 3rem 2rem !important;
        min-height: 200px !important;
    }
    section[data-testid="stFileUploadDropzone"] > div {
        padding: 2rem !important;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 50%, #45B7D1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 0.5rem 0;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .safety-alert {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

#HEADER 
st.markdown('<div class="main-header">Smart Helmet & Vehicle Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Safety Compliance Monitoring with YOLOv11</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#4ECDC4; font-size:0.9rem; margin:0;'>Developed by <b>Ahmed Pasha</b></p>", unsafe_allow_html=True)

# LOAD MODELS
@st.cache_resource
def load_model(model_type):
    if model_type == "YOLOv11n":
        path = "models/best.pt"
    else:  # YOLOv11m
        path = "models/best (1).pt"

    model = YOLO(path)
    return model

# Load both models
model_n = load_model("YOLOv11n")
model_m = load_model("YOLOv11m")

#SIDEBAR 
st.sidebar.markdown("## ⚙️ Control Panel")

# Model details in sidebar
st.sidebar.markdown("### 🤖 Available Models")

with st.sidebar.expander("🚀 YOLOv11n (Nano)", expanded=True):
    st.markdown("""
    **Optimized for Speed**
    - ⚡ Ultra-fast inference
    - 📱 Lightweight (2.6M params)
    - 🎯 ~85% accuracy
    - ✅ Best for: Real-time applications
    - 🔋 Low resource usage
    """)

with st.sidebar.expander("💪 YOLOv11m (Medium)", expanded=True):
    st.markdown("""
    **Optimized for Accuracy**
    - 🎯 High precision detection
    - 📊 Robust (20M params)
    - 🏆 ~92% accuracy
    - ✅ Best for: Critical safety checks
    - 🔍 Better small object detection
    """)

st.sidebar.markdown("---")

# Advanced settings
st.sidebar.markdown("### 🎛️ Advanced Settings")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.3,
    step=0.05,
    help="Minimum confidence for detections"
)

show_both_models = st.sidebar.checkbox(
    "Show Both Model Results",
    value=True,
    help="Display detections from both YOLOv11n and YOLOv11m"
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Use YOLOv11n for faster processing and YOLOv11m for better accuracy in critical scenarios.")

# MAIN LAYOUT - Split into two columns (swapped positions)
left_col, right_col = st.columns([1, 2])

with left_col:
    # Source selection - moved to left side
    st.markdown("### 📥 Input Source")
    source = st.radio(
        "Select input type:",
        ("📸 Image", "🎬 Video", "🎥 Webcam"),
        label_visibility="collapsed"
    )
    source = source.split()[1]  # Extract the actual source name
    
    # Add helpful text based on selection
    if source == "Image":
        st.caption("📸 Select an image file to detect helmets and vehicles")
    elif source == "Video":
        st.caption("🎬 Upload a video for frame-by-frame analysis")
    else:
        st.caption("🎥 Use your webcam for real-time detection")

with right_col:
    #  CLASS INFORMATION 
    st.markdown("### 🎯 Detection Classes")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="class-badge">✅ With Helmet</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="class-badge">❌ Without Helmet</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="class-badge">🔢 Number Plate</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="class-badge">🏍️ 2 Wheeler</div>', unsafe_allow_html=True)

st.markdown("")  # Minimal spacing

#  PREDICTION FUNCTION 
def predict_and_display(model, model_name, img, conf_threshold):
    start_time = time.time()
    results = model(img, conf=conf_threshold)
    inference_time = time.time() - start_time
    
    annotated = results[0].plot()
    boxes = results[0].boxes
    names = model.names

    # Count detections per class
    detected_classes = [names[int(cls)] for cls in boxes.cls]
    class_counts = dict(Counter(detected_classes))
    
    # Safety check
    helmet_count = class_counts.get('with helmet', 0) + class_counts.get('helmet', 0)
    no_helmet_count = class_counts.get('without helmet', 0) + class_counts.get('no helmet', 0)
    
    safety_status = "SAFE ✅" if no_helmet_count == 0 and helmet_count > 0 else "VIOLATION ⚠️" if no_helmet_count > 0 else "NO RIDERS"
    
    return annotated, class_counts, inference_time, safety_status

# IMAGE MODE 
if source == "Image":
    st.markdown("### 📸 Image Upload & Detection")
    
    # Move browse file up
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        
        # Display original image
        st.markdown("#### 🖼️ Uploaded Image")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.image(img, width=350)

        st.write("")  # small spacing

        detect_btn = st.button("🔍 Start Detection", type="primary", use_container_width=True)

        if detect_btn:
            st.markdown("---")
            
            if show_both_models:
                st.markdown("### 📊 Detection Results - Both Models")
                
                col1, col2 = st.columns(2)
                
                # YOLOv11n Detection
                with col1:
                    st.markdown('<div class="model-card"><h3>🚀 YOLOv11n Results</h3></div>', unsafe_allow_html=True)
                    with st.spinner("Processing with YOLOv11n..."):
                        annotated_n, counts_n, time_n, safety_n = predict_and_display(
                            model_n, "YOLOv11n", np.array(img), confidence_threshold
                        )
                    
                    st.image(annotated_n, width=350)
                    
                    # Metrics
                    m1, m2 = st.columns(2)
                    m1.markdown(f'<div class="metric-box">⏱️ {time_n:.3f}s</div>', unsafe_allow_html=True)
                    m2.markdown(f'<div class="metric-box">🎯 {sum(counts_n.values())} Objects</div>', unsafe_allow_html=True)
                    
                    # Safety status
                    st.markdown(f'<div class="safety-alert">Safety Status: {safety_n}</div>', unsafe_allow_html=True)
                    
                    # Detailed counts
                    if counts_n:
                        st.markdown("**📋 Detections:**")
                        for cls_name, count in counts_n.items():
                            icon = "✅" if "helmet" in cls_name.lower() and "without" not in cls_name.lower() else "❌" if "without" in cls_name.lower() else "🔢" if "plate" in cls_name.lower() else "🏍️"
                            st.markdown(f"{icon} **{cls_name.title()}**: {count}")
                    else:
                        st.info("No objects detected")
                
                # YOLOv11m Detection
                with col2:
                    st.markdown('<div class="model-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);"><h3>💪 YOLOv11m Results</h3></div>', unsafe_allow_html=True)
                    with st.spinner("Processing with YOLOv11m..."):
                        annotated_m, counts_m, time_m, safety_m = predict_and_display(
                            model_m, "YOLOv11m", np.array(img), confidence_threshold
                        )
                    
                    st.image(annotated_m, width=350)
                    
                    # Metrics
                    m1, m2 = st.columns(2)
                    m1.markdown(f'<div class="metric-box">⏱️ {time_m:.3f}s</div>', unsafe_allow_html=True)
                    m2.markdown(f'<div class="metric-box">🎯 {sum(counts_m.values())} Objects</div>', unsafe_allow_html=True)
                    
                    # Safety status
                    st.markdown(f'<div class="safety-alert">Safety Status: {safety_m}</div>', unsafe_allow_html=True)
                    
                    # Detailed counts
                    if counts_m:
                        st.markdown("**📋 Detections:**")
                        for cls_name, count in counts_m.items():
                            icon = "✅" if "helmet" in cls_name.lower() and "without" not in cls_name.lower() else "❌" if "without" in cls_name.lower() else "🔢" if "plate" in cls_name.lower() else "🏍️"
                            st.markdown(f"{icon} **{cls_name.title()}**: {count}")
                    else:
                        st.info("No objects detected")
                
            else:
                # Single model (use YOLOv11n by default)
                st.markdown("### 📊 Detection Results")
                col1, col2, col3 = st.columns([1.5, 1, 1.5])
                with col2:
                    with st.spinner("Processing..."):
                        annotated, counts, inf_time, safety = predict_and_display(
                            model_n, "YOLOv11n", np.array(img), confidence_threshold
                        )
                    
                    st.image(annotated, width=400)
                
                # Metrics row
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⏱️ Inference Time", f"{inf_time:.3f}s")
                with col2:
                    st.metric("🎯 Total Detections", sum(counts.values()) if counts else 0)
                with col3:
                    st.metric("🛡️ Safety Status", safety)
                
                # Detailed counts
                if counts:
                    st.markdown("#### 📋 Detection Details")
                    for cls_name, count in counts.items():
                        icon = "✅" if "helmet" in cls_name.lower() and "without" not in cls_name.lower() else "❌" if "without" in cls_name.lower() else "🔢" if "plate" in cls_name.lower() else "🏍️"
                        st.markdown(f"{icon} **{cls_name.title()}**: {count}")
                else:
                    st.warning("⚠️ No objects detected. Try lowering the confidence threshold.")

            st.success("✅ Detection Completed Successfully!")

# VIDEO MODE 
elif source == "Video":
    st.markdown("### 🎬 Video Upload & Detection")
    
    # Move browse file up
    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        
        st.success("✅ Video uploaded successfully!")
        
        # Model selection for video
        video_model = st.radio("Select model for video processing:", ("YOLOv11n (Faster)", "YOLOv11m (More Accurate)"), horizontal=True)
        model = model_n if "n" in video_model else model_m

        if st.button("▶️ Start Video Detection", type="primary", use_container_width=True):
            st.markdown("---")
            
            # Create centered container for video
            video_col1, video_col2, video_col3 = st.columns([1.5, 1, 1.5])
            with video_col2:
                stframe = st.empty()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            frame_class_counts = Counter()
            frame_count = 0
            total_violations = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, conf=confidence_threshold)
                annotated = results[0].plot()
                stframe.image(annotated, channels="BGR", width=500)

                boxes = results[0].boxes
                names = model.names
                detected_classes = [names[int(cls)] for cls in boxes.cls]
                frame_class_counts.update(detected_classes)
                
                # Count violations
                if any('without' in cls.lower() or 'no helmet' in cls.lower() for cls in detected_classes):
                    total_violations += 1
                
                frame_count += 1
                progress_bar.progress(frame_count / total_frames)
                status_text.text(f"⏳ Processing: Frame {frame_count}/{total_frames} | FPS: {fps}")

            cap.release()
            progress_bar.empty()
            status_text.empty()

            # Video Summary
            st.markdown("### 📊 Video Analysis Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🎞️ Total Frames", total_frames)
            col2.metric("🎯 Total Detections", sum(frame_class_counts.values()))
            col3.metric("⚠️ Violation Frames", total_violations)
            col4.metric("✅ Safe Frames", total_frames - total_violations)
            
            st.markdown("#### 📋 Detailed Detection Count")
            if frame_class_counts:
                for cls_name, count in frame_class_counts.items():
                    icon = "✅" if "helmet" in cls_name.lower() and "without" not in cls_name.lower() else "❌" if "without" in cls_name.lower() else "🔢" if "plate" in cls_name.lower() else "🏍️"
                    st.markdown(f"{icon} **{cls_name.title()}**: {count}")
            else:
                st.warning("⚠️ No objects detected in the video.")
            
            st.success("✅ Video Processing Completed!")

#  WEBCAM MODE 
elif source == "Webcam":
    st.markdown("### 🎥 Live Webcam Detection")
    st.info("💡 Real-time detection using your webcam. Click Start to begin monitoring.")
    
    # Model selection for webcam
    webcam_model = st.radio("Select model for live detection:", ("YOLOv11n (Faster)", "YOLOv11m (More Accurate)"), horizontal=True)
    model = model_n if "n" in webcam_model else model_m
    
    col1, col2 = st.columns(2)
    start_button = col1.button("🎥 Start Webcam", type="primary", use_container_width=True)
    stop_button = col2.button("🛑 Stop Webcam", type="secondary", use_container_width=True)
    
    if start_button:
        st.markdown("---")
        cap = cv2.VideoCapture(0)
        
        # Create centered container for webcam
        webcam_col1, webcam_col2, webcam_col3 = st.columns([1.5, 1, 1.5])
        with webcam_col2:
            stframe = st.empty()
        
        metrics_placeholder = st.empty()
        frame_class_counts = Counter()
        frame_count = 0
        violation_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Cannot access webcam!")
                break

            results = model(frame, conf=confidence_threshold)
            annotated = results[0].plot()
            stframe.image(annotated, channels="BGR", width=500)

            boxes = results[0].boxes
            names = model.names
            detected_classes = [names[int(cls)] for cls in boxes.cls]
            frame_class_counts.update(detected_classes)
            
            # Count violations
            if any('without' in cls.lower() or 'no helmet' in cls.lower() for cls in detected_classes):
                violation_count += 1
            
            frame_count += 1
            
            # Update metrics every 15 frames
            if frame_count % 15 == 0:
                with metrics_placeholder.container():
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("🎞️ Frames", frame_count)
                    col2.metric("🎯 Detections", sum(frame_class_counts.values()))
                    col3.metric("⚠️ Violations", violation_count)
                    col4.metric("✅ Safe", frame_count - violation_count)

            if stop_button:
                break

        cap.release()

        # Webcam Summary
        st.markdown("### 📊 Session Summary")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Frames Processed", frame_count)
        col2.metric("Total Detections", sum(frame_class_counts.values()))
        col3.metric("Violation Rate", f"{(violation_count/frame_count*100):.1f}%" if frame_count > 0 else "0%")
        
        if frame_class_counts:
            st.markdown("#### 📋 Detection Breakdown")
            for cls_name, count in frame_class_counts.items():
                icon = "✅" if "helmet" in cls_name.lower() and "without" not in cls_name.lower() else "❌" if "without" in cls_name.lower() else "🔢" if "plate" in cls_name.lower() else "🏍️"
                st.markdown(f"{icon} **{cls_name.title()}**: {count}")
        else:
            st.info("No detections during this session.")
        
        st.success("✅ Webcam Session Ended")

#  FOOTER 
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🛡️ <strong>Smart Safety Monitoring System</strong> | Powered by YOLOv11 & Streamlit</p>
        <p style='font-size: 0.9rem;'>Helping ensure road safety through AI-powered detection</p>
    </div>
""", unsafe_allow_html=True)