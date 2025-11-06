# Helmet & Number Plate Detection using YOLO11

This project focuses on detecting helmets, number plates, and two-wheelers using the YOLO11 object detection model. The system is specifically designed to support traffic safety monitoring and automated surveillance applications.

##  Project Overview

The model is trained to classify **four categories**:

1. **Helmet**
2. **Without Helmet**
3. **Number Plate**
4. **2-Wheeler**

The dataset consists of images from real-world road environments with diverse lighting and camera angles.



##  Models Used

Two YOLO11 model variants were used in training:

| Model | Accuracy Achieved |
|-------|------------------|
| **YOLO11n** (Nano) | **90%** |
| **YOLO11m** (Medium) | **92%** |

 **YOLO11n**: Lightweight and fast, suitable for edge deployment (Raspberry Pi / Jetson Nano).
 **YOLO11m**: More accurate due to larger backbone, suitable for high-performance inference.



## 🛠️ Tech Stack

| Component | Details |
|----------|---------|
| Model | YOLO11 (Ultralytics) |
| Framework | PyTorch |
| Deployment UI | Streamlit |
| Image Pre-processing | OpenCV |
| Annotation Format | YOLO TXT Format |

---

##  Usage

### 1. Clone Repository
```bash
git clone <your-repository-link>
cd helmet-detection
```

### 2. Install Dependencies
```bash
pip install ultralytics streamlit opencv-python pillow numpy
```

### 3. Run the Application
```bash
streamlit run app.py
```


## 📊 Results

- YOLO11n achieved **90% accuracy**
- YOLO11m achieved **92% accuracy**
- The model successfully identifies:
  - Riders wearing helmet
  - Riders without helmet
  - Number plate of vehicle
  - Different 2-wheelers

This helps in **traffic monitoring, rule enforcement, and safety analytics**.

---

## 🔄 Next Phase (Under Development)

I am currently developing the **next project**:

### 🎥 Object Tracking + Vehicle Counting System

- Tracking vehicles using **ByteTrack / DeepSORT**
- Counting number of 2-wheelers passing through a region
- Counting helmet violations in real-time
- Generating traffic rule violation reports

This module will make the system fully operational for **Smart Traffic Surveillance**.

---

## 📎 Contact

**Developer:** Ahmad Pasha  
**Role:** Data Scientist / Computer Vision Engineer  
**Location:** India  

---



