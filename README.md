# Mobile Activity Monitoring  

In modern retail environments, employee productivity and customer service quality directly influence store performance. However, the increasing use of mobile phones during work hours often results in distractions, reduced efficiency, and a negative impact on customer experience.  

To address this challenge, we propose a **computer vision–based system** that can detect and classify mobile phone usage activities—such as calling, texting, or browsing—by employees during working hours. By leveraging **object detection, action recognition, and deep learning**, the system can automatically identify instances of mobile usage in real time from CCTV footage.  

This monitoring framework enables store managers to gain insights into employee activity patterns, enforce compliance with workplace policies, and ultimately ensure better customer engagement and operational efficiency.  

---

## Table of Contents  
1. [About the Data](#about-the-data)  
2. [Limitations of Pretrained Model](#limitations-of-pretrained-model)  
3. [Need for Fine-Tuning](#need-for-fine-tuning)  
4. [Preprocessing and Model Training](#preprocessing-and-model-training)  
5. [Evaluation Metrics](#evaluation-metrics)  
6. [Inference Results](#inference-results)  
7. [Workflow for Tracking Mobile Usage](#workflow-for-tracking-mobile-usage)  
8. [Installations and Code Usage](#installations-and-code-usage)  

---

## About the Data  

- Dataset contains **6 videos** in MP4, AVI, MOV, and MKV formats.  
- Split: **4 videos** for **training**, **1** for **validation**, and **1** for **testing**.  
- Annotation performed using **Roboflow** and exported in **YOLOv11** format.  

---

## Limitations of Pretrained Model  

- **Inconsistent detections** for mobile phones across frames.  
- **Unreliable worker detection** in different videos.  
- **Static phones** (e.g., left on tables) are wrongly detected as active usage.  

Examples:  
- ![Inconsistency](visuals/inconsistency.png)  
- ![Inconsistency Worker](visuals/inconsistency_worker.png)  
- ![Static Device](visuals/static.png)  

---

## Need for Fine-Tuning  

Fine-tuning is essential to adapt YOLO for this domain-specific problem:  

- **Domain Adaptation**: Trains the model to retail-specific environments and conditions.  
- **Improved Accuracy**: Reduces inconsistent detections for both workers and phones.  
- **Task-Specific Filtering**: Ignores static/idle phones, focusing only on active usage.  
- **Robustness**: Handles varied lighting, camera angles, and video quality.  

---

## Preprocessing and Model Training  

### Preprocessing  

- **Histogram Equalization** for uniform contrast.  
- **Augmentations**:  
  - Horizontal Flip → simulates mirrored views.  
  - Blur (≤0.8 px) → accounts for motion blur.  
  - Noise (≤4.2% pixels) → improves robustness.  
  - Mosaic Augmentation → better contextual learning.  
- **Bounding Box Augmentations**:  
  - Brightness adjustments (0% → +37%).  
  - Localized blur within bounding boxes (≤0.6 px).  

### Model Training  

- Model: **YOLOv11** (fine-tuned).  
- Input size: **1088 × 1088**.  
- Dataset: Annotated and exported via **Roboflow**.  

---

## Evaluation Metrics  

Tested on **180 images** from the unseen test video:  

| Class   | Precision (P) | Recall (R) | mAP@50 | mAP@50-95 |
|---------|---------------|------------|--------|-----------|
| Mobile  | 0.997         | 0.993      | 0.995  | 0.839     |
| Worker  | 0.997         | 0.999      | 0.995  | 0.897     |
| **Overall** | **0.997** | **0.996** | **0.995** | **0.868** |

**Key Insights**:  
- Very high **precision/recall ≥0.99** → reliable detection.  
- **mAP@50 = 0.995** → strong localization performance.  
- **mAP@50-95** slightly lower (expected due to stricter IoU thresholds).  
- Significant improvement over pre-trained baseline, especially in ignoring static phones.  

---

## Inference Results  

Comparison of **pre-trained vs fine-tuned model**:  

| Image | Pre-trained Model | Fine-tuned Model |
|-------|------------------|------------------|
| Sample 1 | ![Pretrained Result 1](visuals/inconsistency1.png) | ![Fine-tuned Result 1](visuals/consistency.png) |
| Sample 2 | ![Pretrained Result 2](visuals/inconsistency_worker.png) | ![Fine-tuned Result 2](visuals/consistency_worker.png) |
| Sample 3 | ![Pretrained Result 3](visuals/static.png) | ![Fine-tuned Result 3](visuals/notstatic.png) |

**Observations**:  
- Pretrained → inconsistent detection + static phone errors.  
- Fine-tuned → consistent worker detection + only active usage detected.  

---

## Workflow for Tracking Mobile Usage  

This pipeline detects and tracks **employee mobile phone usage** in retail videos.  

1. **Model Loading**  
   - Load YOLO model (`.pt` / `.onnx`).  
   - If `.onnx` missing, auto-export from `.pt`.  
   - Map classes → `worker`, `mobile`.  

2. **Frame Processing**  
   - Read video frame-by-frame.  
   - Run YOLOv11 inference.  
   - Extract bounding boxes for both classes.  

3. **Phone Usage Detection**  
   - `is_inside()` checks if mobile is within worker’s box.  
   - Worker box compressed via `compress_box()` for stricter overlap.  
   - If overlap → mark as **phone usage**.  

4. **Buffer & IoU Validation**  
   - Prevents false negatives when phones disappear briefly.  
   - Uses buffer of past frames + IoU checks.  

5. **Tracking & Statistics**  
   - Logs start & end frames of phone usage.  
   - Computes:  
     - Total usage time (s).  
     - Frames with usage.  
     - Usage % of total video.  

6. **Overlay on Video**  
   - Bounding boxes:  
     -  Red : Worker using phone.  
   - Live overlay:  
     - Phone Usage Time (s).  
     - FPS.  
   - Semi-transparent background improves readability.  

7. **Export Results**  
   - Annotated video : `output_videos/`.  
   - CSV summary with columns:  
   ```
| video_name | start_frame | end_frame | start_sec | end_sec | total_frames | frames_with_phone | usage_percentage |
|------------|-------------|-----------|-----------|---------|--------------|-------------------|------------------|
| video1.mp4 | 36          | 96        | 3.0       | 8.0     | 120          | 60                | 50.0%            |


   

