# Mobile Activity Monitoring
In modern retail environments, employee productivity and customer service quality play a crucial role in overall store performance. However, the increasing use of mobile phones during work hours can lead to distractions, reduced efficiency, and a negative impact on customer experience. To address this challenge, mobile activity detection systems are being developed to monitor and analyze employee behavior in retail stores.

The objective of this project is to build an intelligent computer vision–based system that can detect and classify mobile phone usage activities, such as calling, texting, or browsing—by employees during working hours. Using techniques like object detection, action recognition, and deep learning models, the system can automatically identify instances of mobile usage in real time from CCTV footage.

Such monitoring enables store managers to gain insights into employee activity patterns, enforce compliance with workplace policies, and ensure better customer engagement, ultimately improving operational efficiency and customer satisfaction.

1. [About the Data](#about-the-data)
2. [Limitations With Pretrained Model](#limitations-with-pretrained-model)
3. [Need of fine-tuning](#need-for-fine-tuning)
4. [Model and Preprocessing](#model-and-preprocessing)
5. [Training Configuration](#training-configuration)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Inference Results](#inference-results)


## About the Data
- The dataset consists of **6 videos** in formats such as MP4, AVI, MOV, and MKV.
- Out of these, **4 videos** were used for **training**, **1 video** for **validation** and **1 video** for **testing** on unseen data.
- The dataset was annotated using **Roboflow**, and the annotations were exported in the **YOLOv11** format.

## Limitations With Pretrained Model
- The pre-trained model shows **inconsistent performance** in detecting mobile phones, making it **unreliable** for tracking actual mobile usage by workers.
![Inconsistency](visuals/inconsistency.png)
- Worker detection is also **inconsistent** across different videos, leading to gaps in accurate monitoring.
  ![Inconsistency Worker](visuals/inconsistency_worker.png)
- The model detects **static/idle** mobile phones (e.g., placed on tables), which are **not relevant** for this project, since the focus is only on **active mobile usage** by workers.
  ![Static Device](visuals/static.png)
