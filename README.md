---
title: AI7 Diabetes Prediction
emoji: 🩺
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
---

# AI7: Diabetes Prediction with Backpropagation ANN

A professional medical screening web application built with Flask and powered by a Backpropagation Neural Network (ANN) trained on 100,000 clinical records.

## 🌟 Features
- **High Recall (89%+):** Optimized for medical screening to minimize False Negatives.
- **Modern UI:** Clean medical dashboard using Bootstrap 5 and PMI (Red Cross) aesthetic.
- **Real-time Prediction:** Instant risk analysis based on clinical parameters (HbA1c, BMI, Blood Glucose, etc.).
- **User Education:** Explanatory tooltips for medical terms like HbA1c to help lay users.

## 🛠️ Technology Stack
- **Backend:** Flask (Python)
- **Model:** TensorFlow/Keras (Sequential ANN)
- **Frontend:** HTML5, CSS3 (Medical Theme), JavaScript (AJAX)
- **Deployment:** Docker & Hugging Face Spaces

## 📊 Model Performance
| Metric | Score |
| :--- | :--- |
| **Accuracy** | 91.33% |
| **Recall (Sensitivity)** | 90.11% |
| **Precision** | 49.45% |

## 🚀 Deployment Instructions
This project is containerized using Docker. To run locally:
```bash
docker build -t diabetes-prediction .
docker run -p 7860:7860 diabetes-prediction
```

## 📖 Research Report
The full research report following the **Computing Journal Informatika** template is available in `Laporan_Diabetes.md`.

---
*Developed for AI Medical Research Project.*
