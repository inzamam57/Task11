# Breast Cancer Classification using SVM

## 📌 Project Overview
This project demonstrates an end-to-end machine learning workflow for **Breast Cancer classification**
using a **Support Vector Machine (SVM)** model. The model performance is evaluated using the
**ROC Curve** and **AUC score**.

The dataset used is synthetically generated and inspired by the Breast Cancer Wisconsin dataset.

---

## 📂 Project Files
- breast_cancer_400_entries.csv – Dataset with 400 samples  
- svm_breast_cancer_pipeline.pkl – Saved trained SVM pipeline  
- Breast_Cancer_SVM_Report.pdf – Final project report (PDF)  
- Breast_Cancer_SVM_Report.docx – Final project report (DOCX)  
- README.md – Project documentation  

---

## ⚙️ Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Joblib  

---

## 🧪 Workflow Steps
1. Load and inspect the dataset  
2. Encode target labels (Benign / Malignant)  
3. Normalize feature values using StandardScaler  
4. Split data into training and testing sets  
5. Train SVM with Linear and RBF kernels  
6. Tune hyperparameters using GridSearchCV  
7. Evaluate model using:
   - Confusion Matrix
   - Classification Report
   - ROC Curve
   - AUC Score
8. Save the trained pipeline for reuse  

---

## 📊 Model Performance
- Algorithm: Support Vector Machine (RBF Kernel)  
- AUC Score: 0.39  

> Note: Lower AUC is expected due to synthetic data.

---

## ▶️ How to Run
```bash
pip install pandas numpy scikit-learn matplotlib joblib
