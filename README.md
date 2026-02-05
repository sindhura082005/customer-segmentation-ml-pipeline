## Customer Segmentation & Analytics Platform- K Means

An **end-to-end customer segmentation project** built using **Python, Machine Learning, and Streamlit**.  
This application allows users to upload customer data and automatically segment customers into meaningful groups using a **pre-trained K-Means clustering model**, with PCA-based visualizations and downloadable results.

---

## 🚀 Project Overview

Customer segmentation helps businesses understand customer behavior and design targeted marketing strategies.  
This project demonstrates the complete analytics pipeline:

- Data preprocessing & feature engineering  
- Unsupervised learning using K-Means  
- Dimensionality reduction with PCA  
- Deployment as an interactive Streamlit application  

---

## ✨ Key Features

- Upload CSV or Excel customer datasets  
- Automatic customer segmentation using K-Means  
- Advanced feature engineering (Age, Family Size, Total Spend, etc.)  
- PCA-based cluster visualization  
- Cluster-level summary insights  
- Download segmented customer data  

---

## 🧠 Problem Statement

Businesses often struggle to identify distinct customer groups based on behavior and demographics.  
This project solves that problem by clustering customers using purchasing patterns, engagement metrics, and demographic features—enabling **data-driven decision making**.

---

## 🗂️ Project Structure

---
| File / Folder                | Description                                   |
|------------------------------|-----------------------------------------------|
| app.py                       | Streamlit application for customer segmentation |
| Customer_Segmentation.ipynb  | Data preprocessing, feature engineering, and model training |
| marketing_campaign.xlsx     | Sample customer dataset used for analysis     |
| kmeans_model.pkl             | Trained K-Means clustering model              |
| scaler.pkl                   | Feature scaler used during model training     |
| requirements.txt             | Python dependencies required to run the project |


--

## 🛠️ Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn (K-Means, PCA, StandardScaler)  
- Matplotlib  
- Streamlit  
- Jupyter Notebook  

---

## 🔄 Workflow

1. Upload customer dataset (CSV / Excel)  
2. Data preprocessing and feature engineering  
3. Feature scaling using StandardScaler  
4. Dimensionality reduction with PCA  
5. Customer segmentation using K-Means  
6. Visualization and cluster summary  
7. Download segmented output  

---

## 📊 Model Details

- **Algorithm:** K-Means Clustering  
- **Preprocessing:** StandardScaler  
- **Dimensionality Reduction:** PCA  
- **Saved Artifacts:**  
  - `kmeans_model.pkl`  
  - `scaler.pkl`  

The trained model and scaler are loaded directly into the Streamlit app for real-time predictions.

---
## 📈 Output

- Cluster-wise customer summary table
- PCA-based 2D visualization of customer clusters
- Downloadable CSV file with assigned customer segments



