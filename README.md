# Customer Segmentation & Analytics Platform

An end-to-end Unsupervised Machine Learning pipeline for intelligent customer segmentation using multiple clustering algorithms, automated model evaluation, PCA-based visualization, and deployment via Streamlit.

---

## 🚀 Project Overview

Businesses often struggle to identify distinct customer groups based on purchasing behavior and demographics.  
This project builds a production-ready ML pipeline that segments customers into meaningful clusters using advanced unsupervised learning techniques.

The system includes:

- Data preprocessing & feature engineering
- Multi-model clustering comparison
- Automated model selection framework
- PCA-based dimensionality reduction
- Interactive Streamlit deployment
- Downloadable segmented outputs

---

## 🧠 Problem Statement

Segment customers based on demographic and behavioral data to enable:

- Targeted marketing strategies
- Personalized offers
- Revenue optimization
- Customer retention planning

---

## ⚙️ ML Pipeline Architecture

1. Data Cleaning & Preprocessing
   - Missing value handling
   - IQR-based outlier treatment
   - Correlation-based feature reduction

2. Feature Engineering
   - Age calculation from Year_Birth
   - TotalSpend aggregation
   - FamilySize computation
   - IsPartner binary feature

3. Feature Scaling
   - StandardScaler for normalization

4. Dimensionality Reduction
   - PCA (95% variance retention)
   - PCA loadings analysis

5. Clustering Algorithms Compared
   - KMeans
   - Agglomerative Clustering
   - DBSCAN
   - Gaussian Mixture Models
   - MiniBatch KMeans

6. Model Evaluation Metrics
   - Silhouette Score
   - Davies-Bouldin Index
   - Calinski-Harabasz Score
   - Custom ranking-based model selection

7. Deployment
   - Streamlit-based interactive web app
   - Real-time dataset upload
   - Cluster summary insights
   - Downloadable segmented CSV output

---

## 📊 Results

- Segmented 2,240 customers into 4 distinct behavioral clusters
- Achieved optimal Silhouette Score of 0.23 using KMeans
- Reduced 32 features to 21 principal components while retaining 95% variance
- Automated model comparison framework improved selection reliability

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



---

## 🛠 Tech Stack

- Python
- Pandas, NumPy
- Scikit-Learn
- PCA
- Matplotlib
- Streamlit
- Pickle

---

## 📈 Key Highlights

- Built a reusable ML pipeline for unsupervised segmentation
- Implemented automated multi-model comparison framework
- Designed cluster-level business insights dashboard
- Production-ready deployment with serialized model artifacts

---
## 🧩 MLOps & Production Readiness

- Model serialization using Pickle for reusable deployment
- Modular ML pipeline separating preprocessing, training, and inference logic
- Reproducible preprocessing workflow (feature engineering + scaling consistency)
- Dependency management using requirements.txt
- Version-controlled project structure for collaborative development
- Production-ready Streamlit interface for real-time inference
---

## 🎯 Business Impact

- Identifies high-value vs budget-conscious customer segments
- Enables data-driven marketing decisions
- Improves campaign targeting efficiency
- Supports revenue growth strategies

---

## 🎥 Live Demo

Watch the demo here:

"C:\Users\91888\Downloads\streamlit-app-2026-02-12-14-34-35.webm"

----
## 📷 Application Preview

<img width="1600" height="759" alt="image" src="https://github.com/user-attachments/assets/063de9e4-99d0-46bd-9a5b-d9e082afc929" />
<img width="1379" height="1050" alt="image" src="https://github.com/user-attachments/assets/23f5703a-cc30-49b6-b7af-c9954ad98332" />




