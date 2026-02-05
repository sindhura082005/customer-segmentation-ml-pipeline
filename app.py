import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from datetime import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Segmentation Using K-Means",
    layout="wide"
)

st.title("📊 Customer Segmentation Using K-Means")
st.write("Upload customer data to predict customer segments.")

# ---------------- LOAD MODEL & SCALER ----------------
@st.cache_resource
def load_artifacts():
    with open("kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return kmeans, scaler

kmeans, scaler = load_artifacts()

# ---------------- PREPROCESSING ----------------
def preprocess_data(df):
    df = df.copy()

    # ---- Clean categorical strings ----
    df["Education"] = df["Education"].astype(str).str.strip()
    df["Marital_Status"] = df["Marital_Status"].astype(str).str.strip()

    # ---- Feature engineering ----
    df["Age"] = datetime.now().year - df["Year_Birth"]
    df["TotalChildren"] = df["Kidhome"] + df["Teenhome"]

    df["IsPartner"] = df["Marital_Status"].isin(
        ["Married", "Together"]
    ).astype(int)

    df["FamilySize"] = df["TotalChildren"] + df["IsPartner"] + 1

    spend_cols = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds"
    ]
    df["TotalSpend"] = df[spend_cols].sum(axis=1)

    # ---- Encode categoricals ----
    education_map = {
        "Basic": 0,
        "2n Cycle": 1,
        "Graduation": 2,
        "Master": 3,
        "PhD": 4
    }
    df["Education"] = df["Education"].map(education_map)

    marital_map = {
        "Single": 0,
        "Together": 1,
        "Married": 2,
        "Divorced": 3,
        "Widow": 4
    }
    df["Marital_Status"] = df["Marital_Status"].map(marital_map)

    # ---- Handle missing values ----
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload Customer Dataset (CSV or Excel)",
    type=["csv", "xlsx"]
)

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # ---------------- PROCESS DATA ----------------
    df_processed = preprocess_data(df)

    # Features used during training
    features = scaler.feature_names_in_

    missing_cols = set(features) - set(df_processed.columns)
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.stop()

    X = df_processed[features]

    # ---------------- SCALE ----------------
    X_scaled = scaler.transform(X)

    # ---------------- PCA (MATCH KMEANS INPUT) ----------------
    n_pca_components = kmeans.cluster_centers_.shape[1]
    pca = PCA(n_components=n_pca_components)
    X_pca = pca.fit_transform(X_scaled)

    # ---------------- PREDICT CLUSTERS ----------------
    df["Cluster"] = kmeans.predict(X_pca)

    # ---------------- CLUSTER SUMMARY ----------------
    st.subheader("📌 Cluster Summary")

    summary = (
        df_processed.assign(Cluster=df["Cluster"])
        .groupby("Cluster")
        .agg(
            Income=("Income", "mean"),
            Total_Spend=("TotalSpend", "mean"),
            Recency=("Recency", "mean"),
            Purchases=("NumStorePurchases", "mean")
        )
        .round(2)
    )

    st.dataframe(summary)

    # ---------------- PCA VISUALIZATION ----------------
    st.subheader("📉 Cluster Visualization (PCA)")

    pca_vis = PCA(n_components=2)
    vis_data = pca_vis.fit_transform(X_scaled)

    vis_df = pd.DataFrame(vis_data, columns=["PC1", "PC2"])
    vis_df["Cluster"] = df["Cluster"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for c in sorted(vis_df["Cluster"].unique()):
        cluster_data = vis_df[vis_df["Cluster"] == c]
        ax.scatter(
            cluster_data["PC1"],
            cluster_data["PC2"],
            label=f"Cluster {c}",
            alpha=0.7
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    st.pyplot(fig)

    # ---------------- DOWNLOAD ----------------
    st.subheader("⬇️ Download Segmented Data")

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        csv,
        "customer_segments.csv",
        "text/csv"
    )
