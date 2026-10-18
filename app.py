import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import requests

# Scikit-Learn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, mean_squared_error, r2_score,
    classification_report, confusion_matrix
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

# UCI & MongoDB & Groq
try:
    from ucimlrepo import fetch_ucirepo
except ImportError:
    fetch_ucirepo = None

try:
    import pymongo
    from bson.objectid import ObjectId
except ImportError:
    pymongo = None

try:
    from groq import Groq
except ImportError:
    Groq = None


st.set_page_config(page_title="Aplikasi AI & Streamlit TAS", layout="wide", page_icon="🤖")

st.title("🤖 Aplikasi Machine Learning, Database & LLM Integrasi")
st.caption("Proyek Akhir AI: Regresi, Klasifikasi, Clustering, PyTorch, MongoDB Atlas & Groq LLM")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 ML Supervised", 
    "🎯 Clustering", 
    "🧠 Deep Learning", 
    "🗄️ MongoDB & Retrain", 
    "💬 Komentar & Groq LLM"
])

# ==========================================
# TAB 1: MACHINE LEARNING (REGRESI & KLASIFIKASI)
# ==========================================
with tab1:
    st.header("1. Regresi & Klasifikasi Supervised Learning")
    
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        task_type = st.radio("Pilih Jenis Tugas ML:", ["Regresi", "Klasifikasi"])
        dataset_source = st.selectbox("Pilih Sumber Dataset:", ["SKLearn Toy Dataset", "UCI Repository (Heart Disease)", "Real-time Simulation Data"])
    
    # Load Dataset
    X, y, feature_names = None, None, []
    if dataset_source == "SKLearn Toy Dataset":
        if task_type == "Regresi":
            ds_name = st.selectbox("Pilih Dataset Regresi:", ["Diabetes", "California Housing"])
            if ds_name == "Diabetes":
                data = datasets.load_diabetes(as_frame=True)
            else:
                data = datasets.fetch_california_housing(as_frame=True)
            X, y, feature_names = data.data, data.target, data.feature_names
        else:
            ds_name = st.selectbox("Pilih Dataset Klasifikasi:", ["Iris", "Wine", "Breast Cancer"])
            if ds_name == "Iris":
                data = datasets.load_iris(as_frame=True)
            elif ds_name == "Wine":
                data = datasets.load_wine(as_frame=True)
            else:
                data = datasets.load_breast_cancer(as_frame=True)
            X, y, feature_names = data.data, data.target, data.feature_names
            
    elif dataset_source == "UCI Repository (Heart Disease)":
        if fetch_ucirepo:
            try:
                heart_ds = fetch_ucirepo(id=45)
                X_df = heart_ds.data.features.dropna()
                y_df = heart_ds.data.targets.loc[X_df.index]
                X = X_df.values
                y = (y_df.values.ravel() > 0).astype(int) # Binary classification
                feature_names = list(X_df.columns)
                st.success("Berhasil mengambil dataset Heart Disease dari UCI Repository!")
            except Exception as e:
                st.error(f"Gagal memuat dataset UCI: {e}")
        else:
            st.warning("Library 'ucimlrepo' belum terpasang.")
            
    else: # Real-time simulation
        st.info("Mengambil simulasi data sensor/real-time ter-update.")
        np.random.seed(42)
        X = np.random.randn(200, 4)
        feature_names = ["Sensor_1", "Sensor_2", "Sensor_3", "Sensor_4"]
        if task_type == "Regresi":
            y = 2.5 * X[:, 0] - 1.5 * X[:, 1] + np.random.randn(200) * 0.5
        else:
            y = (X[:, 0] + X[:, 1] > 0).astype(int)

    if X is not None and y is not None:
        with col_sb2:
            test_size = st.slider("Test Size Ratio", 0.1, 0.4, 0.2, key="ml_test_size")
            if task_type == "Regresi":
                model_name = st.selectbox("Pilih Model Regresi:", ["Linear Regression", "Ridge", "Lasso", "Decision Tree", "Random Forest"])
            else:
                model_name = st.selectbox("Pilih Model Klasifikasi:", ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "SVM"])
            
            btn_train = st.button("🚀 Train Model ML", key="btn_train_ml")

        if btn_train:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Model Selection
            if task_type == "Regresi":
                models = {
                    "Linear Regression": LinearRegression(),
                    "Ridge": Ridge(),
                    "Lasso": Lasso(),
                    "Decision Tree": DecisionTreeRegressor(random_state=42),
                    "Random Forest": RandomForestRegressor(random_state=42)
                }
                model = models[model_name]
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_test_scaled)
                
                mse = mean_squared_error(y_test, preds)
                r2 = r2_score(y_test, preds)
                
                st.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
                st.metric("R² Score", f"{r2:.4f}")
                
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(y_test, preds, alpha=0.7, color="blue")
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                ax.set_xlabel("Nilai Aktual")
                ax.set_ylabel("Nilai Prediksi")
                ax.set_title("Actual vs Predicted Scatter Plot")
                st.pyplot(fig)
                
            else:
                models = {
                    "Logistic Regression": LogisticRegression(),
                    "KNN": KNeighborsClassifier(),
                    "Decision Tree": DecisionTreeClassifier(random_state=42),
                    "Random Forest": RandomForestClassifier(random_state=42),
                    "SVM": SVC()
                }
                model = models[model_name]
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_test_scaled)
                
                acc = accuracy_score(y_test, preds)
                st.metric("Akurasi Model", f"{acc * 100:.2f}%")
                
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.text("Classification Report:")
                    st.code(classification_report(y_test, preds))
                with col_res2:
                    fig, ax = plt.subplots(figsize=(4, 3))
                    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt="d", cmap="Blues", ax=ax)
                    ax.set_title("Confusion Matrix")
                    st.pyplot(fig)

# ==========================================
# TAB 2: CLUSTERING (K-MEANS)
# ==========================================
with tab2:
    st.header("2. Unsupervised Learning: K-Means Clustering")
    
    st.write("Mengelompokkan data tanpa label berdasarkan kemiripan fiturnya.")
    cluster_k = st.slider("Pilih Jumlah Cluster (K):", 2, 8, 3)
    
    data_iris = datasets.load_iris(as_frame=True)
    X_cluster = data_iris.data.iloc[:, :2] # Ambil 2 fitur pertama untuk visualisasi 2D
    
    kmeans = KMeans(n_clusters=cluster_k, random_state=42)
    labels = kmeans.fit_predict(X_cluster)
    
    fig_c, ax_c = plt.subplots(figsize=(7, 4))
    scatter = ax_c.scatter(X_cluster.iloc[:, 0], X_cluster.iloc[:, 1], c=labels, cmap="viridis", s=50)
    ax_c.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroid')
    ax_c.set_xlabel(X_cluster.columns[0])
    ax_c.set_ylabel(X_cluster.columns[1])
    ax_c.set_title(f"Visualisasi K-Means dengan K={cluster_k}")
    ax_c.legend()
    st.pyplot(fig_c)

# ==========================================
# TAB 3: DEEP LEARNING (PYTORCH)
# ==========================================
with tab3:
    st.header("3. Deep Learning Sederhana Menggunakan PyTorch")
    
    st.write("Melatih Neural Network ringan (Multi-Layer Perceptron) untuk klasifikasi Iris.")
    epochs = st.slider("Jumlah Epoch Training:", 10, 200, 50, step=10)
    lr = st.select_slider("Learning Rate:", options=[0.001, 0.01, 0.05, 0.1], value=0.01)
    
    if st.button("🔥 Train Neural Network (PyTorch)"):
        iris_dl = datasets.load_iris()
        X_dl = torch.tensor(iris_dl.data, dtype=torch.float32)
        y_dl = torch.tensor(iris_dl.target, dtype=torch.long)
        
        # Sederhana MLP Model
        class SimpleMLP(nn.Module):
            def __init__(self):
                super(SimpleMLP, self).__init__()
                self.fc1 = nn.Linear(4, 16)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(16, 3)
                
            def forward(self, x):
                return self.fc2(self.relu(self.fc1(x)))
        
        model_nn = SimpleMLP()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_nn.parameters(), lr=lr)
        
        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model_nn(X_dl)
            loss = criterion(outputs, y_dl)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        st.success("Training PyTorch Selesai!")
        
        fig_dl, ax_dl = plt.subplots(figsize=(6, 3))
        ax_dl.plot(range(1, epochs + 1), losses, color="orange")
        ax_dl.set_xlabel("Epoch")
        ax_dl.set_ylabel("Loss")
        ax_dl.set_title("Grafik Loss Training Neural Network")
        st.pyplot(fig_dl)

# ==========================================
# TAB 4: MONGODB CRUD & RETRAIN MODEL
# ==========================================
with tab4:
    st.header("4. MongoDB Atlas Integration & Model Retraining")
    
    mongodb_uri = st.secrets.get("MONGODB_URI", "")
    db_connected = False
    
    if mongodb_uri and pymongo:
        try:
            client = pymongo.MongoClient(mongodb_uri)
            db = client["test_db"]
            iris_col = db["iris_db"]
            db_connected = True
            st.success("✅ Terhubung ke MongoDB Atlas!")
        except Exception as e:
            st.error(f"Gagal koneksi MongoDB: {e}")
    else:
        st.info("💡 MONGODB_URI belum dikonfigurasi di secrets.toml. Menggunakan simulasi lokal.")

    st.subheader("Tambah Data Iris Baru ke MongoDB")
    with st.form("form_iris"):
        c1, c2 = st.columns(2)
        sl = c1.number_input("Sepal Length", value=5.1)
        sw = c2.number_input("Sepal Width", value=3.5)
        pl = c1.number_input("Petal Length", value=1.4)
        pw = c2.number_input("Petal Width", value=0.2)
        species = st.selectbox("Species", ["setosa", "versicolor", "virginica"])
        
        btn_save = st.form_submit_button("💾 Simpan ke MongoDB")
        if btn_save:
            data_doc = {
                "sepal_length": sl, "sepal_width": sw,
                "petal_length": pl, "petal_width": pw,
                "species": species, "timestamp": datetime.now()
            }
            if db_connected:
                iris_col.insert_one(data_doc)
                st.success("Data berhasil disimpan ke MongoDB Atlas!")
            else:
                st.success("Simulasi simpan data berhasil (Lokal).")

    st.subheader("Data Tersimpan & Retrain Model")
    if db_connected:
        docs = list(iris_col.find())
        if docs:
            df_mongo = pd.DataFrame(docs)
            if "_id" in df_mongo.columns:
                df_mongo["_id"] = df_mongo["_id"].astype(str)
            st.dataframe(df_mongo)
            
            if st.button("🔄 Retrain Model dengan Data Gabungan"):
                # Data Asli
                iris_base = datasets.load_iris(as_frame=True)
                df_base = iris_base.frame
                species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
                df_base["species"] = df_base["target"].map(species_map)
                
                # Gabung data MongoDB
                df_combined = pd.concat([
                    df_base[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)", "species"]],
                    df_mongo[["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]].rename(columns={
                        "sepal_length": "sepal length (cm)", "sepal_width": "sepal width (cm)",
                        "petal_length": "petal length (cm)", "petal_width": "petal width (cm)"
                    })
                ], ignore_index=True)
                
                rev_map = {"setosa": 0, "versicolor": 1, "virginica": 2}
                X_retrain = df_combined.drop(columns=["species"])
                y_retrain = df_combined["species"].map(rev_map)
                
                rf_retrained = RandomForestClassifier()
                rf_retrained.fit(X_retrain, y_retrain)
                st.success(f"Model berhasil di-retrain dengan total {len(df_combined)} sampel data!")
    else:
        st.write("Total dokumen simulasi: 0 data tambahan.")

# ==========================================
# TAB 5: KOMENTAR PUBLIK & GROQ LLM
# ==========================================
with tab5:
    st.header("5. Komentar Publik & Auto-Summary / Analisis Sentimen via Groq LLM")
    
    groq_api_key = st.secrets.get("GROQ_API_KEY", "")
    
    st.subheader("💬 Tulis Komentar Publik")
    with st.form("form_comment"):
        user_name = st.text_input("Nama Anda:")
        user_comment = st.text_area("Komentar / Masukan Anda:")
        btn_comment = st.form_submit_button("Kirim Komentar")
        
        if btn_comment and user_comment:
            comment_doc = {"nama": user_name, "komentar": user_comment, "timestamp": datetime.now()}
            if db_connected:
                db["comments_db"].insert_one(comment_doc)
                st.success("Komentar disimpan ke MongoDB!")
            else:
                st.success("Komentar tersimpan (Simulasi lokal)!")

    st.subheader("📋 Daftar Komentar & Analisis Groq LLM")
    comments_list = []
    if db_connected:
        comments_list = list(db["comments_db"].find())
        for c in comments_list:
            st.write(f"**{c.get('nama', 'Anonim')}**: {c.get('komentar', '')}")
            
    if st.button("🤖 Analisa Semua Komentar dengan Groq LLM"):
        if groq_api_key and Groq:
            try:
                client_groq = Groq(api_key=groq_api_key)
                all_text = "\n".join([c.get('komentar', '') for c in comments_list]) if comments_list else "Aplikasi ini sangat bagus dan interaktif!"
                
                prompt = f"Tolong buatkan ringkasan singkat dan analisis sentimen (Positif/Netral/Negatif) untuk sekumpulan komentar berikut:\n{all_text}"
                
                completion = client_groq.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                )
                
                st.markdown("### Hasil Analisis Groq LLM:")
                st.write(completion.choices[0].message.content)
            except Exception as e:
                st.error(f"Error Groq API: {e}")
        else:
            st.warning("GROQ_API_KEY belum dikonfigurasi di file secrets.toml. Masukkan API key Groq terlebih dahulu.")