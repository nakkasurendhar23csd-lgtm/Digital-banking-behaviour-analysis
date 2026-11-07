# Team 9 – Digital Banking Behaviour Analysis
### Customer Segmentation, Behavioural Pattern Discovery & Comparative Machine Learning (K-Means, Hierarchical Clustering, PCA)

An enterprise-grade, end-to-end machine learning and analytics solution built to analyze digital banking customer touchpoints, segment customers by financial capacity and digital channel engagement, compare unsupervised clustering methodologies, and deliver an interactive executive dashboard.

---

## 🌟 Key Highlights & Deliverables
1. **Interactive Digital Banking Behaviour Dashboard (`app.py`)**:
   - Built with Streamlit & Plotly.
   - Live URL: `http://localhost:8501`
   - Features 5 comprehensive tabs:
     - **📊 Exploratory Behaviour Analytics**: 24-hour transaction distributions, branch vs. 24/7 digital window, balance vs spend, age dynamics.
     - **🤖 Clustering & Model Comparison**: K-Means Elbow curve, Silhouette plots, Agglomerative Dendrogram, and side-by-side algorithmic benchmark matrix.
     - **🌐 PCA & Dimensionality Reduction**: Scree plots, factor loadings heatmap, 2D and 3D interactive projections.
     - **👥 Behavioural Segments & Strategic Playbook**: Spider/radar charts, wallet share breakdown, and in-depth business playbooks for each segment.
     - **🎯 Real-Time Customer Profiler & Recommender**: Interactive form to input custom customer indicators, predict segment in real-time, position in PCA space, and deliver personalized next-best-action (NBA) offers.

2. **Comparative Machine Learning Suite (`src/clustering_models.py`)**:
   - **K-Means (k-means++)**: Scalable, centroid-based partitional clustering ($k=4$).
   - **Hierarchical Clustering (Agglomerative)**: Ward and Average linkage models with tree dendrogram visualization and cutting thresholds.
   - **Principal Component Analysis (PCA)**: Dimensionality reduction preserving >60% of variance across 3 principal components (Digital Engagement, Financial Affluence, Transaction Velocity).
   - **Rigorous Evaluation Metrics**: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, and computational complexity analysis.

3. **Complete Jupyter Data Science Story (`digital_banking_analysis.ipynb`)**:
   - Structured following strict ML best practices: every code block is paired with analytical markdown interpretations, visualizations, and an overarching executive conclusion.

4. **Robust Data Pipeline (`src/data_processing.py`)**:
   - Ingests Kaggle's *Bank Customer Segmentation (1M+ Transactions)* dataset (`bank_transactions.csv`).
   - Extracts RFM metrics and digital proxy indicators (e.g. 24/7 off-peak transactions outside branch hours 10 AM - 4 PM, weekend activity, velocity).
   - Augments dataset with clearly labelled digital activity metrics (mobile app logins, desktop web sessions, product diversity, composite digital score) in direct compliance with the project guidelines.

---

## 👥 Discovered Banking Personas & Strategic Playbook

| Cluster ID | Persona Name | Key Behavioural Markers | Strategic Action Playbook |
| :--- | :--- | :--- | :--- |
| **Cluster 0** | **💎 Affluent Digital Premium Transactors** | Avg Balance: ₹2,20,000+; High ticket spend (₹3,600+); **96.8% off-peak 24/7 digital usage**; high mobile & web logins. | Cross-sell digital wealth management, portfolio advisory, high-yield smart FDs, and metal travel credit cards with concierge support. |
| **Cluster 1** | **🏛️ Traditional Branch-Bound Savers** | Avg Balance: ₹99,000+; **<0.5% off-peak activity (strictly daytime branch hours)**; low app logins; lowest digital engagement (14.4/100). | Assisted digital migration via in-branch tablet onboarding, biometric security reassurance, and relationship manager outreach for long-term deposits. |
| **Cluster 2** | **⚡ Omni-Channel Multi-Product Users** | **2x transaction frequency & velocity**; highest product count (avg 2.93 digital services); balanced daytime/night digital touchpoints. | Lock in multi-product ecosystem through rewards, digital Demat accounts, and instant pre-approved auto/home loans. |
| **Cluster 3** | **📱 Mobile-First Micro-Transactors** | **Highest mobile app logins (21.4/mo)**; 99.9% off-peak smartphone transactions; modest balance (₹31,700); micro-spends (₹290 avg). | Deploy instant Buy-Now-Pay-Later (BNPL), UPI scan-and-pay cashback incentives, micro-SIP roundups, and merchant perks (food/OTT). |

---

## ⚖️ Algorithmic Comparison Matrix

| Evaluation Metric | K-Means (k-means++) | Hierarchical (Ward Linkage) | Hierarchical (Average Linkage) |
| :--- | :--- | :--- | :--- |
| **Silhouette Score** *(Higher is better)* | **0.2320** | 0.2252 | 0.4313 (biased by singleton chains) |
| **Davies-Bouldin Index** *(Lower is better)* | **1.4246** | 1.7870 | 0.7078 |
| **Calinski-Harabasz Score** *(Higher is better)* | **948.6** | 817.5 | 249.0 |
| **Time Complexity** | **$O(k \cdot n \cdot d)$** | $O(n^2 \log n)$ | $O(n^2)$ |
| **Space / Memory Complexity** | **$O(n \cdot d)$** | $O(n^2)$ distance matrix | $O(n^2)$ distance matrix |
| **Production Recommendation** | **Primary Engine for Production** (scalable to 1M+ transactions & real-time API scoring) | **Taxonomy Discovery & Visualization** (confirms tree hierarchy and optimal cut) | Outlier detection & irregular density separation |

---

## 🛠️ Project Structure
```
digital_banking_behaviour_analysis/
├── .venv/                         # Isolated Python 3.13 virtual environment
├── app.py                         # Interactive Streamlit dashboard
├── digital_banking_analysis.ipynb # Comprehensive Jupyter Notebook
├── requirements.txt               # Pinned project dependencies
├── README.md                      # Project documentation and playbook
├── src/
│   ├── __init__.py
│   ├── data_processing.py         # Data cleaning, RFM & digital feature extraction
│   ├── clustering_models.py       # K-Means, Hierarchical (Ward/Average), PCA & metrics
│   ├── run_analysis.py            # Automated pipeline & artifact serialization
│   └── create_notebook.py         # Script generating standard .ipynb format
├── data/
│   ├── processed_customers.parquet
│   ├── customers_segmented.parquet
│   ├── elbow_silhouette_metrics.csv
│   ├── model_comparison.csv
│   ├── pca_loadings.csv
│   └── pca_variance.csv
└── models/
    ├── scaler.joblib
    ├── kmeans_model.joblib
    ├── pca_model.joblib
    ├── dendrogram_linkage.joblib
    └── personas.joblib
```

---

## 🚀 How to Run

### 1. Launch the Interactive Dashboard
The Streamlit application is currently running at:
**[http://localhost:8501](http://localhost:8501)**

To launch or restart manually:
```bash
cd C:\Users\sanju\.gemini\antigravity\scratch\digital_banking_behaviour_analysis
.venv\Scripts\streamlit run app.py --server.port 8501
```

### 2. Run the Notebook
Open `digital_banking_analysis.ipynb` in VS Code / Jupyter:
```bash
.venv\Scripts\jupyter notebook digital_banking_analysis.ipynb
```

### 3. Re-train Models or Process Fresh Data
```bash
.venv\Scripts\python -m src.run_analysis
```

