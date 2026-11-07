"""
Interactive Digital Banking Behaviour Dashboard
Team 9 - Digital Banking Behaviour Analysis
ML Models: K-Means, Hierarchical Clustering, PCA
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram
import joblib

# Page configuration
st.set_page_config(
    page_title="Digital Banking Behaviour Analysis | Team 9",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern fintech aesthetic
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #edf2f7 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .persona-card {
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        border-left: 6px solid #2563eb;
        background: #f8fafc;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

@st.cache_data(show_spinner="Loading segmented banking dataset...")
def load_data():
    seg_file = os.path.join(DATA_DIR, 'customers_segmented.parquet')
    if not os.path.exists(seg_file):
        st.error(f"Dataset not found at {seg_file}. Please run the pipeline first.")
        st.stop()
    df = pd.read_parquet(seg_file)
    elbow_df = pd.read_csv(os.path.join(DATA_DIR, 'elbow_silhouette_metrics.csv'))
    comparison_df = pd.read_csv(os.path.join(DATA_DIR, 'model_comparison.csv'))
    loadings = pd.read_csv(os.path.join(DATA_DIR, 'pca_loadings.csv'), index_col=0)
    var_df = pd.read_csv(os.path.join(DATA_DIR, 'pca_variance.csv'))
    return df, elbow_df, comparison_df, loadings, var_df

@st.cache_resource(show_spinner="Loading machine learning models...")
def load_models():
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
    kmeans = joblib.load(os.path.join(MODELS_DIR, 'kmeans_model.joblib'))
    pca = joblib.load(os.path.join(MODELS_DIR, 'pca_model.joblib'))
    Z_linkage = joblib.load(os.path.join(MODELS_DIR, 'dendrogram_linkage.joblib'))
    personas = joblib.load(os.path.join(MODELS_DIR, 'personas.joblib'))
    return scaler, kmeans, pca, Z_linkage, personas

# Load artifacts
df, elbow_df, comparison_df, loadings, var_df = load_data()
scaler, kmeans, pca, Z_linkage, personas = load_models()

# SIDEBAR: Filtering and Mode Control
st.sidebar.image("https://images.unsplash.com/photo-1559526324-4b87b5e36e44?w=400&auto=format&fit=crop&q=60", use_container_width=True)
st.sidebar.title("Digital Banking Analytics")
st.sidebar.markdown("**Team 9** • Clustering & Behaviour")

st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Explorer Filters")

all_segments = sorted(df['Segment_Name'].dropna().unique().tolist())
selected_segments = st.sidebar.multiselect(
    "Customer Segments",
    options=all_segments,
    default=all_segments
)

channel_options = ['All'] + sorted(df['DigitalChannelPreference'].dropna().unique().tolist())
selected_channel = st.sidebar.selectbox("Digital Channel Preference", options=channel_options, index=0)

age_range = st.sidebar.slider(
    "Customer Age Range",
    min_value=int(df['Age'].min()),
    max_value=int(df['Age'].max()),
    value=(18, 75)
)

balance_filter = st.sidebar.slider(
    "Max Balance Cap (for visualization)",
    min_value=10000,
    max_value=1000000,
    value=500000,
    step=20000
)

# Apply filters
filtered_df = df[
    (df['Segment_Name'].isin(selected_segments)) &
    (df['Age'] >= age_range[0]) &
    (df['Age'] <= age_range[1]) &
    (df['AvgBalance'] <= balance_filter)
]

if selected_channel != 'All':
    filtered_df = filtered_df[filtered_df['DigitalChannelPreference'] == selected_channel]

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Note on Dataset:** Sourced from Kaggle *Bank Customer Segmentation (1M+ Transactions)* "
    "with 24/7 self-service transaction proxy indicators & augmented digital activity metrics."
)

# HEADER & TOP METRICS
st.markdown('<div class="main-header">🏦 Digital Banking Behaviour Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Customer Segmentation, Behavioural Pattern Discovery & ML Clustering Comparison (K-Means, Hierarchical, PCA)</div>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Customers Filtered</div>
        <div class="metric-val">{len(filtered_df):,}</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Avg Balance</div>
        <div class="metric-val">₹{filtered_df['AvgBalance'].mean():,.0f}</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Avg Spend</div>
        <div class="metric-val">₹{filtered_df['TotalSpend'].mean():,.0f}</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Off-Peak / 24x7 Share</div>
        <div class="metric-val">{filtered_df['OffPeakRatio'].mean()*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
with col5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Digital Engagement</div>
        <div class="metric-val">{filtered_df['DigitalEngagementScore'].mean():.1f}/100</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# MAIN TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 1. Exploratory Behaviour Analytics",
    "🤖 2. Clustering & Model Comparison",
    "🌐 3. PCA & Dimensionality Reduction",
    "👥 4. Behavioural Segments & Playbook",
    "🎯 5. Real-Time Customer Profiler"
])

# -------------------------------------------------------------
# TAB 1: EXPLORATORY BEHAVIOUR ANALYTICS
# -------------------------------------------------------------
with tab1:
    st.subheader("Digital & Financial Activity Distributions")
    st.markdown(
        "Explore how customers utilize banking channels, transaction timing, balances, and digital frequency."
    )

    t1_col1, t1_col2 = st.columns(2)
    
    with t1_col1:
        # Off-Peak vs Branch Hours comparison
        st.markdown("##### 🕒 Transaction Window: Branch Hours vs 24/7 Digital Self-Service")
        off_peak_summary = pd.DataFrame({
            'Channel Window': ['24/7 Digital Self-Service (Off-Peak/Evening/Night)', 'Physical Branch Hours (10 AM - 4 PM)'],
            'Transactions Count': [
                int((filtered_df['OffPeakRatio'] > 0.5).sum()),
                int((filtered_df['OffPeakRatio'] <= 0.5).sum())
            ]
        })
        fig_donut = px.pie(
            off_peak_summary,
            values='Transactions Count',
            names='Channel Window',
            color='Channel Window',
            color_discrete_map={
                '24/7 Digital Self-Service (Off-Peak/Evening/Night)': '#06d6a0',
                'Physical Branch Hours (10 AM - 4 PM)': '#f77f00'
            },
            hole=0.45
        )
        fig_donut.update_layout(height=340, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_donut, use_container_width=True)

    with t1_col2:
        st.markdown("##### 📱 Digital Channel Preference Breakdown")
        ch_counts = filtered_df['DigitalChannelPreference'].value_counts().reset_index()
        ch_counts.columns = ['Channel', 'Customer Count']
        fig_bar = px.bar(
            ch_counts,
            x='Channel',
            y='Customer Count',
            color='Channel',
            color_discrete_sequence=px.colors.qualitative.Prism,
            text='Customer Count'
        )
        fig_bar.update_layout(height=340, margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    t1_col3, t1_col4 = st.columns(2)

    with t1_col3:
        st.markdown("##### 💰 Account Balance vs Total Spend (Log Scale)")
        sample_viz = filtered_df.sample(min(2500, len(filtered_df)), random_state=42)
        fig_scatter = px.scatter(
            sample_viz,
            x='AvgBalance',
            y='TotalSpend',
            color='Segment_Name',
            log_x=True,
            log_y=True,
            hover_data=['CustomerID', 'MobileAppLogins', 'OffPeakRatio'],
            labels={'AvgBalance': 'Average Account Balance (₹, log)', 'TotalSpend': 'Total Spend (₹, log)'},
            color_discrete_sequence=['#0077b6', '#fb5607', '#8338ec', '#06d6a0']
        )
        fig_scatter.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with t1_col4:
        st.markdown("##### 👥 Age vs Digital Engagement Score by Gender")
        fig_age = px.scatter(
            sample_viz,
            x='Age',
            y='DigitalEngagementScore',
            color='CustGender',
            opacity=0.6,
            color_discrete_map={'M': '#2b7fff', 'F': '#ff4d6d'},
            labels={'DigitalEngagementScore': 'Digital Engagement Score (0-100)', 'Age': 'Customer Age'}
        )
        fig_age.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_age, use_container_width=True)


# -------------------------------------------------------------
# TAB 2: CLUSTERING & MODEL COMPARISON
# -------------------------------------------------------------
with tab2:
    st.subheader("Clustering Evaluation: K-Means vs Hierarchical Clustering")
    st.markdown(
        "Comprehensive model diagnostics including the Elbow Method, Silhouette Analysis, "
        "Agglomerative Dendrogram, and algorithmic comparison metrics."
    )

    t2_col1, t2_col2 = st.columns(2)
    
    with t2_col1:
        st.markdown("##### 📉 K-Means Elbow Curve (Inertia vs k)")
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=elbow_df['k'],
            y=elbow_df['inertia'],
            mode='lines+markers',
            name='Inertia (WCSS)',
            line=dict(color='#2563eb', width=3),
            marker=dict(size=9, color='#1d4ed8')
        ))
        fig_elbow.add_vline(x=4, line_dash="dash", line_color="#ef4444", annotation_text="Optimal k=4 (Elbow point)")
        fig_elbow.update_layout(
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Within-Cluster Sum of Squares (Inertia)",
            height=340,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

    with t2_col2:
        st.markdown("##### 🎯 Silhouette Score by Cluster Count (k)")
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Bar(
            x=elbow_df['k'],
            y=elbow_df['silhouette'],
            marker_color=['#93c5fd', '#3b82f6', '#2563eb', '#60a5fa', '#93c5fd'],
            text=[f"{v:.3f}" for v in elbow_df['silhouette']],
            textposition='auto'
        ))
        fig_sil.update_layout(
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Silhouette Score (Higher is Better)",
            height=340,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_sil, use_container_width=True)

    st.markdown("---")
    st.markdown("##### 🌳 Hierarchical Clustering Dendrogram (Ward Linkage)")
    st.markdown(
        "The dendrogram below illustrates the hierarchical agglomeration of customer samples. "
        "The horizontal cutoff line highlights the partition into 4 distinct clusters."
    )
    
    import matplotlib.pyplot as plt
    fig_dendro, ax = plt.subplots(figsize=(12, 3.5))
    dendrogram(
        Z_linkage,
        truncate_mode='lastp',
        p=30,
        leaf_rotation=90.,
        leaf_font_size=9.,
        show_contracted=True,
        ax=ax,
        color_threshold=15
    )
    ax.set_title("Agglomerative Hierarchical Dendrogram (Top 30 Merged Branches)")
    ax.set_xlabel("Customer Cluster Index / Merged Node Size")
    ax.set_ylabel("Ward Distance Metric")
    ax.axhline(y=15, color='r', linestyle='--', label='Cutoff Threshold (k=4)')
    ax.legend(loc='upper right')
    st.pyplot(fig_dendro)

    st.markdown("---")
    st.markdown("##### ⚖️ Head-to-Head Algorithm Comparison")
    st.table(comparison_df)

    st.markdown(r"""
    > [!NOTE]
    > **Methodological Synthesis:**
    > - **K-Means (k-means++)** demonstrates superior operational efficiency ($O(k \cdot n \cdot d)$) making it optimal for large-scale banking transaction pipelines with 1M+ transactions.
    > - **Hierarchical Clustering (Ward)** provides deterministic taxonomic grouping without random initialization variance, confirming the natural split into 4 core customer archetypes.
    > - Combining both methods validates that customer segmentation is robust, stable, and mathematically sound.
    """)


# -------------------------------------------------------------
# TAB 3: PCA & DIMENSIONALITY REDUCTION
# -------------------------------------------------------------
with tab3:
    st.subheader("Principal Component Analysis (PCA)")
    st.markdown(
        "Decomposition of multidimensional banking features into principal components, "
        "explaining behavioural variance and revealing latent customer dimensions."
    )

    t3_col1, t3_col2 = st.columns(2)
    
    with t3_col1:
        st.markdown("##### 📊 Explained & Cumulative Variance")
        fig_var = go.Figure()
        fig_var.add_trace(go.Bar(
            x=var_df['PC'],
            y=var_df['ExplainedVarianceRatio'] * 100,
            name='Individual Variance (%)',
            marker_color='#3b82f6',
            text=[f"{v*100:.1f}%" for v in var_df['ExplainedVarianceRatio']],
            textposition='auto'
        ))
        fig_var.add_trace(go.Scatter(
            x=var_df['PC'],
            y=var_df['CumulativeVariance'] * 100,
            name='Cumulative Variance (%)',
            mode='lines+markers',
            line=dict(color='#ef4444', width=3),
            marker=dict(size=8)
        ))
        fig_var.update_layout(
            yaxis_title="Variance Explained (%)",
            height=340,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", y=-0.2)
        )
        st.plotly_chart(fig_var, use_container_width=True)

    with t3_col2:
        st.markdown("##### 🧬 Principal Component Loadings Heatmap")
        fig_heat = px.imshow(
            loadings,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            labels=dict(x="Principal Component", y="Banking Feature", color="Loading")
        )
        fig_heat.update_layout(height=340, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("""
    **Interpretation of Principal Components:**
    - **PC1 (25.3% Variance) - "Digital Channel Self-Service & Mobile Engagement"**: Heavily loaded on `OffPeakRatio` (+0.47), `MobileAppLogins` (+0.47), and `DigitalEngagementScore` (+0.54). Separates branch-bound customers from 24/7 digital users.
    - **PC2 (21.5% Variance) - "Financial Affluence & Spend Capacity"**: Heavily loaded on `TotalSpend` (+0.53), `AvgSpend` (+0.52), `WebPortalSessions` (+0.41), and `AvgBalance` (+0.31). Distinguishes high-net-worth customers from micro-transactors.
    - **PC3 (13.7% Variance) - "Transaction Velocity & Frequency"**: Dominated by `TxFrequency` (+0.67) and `TxVelocity` (+0.67). Captures active daily repeat users versus one-off transactors.
    """)

    st.markdown("---")
    st.markdown("##### 🌌 2D & 3D Interactive PCA Projection")
    
    t3_p1, t3_p2 = st.columns(2)
    sample_pca = filtered_df.sample(min(3000, len(filtered_df)), random_state=42)

    with t3_p1:
        st.markdown("###### 2D PCA Space (PC1 vs PC2)")
        fig_2d = px.scatter(
            sample_pca,
            x='PCA1',
            y='PCA2',
            color='Segment_Name',
            hover_data=['CustomerID', 'AvgBalance', 'TotalSpend', 'OffPeakRatio', 'MobileAppLogins'],
            color_discrete_sequence=['#0077b6', '#fb5607', '#8338ec', '#06d6a0']
        )
        fig_2d.update_layout(
            xaxis_title="PC1: Digital Engagement & Self-Service",
            yaxis_title="PC2: Financial Affluence & Spend",
            height=420,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", y=-0.25)
        )
        st.plotly_chart(fig_2d, use_container_width=True)

    with t3_p2:
        st.markdown("###### 3D PCA Space (PC1 vs PC2 vs PC3)")
        fig_3d = px.scatter_3d(
            sample_pca,
            x='PCA1',
            y='PCA2',
            z='PCA3',
            color='Segment_Name',
            hover_data=['CustomerID', 'AvgBalance', 'TxFrequency'],
            color_discrete_sequence=['#0077b6', '#fb5607', '#8338ec', '#06d6a0'],
            opacity=0.7
        )
        fig_3d.update_layout(
            scene=dict(
                xaxis_title='PC1: Digital',
                yaxis_title='PC2: Affluence',
                zaxis_title='PC3: Velocity'
            ),
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False
        )
        st.plotly_chart(fig_3d, use_container_width=True)


# -------------------------------------------------------------
# TAB 4: BEHAVIOURAL SEGMENTS & STRATEGIC PLAYBOOK
# -------------------------------------------------------------
with tab4:
    st.subheader("Customer Behavioural Segments & Strategic Action Playbook")
    st.markdown(
        "In-depth breakdown of the four discovered customer segments, their behavioural radar, "
        "and actionable banking business strategies."
    )

    t4_col1, t4_col2 = st.columns([1, 1])

    with t4_col1:
        st.markdown("##### 🕸️ Segment Behavioural Radar (Normalized Dimensions)")
        
        # Radar chart of centroids
        radar_features = [
            'AvgBalance', 'TotalSpend', 'TxFrequency',
            'OffPeakRatio', 'MobileAppLogins', 'WebPortalSessions',
            'DigitalProductsCount'
        ]
        
        # Normalize across all segments [0, 1] for radar comparison
        seg_means = df.groupby('Segment_Name')[radar_features].mean()
        seg_norm = (seg_means - seg_means.min()) / (seg_means.max() - seg_means.min() + 1e-6)

        fig_radar = go.Figure()
        colors = ['#0077b6', '#06d6a0', '#8338ec', '#fb5607']
        categories = [
            'Account Balance', 'Total Spend', 'Tx Frequency',
            '24/7 Off-Peak %', 'Mobile Logins', 'Web Sessions', 'Products Used'
        ]

        for i, (seg_name, row) in enumerate(seg_norm.iterrows()):
            vals = row.values.tolist()
            vals.append(vals[0])  # Close polygon
            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=categories + [categories[0]],
                fill='toself',
                name=seg_name,
                line_color=colors[i % len(colors)]
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=400,
            margin=dict(l=40, r=40, t=20, b=20),
            legend=dict(orientation="h", y=-0.25)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with t4_col2:
        st.markdown("##### 📊 Segment Size & Share of Wallet")
        seg_counts = df['Segment_Name'].value_counts().reset_index()
        seg_counts.columns = ['Segment', 'Customer Count']
        fig_seg_pie = px.pie(
            seg_counts,
            names='Segment',
            values='Customer Count',
            color='Segment',
            color_discrete_sequence=['#06d6a0', '#fb5607', '#0077b6', '#8338ec'],
            hole=0.45
        )
        fig_seg_pie.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_seg_pie, use_container_width=True)

    st.markdown("---")
    st.markdown("##### 📋 Actionable Banking Personas & Playbook")

    persona_cols = st.columns(2)

    with persona_cols[0]:
        st.markdown("""
        <div class="persona-card" style="border-left-color: #0077b6;">
            <h4>💎 Affluent Digital Premium Transactors</h4>
            <span class="badge" style="background:#e0f2fe; color:#0369a1;">High Balance</span>
            <span class="badge" style="background:#dbeafe; color:#1e40af;">24/7 Digital</span>
            <span class="badge" style="background:#f1f5f9; color:#334155;">High Ticket</span>
            <p style="margin-top:10px; color:#334155;">
                <b>Profile:</b> Average balance ₹2.2L+, high average spend (₹3,600+), 96.8% off-peak/evening transactions. Active both on mobile and web banking portals.
            </p>
            <p style="color:#0f172a;"><b>Strategic Action Playbook:</b></p>
            <ul style="color:#475569; font-size:0.95rem;">
                <li><b>Wealth Management:</b> Offer robo-advisory, portfolio PMS, and high-yield digital fixed deposits.</li>
                <li><b>Premium Cards:</b> Target with metal travel/lifestyle credit cards with lounge access & forex perks.</li>
                <li><b>Priority Support:</b> Provide dedicated 24/7 in-app chat assistance and virtual relationship managers.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="persona-card" style="border-left-color: #fb5607;">
            <h4>🏛️ Traditional Branch-Bound Savers</h4>
            <span class="badge" style="background:#ffedd5; color:#c2410c;">Branch Hours Only</span>
            <span class="badge" style="background:#fef3c7; color:#b45309;">Solid Deposits</span>
            <span class="badge" style="background:#fee2e2; color:#b91c1c;">Low App Usage</span>
            <p style="margin-top:10px; color:#334155;">
                <b>Profile:</b> Average balance ₹99,000+. <b><0.5% off-peak activity</b> — all banking is conducted during physical branch hours (10 AM - 4 PM). Lowest digital engagement (14.4/100).
            </p>
            <p style="color:#0f172a;"><b>Strategic Action Playbook:</b></p>
            <ul style="color:#475569; font-size:0.95rem;">
                <li><b>Assisted Digital Migration:</b> Branch tablet onboarding to install mobile app with safety tutorials.</li>
                <li><b>Security Trust Building:</b> Biometric login demos, two-factor authentication reassurance, zero-liability guarantee.</li>
                <li><b>Branch-Assisted Products:</b> Long-term recurring deposits, senior citizen savings schemes, physical gold bonds.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with persona_cols[1]:
        st.markdown("""
        <div class="persona-card" style="border-left-color: #8338ec;">
            <h4>⚡ Omni-Channel Multi-Product Users</h4>
            <span class="badge" style="background:#f3e8ff; color:#6b21a8;">Highest Velocity</span>
            <span class="badge" style="background:#ede9fe; color:#5b21b6;">Multi-Product</span>
            <span class="badge" style="background:#e0e7ff; color:#3730a3;">High Frequency</span>
            <p style="margin-top:10px; color:#334155;">
                <b>Profile:</b> 2x higher transaction frequency and velocity than other segments. Adopts the highest number of digital products (avg 2.93 services) including UPI, Bill Pay, and Card Controls.
            </p>
            <p style="color:#0f172a;"><b>Strategic Action Playbook:</b></p>
            <ul style="color:#475569; font-size:0.95rem;">
                <li><b>Cross-Sell Ecosystem:</b> One-click digital demat accounts, pre-approved vehicle/home loans.</li>
                <li><b>Loyalty Programs:</b> Multi-tiered rewards program for bill payments, UPI transactions, and recurring deposits.</li>
                <li><b>Retention:</b> Early alert systems for churn prevention if monthly transaction frequency dips.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="persona-card" style="border-left-color: #06d6a0;">
            <h4>📱 Mobile-First Micro-Transactors</h4>
            <span class="badge" style="background:#d1fae5; color:#065f46;">Highest App Logins</span>
            <span class="badge" style="background:#ccfbf1; color:#115e59;">UPI / Micro-Spend</span>
            <span class="badge" style="background:#e0f2fe; color:#075985;">Modest Balance</span>
            <p style="margin-top:10px; color:#334155;">
                <b>Profile:</b> Highest mobile app sessions (21.4 logins/month) and 99.9% off-hour usage. Modest average balance (₹31,700) and low ticket transactions (₹290 avg).
            </p>
            <p style="color:#0f172a;"><b>Strategic Action Playbook:</b></p>
            <ul style="color:#475569; font-size:0.95rem;">
                <li><b>Micro-Credit & BNPL:</b> Instant merchant scan-and-pay EMI, UPI credit line, micro overdrafts.</li>
                <li><b>Gamified Savings:</b> Auto-roundup spare change into digital gold or micro-SIP investments (starting ₹100).</li>
                <li><b>Merchant Partnerships:</b> Food delivery, OTT streaming, and ride-hailing discounts via debit card.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# -------------------------------------------------------------
# TAB 5: REAL-TIME CUSTOMER PROFILER
# -------------------------------------------------------------
with tab5:
    st.subheader("🎯 Real-Time Customer Segment Predictor & Recommender")
    st.markdown(
        "Enter or adjust customer digital and financial indicators to test the trained model "
        "and obtain instant segment classification, PCA positioning, and next-best-action guidance."
    )

    with st.form("customer_sim_form"):
        sim_col1, sim_col2, sim_col3 = st.columns(3)

        with sim_col1:
            st.markdown("###### 💰 Financial Capacity")
            inp_balance = st.number_input("Account Balance (₹)", min_value=100.0, max_value=2000000.0, value=75000.0, step=5000.0)
            inp_spend = st.number_input("Monthly Spend (₹)", min_value=50.0, max_value=500000.0, value=2500.0, step=500.0)
            inp_avg_spend = st.number_input("Average Transaction Amount (₹)", min_value=10.0, max_value=100000.0, value=1200.0, step=200.0)

        with sim_col2:
            st.markdown("###### 🕒 Timing & Channel Activity")
            inp_freq = st.slider("Transaction Frequency", min_value=1, max_value=10, value=2)
            inp_velocity = st.slider("Monthly Velocity", min_value=0.5, max_value=10.0, value=1.8, step=0.1)
            inp_offpeak = st.slider("Off-Peak / 24x7 Usage Ratio", min_value=0.0, max_value=1.0, value=0.75, step=0.05)
            inp_weekend = st.slider("Weekend Activity Ratio", min_value=0.0, max_value=1.0, value=0.30, step=0.05)

        with sim_col3:
            st.markdown("###### 📱 Digital Engagement")
            inp_mobile = st.slider("Mobile App Logins / Month", min_value=0, max_value=60, value=18)
            inp_web = st.slider("Web Portal Sessions / Month", min_value=0, max_value=30, value=6)
            inp_products = st.slider("Digital Products Used", min_value=1, max_value=6, value=3)
            inp_recency = st.slider("Days Since Last Transaction", min_value=1, max_value=90, value=12)

        submit_btn = st.form_submit_button("🚀 Classify Customer & Generate Strategy", use_container_width=True)

    if submit_btn:
        # Calculate composite digital score
        norm_mobile = min(inp_mobile / 40.0, 1.0)
        norm_web = min(inp_web / 20.0, 1.0)
        norm_offpeak = inp_offpeak
        norm_products = inp_products / 6.0
        norm_freq = min(inp_freq / 5.0, 1.0)
        sim_dig_score = round((0.30 * norm_mobile + 0.20 * norm_web + 0.20 * norm_offpeak + 0.15 * norm_products + 0.15 * norm_freq) * 100, 1)

        # Feature vector matching training order
        feature_cols_all = [
            'TxFrequency', 'TotalSpend', 'AvgSpend', 'AvgBalance',
            'OffPeakRatio', 'WeekendRatio', 'TxVelocity', 'RecencyDays',
            'MobileAppLogins', 'WebPortalSessions', 'DigitalProductsCount',
            'DigitalEngagementScore'
        ]

        raw_vector = pd.DataFrame([{
            'TxFrequency': inp_freq,
            'TotalSpend': np.log1p(inp_spend),
            'AvgSpend': np.log1p(inp_avg_spend),
            'AvgBalance': np.log1p(inp_balance),
            'OffPeakRatio': inp_offpeak,
            'WeekendRatio': inp_weekend,
            'TxVelocity': inp_velocity,
            'RecencyDays': np.log1p(inp_recency),
            'MobileAppLogins': inp_mobile,
            'WebPortalSessions': inp_web,
            'DigitalProductsCount': inp_products,
            'DigitalEngagementScore': sim_dig_score
        }])[feature_cols_all]

        scaled_vector = scaler.transform(raw_vector)
        pred_cluster = int(kmeans.predict(scaled_vector)[0])
        pca_coords = pca.transform(scaled_vector)[0]
        persona_info = personas.get(pred_cluster, {
            'persona': 'Segment ' + str(pred_cluster),
            'tagline': 'Custom Segment',
            'strategy': 'Personalized outreach.',
            'color': '#2563eb'
        })

        st.success(f"**Classification Complete!** Customer Assigned to **Cluster {pred_cluster}: {persona_info['persona']}**")

        res_c1, res_c2 = st.columns([1, 1])

        with res_c1:
            st.markdown(f"""
            <div class="persona-card" style="border-left-color: {persona_info['color']};">
                <h3>{persona_info['persona']}</h3>
                <p><i>{persona_info['tagline']}</i></p>
                <hr>
                <p><b>Predicted Cluster ID:</b> Cluster {pred_cluster}</p>
                <p><b>Computed Digital Engagement Score:</b> <b>{sim_dig_score} / 100</b></p>
                <p><b>PCA Coordinate:</b> (PC1: {pca_coords[0]:.2f}, PC2: {pca_coords[1]:.2f}, PC3: {pca_coords[2]:.2f})</p>
                <hr>
                <h5 style="color:#0f172a;">🎯 Next-Best-Action (NBA) Strategy:</h5>
                <p style="color:#334155;">{persona_info['strategy']}</p>
            </div>
            """, unsafe_allow_html=True)

        with res_c2:
            st.markdown("###### Position in PCA Space Relative to Customer Base")
            sim_sample = df.sample(min(1500, len(df)), random_state=42)
            fig_sim = px.scatter(
                sim_sample,
                x='PCA1',
                y='PCA2',
                color='Segment_Name',
                opacity=0.4,
                color_discrete_sequence=['#0077b6', '#fb5607', '#8338ec', '#06d6a0']
            )
            # Add user point
            fig_sim.add_trace(go.Scatter(
                x=[pca_coords[0]],
                y=[pca_coords[1]],
                mode='markers+text',
                marker=dict(symbol='star', size=18, color='#ff0054', line=dict(width=2, color='white')),
                text=["⭐ Simulated Customer"],
                textposition="top center",
                name="Simulated Customer"
            ))
            fig_sim.update_layout(
                xaxis_title="PC1: Digital Engagement & Self-Service",
                yaxis_title="PC2: Financial Affluence & Spend",
                height=380,
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", y=-0.25)
            )
            st.plotly_chart(fig_sim, use_container_width=True)

st.markdown("---")
st.markdown("<p style='text-align:center; color:#94a3b8; font-size:0.85rem;'>Digital Banking Behaviour Analysis • Team 9 • Antigravity AI Data Analytics Platform</p>", unsafe_allow_html=True)
