import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Styling
st.set_page_config(
    page_title="Network Intrusion Detection",
    page_icon="",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_uploaded' not in st.session_state:
    st.session_state.data_uploaded = False
if 'data_cleaned' not in st.session_state:
    st.session_state.data_cleaned = False
if 'features_extracted' not in st.session_state:
    st.session_state.features_extracted = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# Sidebar navigation
st.sidebar.title("Pipeline Steps")
step = st.sidebar.radio(
    "Select Step:",
    ["Home", "Upload Data", "Data Cleaning", "Feature Extraction", 
     "Model Training", "Evaluation & Results"]
)

# ============================================================================
# HOME PAGE
# ============================================================================
if step == "Home":
    st.markdown('<div class="step-header"><h2>Network Intrusion Detection System</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### What This System Does")
        st.markdown("""
        This application provides a complete machine learning pipeline for network intrusion detection:
        
        1. **Data Upload**: Upload your network traffic CSV file
        2. **Data Cleaning**: Automatic data validation and preprocessing
        3. **Feature Extraction**: Advanced feature engineering and selection
        4. **Model Training**: Train multiple ML models using distributed processing with Apache Spark 
        5. **Evaluation**: Compare models and visualize results 
        """)
    
    with col2:
        st.markdown("### Supported Datasets")
        st.markdown("""
        This system works with **NSL-KDD** network intrusion datasets.
        
        **Required columns:**
        - Network features (duration, protocol_type, service, etc.)
        - Binary flags (logged_in, root_shell, etc.)
        - Rate features (serror_rate, etc.)
        - Target: `label` (normal vs attack types)
        
        **Supported models:**
        - Random Forest Classifier
        - Logistic Regression
        - Isolation Forest (anomaly detection)
        """)
    
    st.markdown("---")
    
    # Display pipeline status
    st.markdown("### Current Pipeline Status")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_upload = "Complete" if st.session_state.data_uploaded else "Pending"
        st.metric("Data Upload", status_upload)
    with col2:
        status_clean = "Complete" if st.session_state.data_cleaned else "Pending"
        st.metric("Data Cleaning", status_clean)
    with col3:
        status_features = "Complete" if st.session_state.features_extracted else "Pending"
        st.metric("Feature Extract", status_features)
    with col4:
        status_models = "Complete" if st.session_state.models_trained else "Pending"
        st.metric("Model Training", status_models)

# ============================================================================
# STEP 1: UPLOAD DATA
# ============================================================================
elif step == "Upload Data":
    st.markdown('<h1 class="main-header"> Network Intrusion Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<div class="step-header"><h2>Step 1: Upload Your Dataset</h2></div>', unsafe_allow_html=True)

    st.markdown("""
    Upload your network traffic dataset in CSV format. The file should contain network connection features.
    """)

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload NSL-KDD dataset"
    )

    if uploaded_file is not None:
        try:
            with st.spinner("Loading data..."):
                df = pd.read_csv(uploaded_file, header=None)

                # NSL-KDD column names
                columns = [
                    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
                    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
                    "num_compromised","root_shell","su_attempted","num_root",
                    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
                    "is_host_login","is_guest_login","count","srv_count","serror_rate",
                    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
                    "diff_srv_rate","srv_diff_host_rate","dst_host_count",
                    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
                    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
                    "dst_host_serror_rate","dst_host_srv_serror_rate",
                    "dst_host_rerror_rate","dst_host_srv_rerror_rate",
                    "label","difficulty"
                ]

                df.columns = columns
                # Keep original labels for visualization
                df['attack_type'] = df['label']

                # Create binary label for ML
                df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

                # Drop difficulty column
                df = df.drop(columns=['difficulty'])

                st.session_state.raw_data = df
                st.session_state.data_uploaded = True

            st.markdown('<div class="success-box"><b>File uploaded successfully!</b></div>', unsafe_allow_html=True)

            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", f"{len(df.columns)}")
            with col3:
                st.metric("File Size", f"{uploaded_file.size / 1024:.2f} KB")

            # Preview
            st.markdown("### Data Preview")
            st.dataframe(df.head(10), width='stretch')

            # Column info
            st.markdown("### Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': [str(dtype) for dtype in df.dtypes.values],
                'Missing': [int(x) for x in df.isnull().sum().values],
                'Unique': [int(x) for x in df.nunique().values]
            })
            st.dataframe(col_info, width='stretch')

            # ================= VISUALIZATIONS =================
            if 'label' in df.columns:
                st.markdown("### Target Distribution")

                fig, ax = plt.subplots(1, 2, figsize=(12, 4))

                # 🔹 BAR CHART (ALL ATTACK TYPES)
                df['attack_type'].value_counts().plot(kind='bar', ax=ax[0], color='steelblue')
                ax[0].set_title('All Attack Types Distribution')
                ax[0].set_xlabel('Attack Type')
                ax[0].set_ylabel('Count')

                # 🔹 PIE CHART (NORMAL vs ATTACK)
                binary_counts = df['attack_type'].apply(
                    lambda x: 'Normal' if x == 'normal' else 'Attack'
                ).value_counts()

                binary_counts.plot(kind='pie', ax=ax[1], autopct='%1.1f%%')
                ax[1].set_title('Normal vs Attack Distribution')
                ax[1].set_ylabel('')

                # SHOW BAR + PIE
                st.pyplot(fig)

                # HEATMAP
                st.markdown("### Feature Correlation Heatmap")

                numeric_df = df.select_dtypes(include=[np.number])

                # limit columns for readability
                if numeric_df.shape[1] > 15:
                    numeric_df = numeric_df.iloc[:, :15]

                fig2, ax2 = plt.subplots(figsize=(10, 6))

                sns.heatmap(
                    numeric_df.corr(),
                    cmap='coolwarm',
                    annot=False,
                    linewidths=0.5,
                    ax=ax2
                )

                ax2.set_title("Feature Correlation Heatmap")

                st.pyplot(fig2)

            else:
                st.warning("No 'label' column found.")

            st.success("Ready for next step! Go to Data Cleaning.")

        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

    else:
        st.info("Please upload a CSV file to begin")

# ============================================================================
# STEP 2: DATA CLEANING
# ============================================================================
elif step == "Data Cleaning":
    st.markdown('<div class="step-header"><h2>Step 2: Data Cleaning & Validation</h2></div>', unsafe_allow_html=True)
    
    if not st.session_state.data_uploaded:
        st.warning("⚠️ Please upload data first!")
        st.stop()
    
    df = st.session_state.raw_data
    
    st.markdown("### Data Quality Check")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        st.warning(f"Found {missing.sum()} missing values")
        st.dataframe(missing[missing > 0])
    else:
        st.success("No missing values found")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        st.warning(f"Found {duplicates} duplicate rows")
    else:
        st.success("No duplicate rows found")
    
    # Data types
    st.markdown("### Data Type Summary")
    type_counts = df.dtypes.value_counts()
    st.write(type_counts)
    
    # Cleaning options
    st.markdown("### Cleaning Options")
    
    col1, col2 = st.columns(2)
    with col1:
        handle_missing = st.selectbox(
            "Handle Missing Values:",
            ["Drop rows with missing values", "Fill with mean", "Fill with median", "Fill with 0"]
        )
    with col2:
        handle_duplicates = st.checkbox("Remove duplicate rows", value=True)
    
    if st.button("Clean Data", type="primary"):
        with st.spinner("Cleaning data..."):
            df_clean = df.copy()
            
            # Handle missing values
            if handle_missing == "Drop rows with missing values":
                df_clean = df_clean.dropna()
            elif handle_missing == "Fill with mean":
                numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
                df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
            elif handle_missing == "Fill with median":
                numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
                df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
            else:
                df_clean = df_clean.fillna(0)
            
            # Handle duplicates
            if handle_duplicates:
                df_clean = df_clean.drop_duplicates()
            
            # Save cleaned data
            st.session_state.cleaned_data = df_clean
            st.session_state.data_cleaned = True
            
            # Show results
            st.markdown('<div class="success-box"><b>Data cleaned successfully!</b></div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Rows", f"{len(df):,}")
            with col2:
                st.metric("Cleaned Rows", f"{len(df_clean):,}")
            with col3:
                removed = len(df) - len(df_clean)
                st.metric("Rows Removed", f"{removed:,}")
            
            st.markdown("### Cleaned Data Preview")
            st.dataframe(df_clean.head(10), width='stretch')
            
            st.success("Ready for next step! Go to **Feature Extraction** in the sidebar.")

# ============================================================================
# STEP 3: FEATURE EXTRACTION
# ============================================================================
elif step == "Feature Extraction":
    st.markdown('<div class="step-header"><h2>Step 3: Feature Engineering & Selection</h2></div>', unsafe_allow_html=True)
    
    if not st.session_state.data_cleaned:
        st.warning("⚠️ Please clean data first!")
        st.stop()
    
    df = st.session_state.cleaned_data
    
    st.markdown("""
    This step performs advanced feature engineering including:
    - Byte-based features (ratios, differences)
    - Feature selection using Mutual Information + Random Forest importance
    """)
    
    # Feature extraction settings
    col1, col2 = st.columns(2)
    with col1:
        num_top_features = st.slider(
            "Number of top features to select:",
            min_value=10, max_value=50, value=30, step=5
        )
    with col2:
        pca_variance = st.slider(
            "PCA variance to retain:",
            min_value=0.80, max_value=0.99, value=0.95, step=0.01
        )
    
    if st.button("Extract Features", type="primary"):
        with st.spinner("Extracting features... This may take a few minutes..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Import feature extraction module
                from feature_extractor import extract_features
                
                status_text.text("Step 1/5: Identifying column groups...")
                progress_bar.progress(20)
                
                status_text.text("Step 2/5: Creating engineered features...")
                progress_bar.progress(40)
                
                # Extract features
                df_features, selected_features = extract_features(
                    df, 
                    n_top_features=num_top_features,
                    pca_variance=pca_variance
                )
                st.markdown("###  Selected Features (Top Important Features)")
                st.write(selected_features)
                status_text.text("Step 3/5: Scaling features...")
                progress_bar.progress(60)
                
                status_text.text("Step 4/5: Selecting best features...")
                progress_bar.progress(80)
                
                status_text.text("Step 5/5: Finalizing...")
                progress_bar.progress(100)
                
                # Save results
                st.session_state.feature_data = df_features
                st.session_state.features_extracted = True
                
                # Clear progress
                progress_bar.empty()
                status_text.empty()
                
                # Show success
                st.markdown('<div class="success-box"><b>Features extracted successfully!</b></div>', unsafe_allow_html=True)
                
                # Show results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Features", f"{len(df.columns)-1}")
                with col2:
                    st.metric("Engineered Features", f"{len(df_features.columns)-1}")
                with col3:
                    improvement = ((len(df_features.columns)-1) / (len(df.columns)-1) - 1) * 100
                    st.metric("Feature Growth", f"+{improvement:.1f}%")
                
                st.markdown("### Feature Data Preview")
                st.dataframe(df_features.head(10), width='stretch')
                
                st.success("Ready for next step! Go to **Model Training** in the sidebar.")
                
            except Exception as e:
                st.error(f"Error during feature extraction: {str(e)}")
                st.info("Using simplified feature extraction...")
                
                # Simplified fallback
                feature_cols = [col for col in df.columns if col != 'label']
                df_features = df[feature_cols + ['label']].copy()
                
                st.session_state.feature_data = df_features
                st.session_state.features_extracted = True
                
                st.warning(" Using basic features. For advanced extraction, ensure all dependencies are installed.")
                st.success("Ready for next step! Go to **Model Training** in the sidebar.")

# ============================================================================
# STEP 4: MODEL TRAINING
# ============================================================================
elif step == "Model Training":
    st.markdown('<div class="step-header"><h2>Step 4: Train ML Models with Spark</h2></div>', unsafe_allow_html=True)
    
    if not st.session_state.features_extracted:
        st.warning("Please extract features first!")
        st.stop()
    
    df = st.session_state.feature_data
    
    st.markdown("""
    Train machine learning models using Spark MLlib on NSL-KDD dataset:
    - **Random Forest**: Ensemble classifier - best accuracy (~95%), 2-3 minutes
    - **Logistic Regression**: Linear classifier - fast (~60 sec), good baseline (~90%)
    - **Isolation Forest**: Anomaly detection - unsupervised learning (~30 sec)
    
     Binary Classification: Normal (0) vs Attack (1)
    """)
    
    # Model selection
    st.markdown("### Select Models to Train")
    col1, col2, col3 = st.columns(3)
    with col1:
        train_rf = st.checkbox("Random Forest ", value=True, help="Best accuracy ~95%")
    with col2:
        train_lr = st.checkbox("Logistic Regression ", value=True, help="Fast ~60 sec")
    with col3:
        train_if = st.checkbox("Isolation Forest ", value=True, help="Anomaly detection")
    
    # Training settings
    st.markdown("### Training Settings")
    st.info("**Data Split**: 70% Training, 15% Validation, 15% Test (Optimized for NSL-KDD)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training", "70%", help="Used to train model parameters")
    with col2:
        st.metric("Validation", "15%", help="Used to evaluate during training")
    with col3:
        st.metric("Test", "15%", help="Final evaluation on unseen data")
    
    if st.button("Train Models", type="primary"):
        with st.spinner("Training models... This may take several minutes..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Import final training module
                from model_trainer import train_models
                
                # Train models
                results = train_models(
                    df,
                    train_rf=train_rf,
                    train_lr=train_lr,
                    train_if=train_if,
                    progress_callback=lambda prog, msg: (
                        progress_bar.progress(prog),
                        status_text.text(msg)
                    )
                )
                
                # Save results
                st.session_state.model_results = results
                st.session_state.models_trained = True
                
                # Clear progress
                progress_bar.empty()
                status_text.empty()
                
                # Show success
                st.markdown('<div class="success-box"><b>Models trained successfully!</b></div>', unsafe_allow_html=True)
                
                # Show training summary
                st.markdown("### Training Summary")
                summary_data = []
                for r in results:
                    summary_data.append({
                        'Model': r['model_name'],
                        'Training Time (s)': f"{r['training_time']:.1f}",
                        'Val Accuracy': f"{r.get('val_accuracy', 0):.4f}",
                        'Test Accuracy': f"{r['test_accuracy']:.4f}",
                        'Test AUC-ROC': f"{r.get('roc_auc', 0):.4f}",
                        'Test F1': f"{r.get('test_f1', 0):.4f}"
                    })
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, width='stretch')
                
                st.success("Ready for next step! Go to **Evaluation & Results** in the sidebar.")
                
            except Exception as e:
                st.error(f"Error during model training: {str(e)}")
                st.info("Please ensure Spark is properly configured.")

# ============================================================================
# STEP 5: EVALUATION & RESULTS
# ============================================================================
elif step == "Evaluation & Results":
    st.markdown('<div class="step-header"><h2>Step 5: Model Evaluation & Results</h2></div>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.warning("Please train models first!")
        st.stop()
    
    results = st.session_state.model_results
    
    st.markdown("### Model Comparison")
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Model': [r['model_name'] for r in results],
        'Val Accuracy': [r.get('val_accuracy', 0) for r in results],
        'Test Accuracy': [r['test_accuracy'] for r in results],
        'AUC-ROC': [r.get('roc_auc', 0) for r in results],
        'F1 Score': [r.get('test_f1', 0) for r in results],
        'Training Time': [r['training_time'] for r in results]
    })
    
    st.dataframe(comparison_df.style.highlight_max(axis=0, subset=['Test Accuracy', 'AUC-ROC', 'F1 Score']), 
                 width='stretch')
    
    # Visualizations
    st.markdown("### Performance Visualizations")
    
    # Select model for detailed view
    model_names = [r['model_name'] for r in results]
    selected_model = st.selectbox("Select model for detailed analysis:", model_names)
    
    # Get selected model results
    model_result = next(r for r in results if r['model_name'] == selected_model)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # 1. Model Comparison - Test Accuracy
    ax = axes[0, 0]
    models = comparison_df['Model']
    test_acc = comparison_df['Test Accuracy']
    bars = ax.bar(range(len(models)), test_acc, color=['#2E86AB', '#A23B72', '#06A77D'])
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title('Test Accuracy Comparison', fontweight='bold')
    ax.set_ylim([0, 1.0])
    for i, v in enumerate(test_acc):
        ax.text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    
    # 2. AUC-ROC Comparison
    ax = axes[0, 1]
    auc_scores = comparison_df['AUC-ROC']
    bars = ax.bar(range(len(models)), auc_scores, color=['#2E86AB', '#A23B72', '#06A77D'])
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('AUC-ROC')
    ax.set_title('ROC-AUC Score Comparison', fontweight='bold')
    ax.set_ylim([0, 1.0])
    for i, v in enumerate(auc_scores):
        if v > 0:  # Skip IF which has 0
            ax.text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    
    # 3. F1 Score Comparison
    ax = axes[0, 2]
    f1_scores = comparison_df['F1 Score']
    bars = ax.bar(range(len(models)), f1_scores, color=['#2E86AB', '#A23B72', '#06A77D'])
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score Comparison', fontweight='bold')
    ax.set_ylim([0, 1.0])
    for i, v in enumerate(f1_scores):
        if v > 0:
            ax.text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    
    # 4. Training Time Comparison
    ax = axes[0, 3]
    train_times = comparison_df['Training Time']
    bars = ax.bar(range(len(models)), train_times, color=['#2E86AB', '#A23B72', '#06A77D'])
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Training Time Comparison', fontweight='bold')
    for i, v in enumerate(train_times):
        ax.text(i, v + max(train_times)*0.02, f'{v:.1f}s', ha='center', fontweight='bold')
    
    # 5. Confusion Matrix - Random Forest
    if 'confusion_matrix' in model_result and model_result['model_name'] in ['Random Forest', 'Logistic Regression']:
        ax = axes[1, 0]
        cm = model_result['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'])
        ax.set_title(f'{selected_model} - Confusion Matrix', fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    else:
        axes[1, 0].axis('off')
        axes[1, 0].text(0.5, 0.5, 'Confusion Matrix\nNot Available', 
                       ha='center', va='center', fontsize=12)
    
    # 6. Val vs Test Accuracy
    ax = axes[1, 1]
    val_accs = [r.get('val_accuracy', 0) for r in results if r['model_name'] != 'Isolation Forest']
    test_accs = [r['test_accuracy'] for r in results if r['model_name'] != 'Isolation Forest']
    model_names = [r['model_name'] for r in results if r['model_name'] != 'Isolation Forest']
    
    x = np.arange(len(model_names))
    width = 0.35
    bars1 = ax.bar(x - width/2, val_accs, width, label='Validation', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_accs, width, label='Test', color='#A23B72', alpha=0.8)
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation vs Test Accuracy', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 7. RF vs LR Performance Radar (using bar chart as proxy)
    ax = axes[1, 2]
    if len([r for r in results if r['model_name'] in ['Random Forest', 'Logistic Regression']]) >= 2:
        rf_result = next((r for r in results if r['model_name'] == 'Random Forest'), None)
        lr_result = next((r for r in results if r['model_name'] == 'Logistic Regression'), None)
        
        if rf_result and lr_result:
            metrics = ['Accuracy', 'AUC-ROC', 'F1 Score']
            rf_scores = [rf_result['test_accuracy'], rf_result.get('roc_auc', 0), rf_result.get('test_f1', 0)]
            lr_scores = [lr_result['test_accuracy'], lr_result.get('roc_auc', 0), lr_result.get('test_f1', 0)]
            
            x = np.arange(len(metrics))
            width = 0.35
            bars1 = ax.bar(x - width/2, rf_scores, width, label='Random Forest', color='#2E86AB')
            bars2 = ax.bar(x + width/2, lr_scores, width, label='Logistic Regression', color='#A23B72')
            ax.set_ylabel('Score')
            ax.set_title('RF vs LR - Metric Comparison', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.legend()
            ax.set_ylim([0, 1.1])
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    else:
        axes[1, 2].axis('off')
    
    # 8. Classification Report
    ax = axes[1, 3]
    ax.axis('off')
    if model_result['model_name'] in ['Random Forest', 'Logistic Regression']:
        test_pred_pd = model_result.get('test_predictions_pd')
        if test_pred_pd is not None:
            report = classification_report(test_pred_pd['label'], test_pred_pd['prediction'],
                                         target_names=['Normal', 'Attack'])
        else:
            # Use confusion matrix to create basic report
            cm = model_result['confusion_matrix']
            tn, fp, fn, tp = cm.ravel()
            report = f"         precision  recall  f1-score  support\n\n"
            report += f"Normal     {tn/(tn+fp):.2f}    {tn/(tn+fn):.2f}    {2*tn/(2*tn+fp+fn):.2f}     {tn+fn}\n"
            report += f"Attack     {tp/(tp+fn):.2f}    {tp/(tp+fp):.2f}    {2*tp/(2*tp+fn+fp):.2f}     {tp+fp}\n"
        ax.text(0.05, 0.95, f'{selected_model}\nClassification Report:', 
               fontsize=11, fontweight='bold', transform=ax.transAxes)
        ax.text(0.05, 0.05, report, fontsize=8, family='monospace', 
               transform=ax.transAxes, verticalalignment='bottom')
    else:
        ax.text(0.5, 0.5, 'Classification Report\nNot Available for IF', 
               ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Download results
    st.markdown("### Download Results")
    col1, col2 = st.columns(2)
    
    with col1:
        # Save comparison to CSV
        csv = comparison_df.to_csv(index=False)
        st.download_button(
            label="Download Comparison CSV",
            data=csv,
            file_name="model_comparison.csv",
            mime="text/csv"
        )
    
    with col2:
        # Save plot
        st.download_button(
            label="Download Visualizations",
            data="Plot saved separately",
            file_name="results_plot.png",
            mime="image/png"
        )
    
    # Interpretation
    st.markdown("### Results Interpretation")
    
    best_model = comparison_df.loc[comparison_df['Test Accuracy'].idxmax(), 'Model']
    best_accuracy = comparison_df['Test Accuracy'].max()
    
    st.markdown(f"""
    **Best Performing Model:** {best_model}  
    **Test Accuracy:** {best_accuracy:.4f} ({best_accuracy*100:.2f}%)
    
    **Key Findings:**
    - The model shows {"good" if best_accuracy > 0.9 else "moderate"} performance on unseen data
    - Validation and test accuracies are {"close" if abs(model_result['val_accuracy'] - model_result['test_accuracy']) < 0.05 else "different"}, 
      indicating {"minimal" if abs(model_result['val_accuracy'] - model_result['test_accuracy']) < 0.05 else "some"} overfitting
    - Training completed in {model_result['training_time']:.2f} seconds
    
    **Practical Implications:**
    - This model can be deployed for real-time network intrusion detection
    - Regular retraining recommended as attack patterns evolve
    - Consider ensemble methods for production deployment
    """)
    
    st.success("Analysis complete! You can retrain models or download results.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
    <p>Network Intrusion Detection System v1.0 | Built with Streamlit & Apache Spark</p>
    </div>
""", unsafe_allow_html=True)
