"""
Feature Extraction Module for Network Intrusion Detection
Implements comprehensive feature engineering pipeline
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def extract_features(df, n_top_features=30, pca_variance=0.95):
    """
    Complete feature extraction pipeline
    
    Args:
        df: Input dataframe with 'label' column
        n_top_features: Number of top features to select
        pca_variance: Variance to retain in PCA
    
    Returns:
        DataFrame with extracted features and label
    """
    
    df = df.copy()
    TARGET_COL = 'label'
    
    # Step 1: Identify column groups
    print("Identifying column groups...")
    
    # Continuous features (common in network datasets)
    continuous_candidates = [
        'duration', 'src_bytes', 'dst_bytes', 'hot', 'num_failed_logins',
        'num_compromised', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'count', 'srv_count', 'dst_host_count', 
        'dst_host_srv_count'
    ]
    CONTINUOUS = [c for c in continuous_candidates if c in df.columns]
    
    # Rate features
    rate_candidates = [
        'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
    ]
    RATE_FEATURES = [c for c in rate_candidates if c in df.columns]
    
    # Binary flags
    binary_candidates = [
        'land', 'wrong_fragment', 'urgent', 'logged_in', 'root_shell',
        'su_attempted', 'is_host_login', 'is_guest_login'
    ]
    BINARY_FLAGS = [c for c in binary_candidates if c in df.columns]
    
    # One-hot encoded columns
    PROTOCOL_COLS = [c for c in df.columns if c.startswith('protocol_type_')]
    SERVICE_COLS = [c for c in df.columns if c.startswith('service_')]
    FLAG_COLS = [c for c in df.columns if c.startswith('flag_')]
    
    # Step 2: Create engineered features
    print("Creating engineered features...")
    
    # Log transforms for skewed continuous features
    for col in CONTINUOUS:
        if col in df.columns:
            df[f'{col}_log1p'] = np.log1p(df[col])
    
    # Byte-based features
    if 'src_bytes' in df.columns and 'dst_bytes' in df.columns:
        df['total_bytes'] = df['src_bytes'] + df['dst_bytes']
        df['byte_ratio'] = np.where(
            df['dst_bytes'] > 0,
            df['src_bytes'] / (df['dst_bytes'] + 1),
            0
        )
        df['byte_diff'] = df['src_bytes'] - df['dst_bytes']
        
        if 'duration' in df.columns:
            df['src_bytes_per_sec'] = np.where(
                df['duration'] > 0,
                df['src_bytes'] / (df['duration'] + 1),
                0
            )
            df['dst_bytes_per_sec'] = np.where(
                df['duration'] > 0,
                df['dst_bytes'] / (df['duration'] + 1),
                0
            )
    
    # Privilege score
    privilege_cols = ['root_shell', 'su_attempted', 'num_root', 'num_compromised']
    available_privilege = [c for c in privilege_cols if c in df.columns]
    if available_privilege:
        df['privilege_score'] = df[available_privilege].sum(axis=1)
    
    # Suspicious score
    suspicious_cols = ['wrong_fragment', 'urgent', 'hot', 'num_failed_logins']
    available_suspicious = [c for c in suspicious_cols if c in df.columns]
    if available_suspicious:
        df['suspicious_score'] = df[available_suspicious].sum(axis=1)
    
    # Connection ratios
    if 'count' in df.columns and 'srv_count' in df.columns:
        df['srv_count_ratio'] = np.where(
            df['count'] > 0,
            df['srv_count'] / (df['count'] + 1),
            0
        )
    
    # Error rate composites
    error_rate_cols = [c for c in df.columns if 'error_rate' in c]
    if error_rate_cols:
        df['total_error_rate'] = df[error_rate_cols].sum(axis=1)
        df['mean_error_rate'] = df[error_rate_cols].mean(axis=1)
    
    # Step 3: Feature scaling
    print("Scaling features...")
    
    # Get all numeric columns except target
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != TARGET_COL]
    
    # Fill any NaN values
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Apply StandardScaler
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    # Step 4: Feature selection
    print("Selecting best features...")
    
    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df_scaled[TARGET_COL])
    
    # Prepare features
    X = df_scaled[numeric_cols].fillna(0)
    
    # Remove near-zero variance features
    vt = VarianceThreshold(threshold=0.01)
    X_vt = vt.fit_transform(X)
    kept_features = [numeric_cols[i] for i in range(len(numeric_cols)) 
                     if vt.get_support()[i]]
    X_vt_df = pd.DataFrame(X_vt, columns=kept_features)
    
    # Mutual Information scoring
    mi_scores = mutual_info_classif(X_vt_df, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=kept_features).sort_values(ascending=False)
    
    # Random Forest importance (on sample for speed)
    sample_size = min(10000, len(X_vt_df))
    idx = np.random.choice(len(X_vt_df), sample_size, replace=False)
    
    rf = RandomForestClassifier(
        n_estimators=50, 
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_vt_df.iloc[idx], y[idx])
    rf_importance = pd.Series(rf.feature_importances_, index=kept_features).sort_values(ascending=False)
    
    # Combine rankings
    mi_norm = (mi_series - mi_series.min()) / (mi_series.max() - mi_series.min() + 1e-9)
    rf_norm = (rf_importance - rf_importance.min()) / (rf_importance.max() - rf_importance.min() + 1e-9)
    combined = (mi_norm + rf_norm).sort_values(ascending=False)
    
    # Select top K features
    top_features = combined.head(n_top_features).index.tolist()
    
    # Step 5: Create final feature matrix
    print(f"Selected {len(top_features)} features")
    print(top_features)
    
    X_final = df_scaled[top_features].copy()
    X_final[TARGET_COL] = df_scaled[TARGET_COL].values
    
    return X_final, top_features


def get_feature_importance(df, top_n=20):
    """
    Get feature importance rankings
    
    Args:
        df: Dataframe with features and 'label'
        top_n: Number of top features to return
    
    Returns:
        DataFrame with feature importance scores
    """
    
    TARGET_COL = 'label'
    
    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET_COL])
    
    # Get features
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols].fillna(0)
    
    # Calculate MI scores
    mi_scores = mutual_info_classif(X, y, random_state=42)
    
    # Train RF for importance
    sample_size = min(10000, len(X))
    idx = np.random.choice(len(X), sample_size, replace=False)
    
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X.iloc[idx], y[idx])
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'MI_Score': mi_scores,
        'RF_Importance': rf.feature_importances_
    })
    
    # Normalize and combine
    importance_df['MI_Norm'] = (
        (importance_df['MI_Score'] - importance_df['MI_Score'].min()) / 
        (importance_df['MI_Score'].max() - importance_df['MI_Score'].min() + 1e-9)
    )
    importance_df['RF_Norm'] = (
        (importance_df['RF_Importance'] - importance_df['RF_Importance'].min()) / 
        (importance_df['RF_Importance'].max() - importance_df['RF_Importance'].min() + 1e-9)
    )
    importance_df['Combined_Score'] = importance_df['MI_Norm'] + importance_df['RF_Norm']
    
    # Sort and return top N
    importance_df = importance_df.sort_values('Combined_Score', ascending=False).head(top_n)
    
    return importance_df[['Feature', 'MI_Score', 'RF_Importance', 'Combined_Score']]
