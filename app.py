"""
Machine Learning Web Application
A user-friendly web interface to upload data and train ML models.
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Page configuration
st.set_page_config(
    page_title="ML Training Platform",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🤖 Machine Learning Training Platform")
st.markdown("""
Upload your dataset and train multiple ML models with just a few clicks!
This app supports CSV files and compares different classification algorithms.
""")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# Initialize session state
if 'trained' not in st.session_state:
    st.session_state.trained = False
    st.session_state.results = {}

def generate_sample_data():
    """Generate sample data for demo purposes"""
    n_samples = 1000
    X = np.random.randn(n_samples, 5)
    y = (X[:, 0] + X[:, 1] + X[:, 2] > 0).astype(int)
    noise_idx = np.random.choice(n_samples, size=int(n_samples * 0.1))
    y[noise_idx] = 1 - y[noise_idx]
    
    df = pd.DataFrame(X, columns=[f'Feature_{i+1}' for i in range(5)])
    df['Target'] = y
    return df

def train_models(X_train, X_test, y_train, y_test, selected_models):
    """Train selected models and return results"""
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (name, model) in enumerate(models.items()):
        if name in selected_models:
            status_text.text(f"Training {name}...")
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'confusion_matrix': cm,
                'model': model
            }
            
            progress_bar.progress((idx + 1) / len(selected_models))
    
    status_text.text("Training complete! ✅")
    progress_bar.empty()
    
    return results

def plot_accuracy_comparison(results):
    """Create accuracy comparison bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in models]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    bars = ax.bar(models, accuracies, color=colors[:len(models)])
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(cm, model_name):
    """Create confusion matrix heatmap"""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=10)
    ax.set_xlabel('Predicted Label', fontsize=10)
    
    plt.tight_layout()
    return fig

# Main app
tab1, tab2, tab3 = st.tabs(["📊 Data Upload", "🎯 Model Training", "📈 Results"])

with tab1:
    st.header("Upload Your Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Option 1: Upload CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
            st.session_state.df = df
        
    with col2:
        st.subheader("Option 2: Use Sample Data")
        if st.button("Generate Sample Data"):
            df = generate_sample_data()
            st.session_state.df = df
            st.success("✅ Sample data generated!")
    
    # Display data preview
    if 'df' in st.session_state:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.df.head(10))
        
        st.subheader("Data Statistics")
        st.write(st.session_state.df.describe())
        
        # Column selection
        st.subheader("Select Target Column")
        target_col = st.selectbox("Target column (what you want to predict):", 
                                  st.session_state.df.columns)
        st.session_state.target_col = target_col

with tab2:
    st.header("Train Machine Learning Models")
    
    if 'df' not in st.session_state:
        st.warning("⚠️ Please upload data or generate sample data first!")
    else:
        # Model selection
        st.subheader("Select Models to Train")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            lr = st.checkbox("Logistic Regression", value=True)
            dt = st.checkbox("Decision Tree", value=True)
        
        with col2:
            rf = st.checkbox("Random Forest", value=True)
            svm = st.checkbox("SVM", value=True)
        
        with col3:
            knn = st.checkbox("K-Nearest Neighbors", value=False)
        
        selected_models = []
        if lr: selected_models.append('Logistic Regression')
        if dt: selected_models.append('Decision Tree')
        if rf: selected_models.append('Random Forest')
        if svm: selected_models.append('SVM')
        if knn: selected_models.append('K-Nearest Neighbors')
        
        # Training parameters
        st.subheader("Training Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
        
        with col2:
            random_state = st.number_input("Random seed", 0, 100, 42)
        
        # Train button
        if st.button("🚀 Train Models", type="primary"):
            if len(selected_models) == 0:
                st.error("Please select at least one model!")
            else:
                with st.spinner("Training models..."):
                    # Prepare data
                    df = st.session_state.df
                    target_col = st.session_state.target_col
                    
                    X = df.drop(columns=[target_col])
                    y = df[target_col]
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    )
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                    
                    # Train models
                    results = train_models(X_train, X_test, y_train, y_test, selected_models)
                    
                    st.session_state.results = results
                    st.session_state.y_test = y_test
                    st.session_state.trained = True
                    
                st.success("✅ Training complete! Check the Results tab.")

with tab3:
    st.header("Training Results")
    
    if not st.session_state.trained:
        st.info("ℹ️ Train some models first to see results here!")
    else:
        results = st.session_state.results
        
        # Accuracy comparison
        st.subheader("📊 Model Accuracy Comparison")
        fig = plot_accuracy_comparison(results)
        st.pyplot(fig)
        
        # Best model
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        st.success(f"🏆 Best Model: **{best_model[0]}** with {best_model[1]['accuracy']:.2%} accuracy")
        
        # Detailed results for each model
        st.subheader("📋 Detailed Results")
        
        for model_name, result in results.items():
            with st.expander(f"{model_name} - Accuracy: {result['accuracy']:.2%}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Accuracy", f"{result['accuracy']:.2%}")
                    
                    # Classification report
                    st.text("Classification Report:")
                    report = classification_report(
                        st.session_state.y_test, 
                        result['predictions'],
                        output_dict=True
                    )
                    st.json(report)
                
                with col2:
                    # Confusion matrix
                    fig = plot_confusion_matrix(result['confusion_matrix'], model_name)
                    st.pyplot(fig)
        
        # Download results
        st.subheader("💾 Download Results")
        
        results_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [results[m]['accuracy'] for m in results.keys()]
        }).sort_values('Accuracy', ascending=False)
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="ml_results.csv",
            mime="text/csv"
        )

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**How to use:**
1. Upload your CSV or generate sample data
2. Select target column
3. Choose models to train
4. Click 'Train Models'
5. View results and compare!
""")