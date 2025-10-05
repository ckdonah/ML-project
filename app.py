"""
Machine Learning Web Application - Enhanced Edition
A comprehensive ML platform with predictions, feature importance, and model insights.
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
import time
import pickle

# Page configuration
st.set_page_config(
    page_title="ML Training Platform Pro",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🤖 ML Training Platform Pro")
st.markdown("""
**Professional Machine Learning Platform** with predictions, feature analysis, and model insights!
Upload data, train models, get predictions, and understand what makes your models tick.
""")

# Initialize session state
if 'trained' not in st.session_state:
    st.session_state.trained = False
    st.session_state.results = {}
    st.session_state.scaler = None
    st.session_state.feature_names = []

def check_data_quality(df):
    """Analyze data quality and return issues"""
    issues = []
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        issues.append({
            'type': 'Missing Values',
            'severity': 'Medium',
            'details': f"Found {missing.sum()} missing values across {(missing > 0).sum()} columns"
        })
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append({
            'type': 'Duplicate Rows',
            'severity': 'Low',
            'details': f"Found {duplicates} duplicate rows ({duplicates/len(df)*100:.1f}%)"
        })
    
    # Check for constant columns
    constant_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                     if df[col].nunique() == 1]
    if constant_cols:
        issues.append({
            'type': 'Constant Columns',
            'severity': 'High',
            'details': f"Columns with no variation: {', '.join(constant_cols)}"
        })
    
    # Check for high cardinality
    high_card = [col for col in df.columns if df[col].nunique() > len(df) * 0.9]
    if high_card:
        issues.append({
            'type': 'High Cardinality',
            'severity': 'Medium',
            'details': f"Columns with too many unique values: {', '.join(high_card)}"
        })
    
    return issues

def get_model_explanation(model_name):
    """Return simple explanations for each model"""
    explanations = {
        'Logistic Regression': {
            'emoji': '📊',
            'simple': 'Draws a straight line to separate classes',
            'when': 'Use when data is linearly separable and you need quick training',
            'pros': '✅ Fast, interpretable, works well with large datasets',
            'cons': '❌ Cannot capture complex patterns, assumes linear relationships'
        },
        'Decision Tree': {
            'emoji': '🌳',
            'simple': 'Makes decisions like a flowchart - if/then rules',
            'when': 'Use when you need to explain decisions easily',
            'pros': '✅ Easy to understand, handles non-linear data, no scaling needed',
            'cons': '❌ Can overfit easily, unstable with small data changes'
        },
        'Random Forest': {
            'emoji': '🌲',
            'simple': 'Many decision trees voting together',
            'when': 'Use when you want high accuracy and can sacrifice speed',
            'pros': '✅ Very accurate, handles overfitting well, robust',
            'cons': '❌ Slower training, harder to interpret, uses more memory'
        },
        'SVM': {
            'emoji': '🎯',
            'simple': 'Finds the widest gap between classes',
            'when': 'Use for small-to-medium datasets with clear margins',
            'pros': '✅ Effective in high dimensions, memory efficient',
            'cons': '❌ Slow on large datasets, needs feature scaling'
        },
        'K-Nearest Neighbors': {
            'emoji': '👥',
            'simple': 'Looks at nearest neighbors to make predictions',
            'when': 'Use when you have small dataset and pattern matching matters',
            'pros': '✅ Simple, no training phase, adapts to new data easily',
            'cons': '❌ Slow predictions, sensitive to irrelevant features'
        }
    }
    return explanations.get(model_name, {})

def generate_sample_data():
    """Generate sample data with clear patterns"""
    np.random.seed(42)
    n_samples = 1000
    
    # Create features with meaningful names
    age = np.random.randint(18, 80, n_samples)
    income = np.random.randint(20000, 150000, n_samples)
    credit_score = np.random.randint(300, 850, n_samples)
    years_employed = np.random.randint(0, 40, n_samples)
    debt_ratio = np.random.uniform(0, 1, n_samples)
    
    # Create target based on logical rules
    approval_score = (
        (credit_score - 300) / 550 * 0.4 +
        (income - 20000) / 130000 * 0.3 +
        (1 - debt_ratio) * 0.2 +
        (years_employed / 40) * 0.1
    )
    
    target = (approval_score > 0.5).astype(int)
    
    # Add some noise
    noise_idx = np.random.choice(n_samples, size=int(n_samples * 0.1))
    target[noise_idx] = 1 - target[noise_idx]
    
    df = pd.DataFrame({
        'Age': age,
        'Income': income,
        'Credit_Score': credit_score,
        'Years_Employed': years_employed,
        'Debt_Ratio': debt_ratio,
        'Loan_Approved': target
    })
    
    return df

def train_models(X_train, X_test, y_train, y_test, selected_models, feature_names):
    """Train selected models and return comprehensive results"""
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42, probability=True),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (name, model) in enumerate(models.items()):
        if name in selected_models:
            status_text.text(f"Training {name}...")
            
            # Measure training time
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Get prediction probabilities if available
            try:
                y_proba = model.predict_proba(X_test)
            except:
                y_proba = None
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            # Feature importance
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                feature_importance = np.abs(model.coef_[0])
            
            results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'probabilities': y_proba,
                'confusion_matrix': cm,
                'model': model,
                'training_time': training_time,
                'feature_importance': feature_importance
            }
            
            progress_bar.progress((idx + 1) / len(selected_models))
    
    status_text.text("Training complete! ✅")
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    
    return results

def plot_accuracy_comparison(results):
    """Create accuracy comparison with training time"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in models]
    times = [results[m]['training_time'] for m in models]
    
    # Accuracy plot
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    bars = ax1.bar(models, accuracies, color=colors[:len(models)])
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Training time plot
    bars2 = ax2.bar(models, times, color=colors[:len(models)], alpha=0.7)
    ax2.set_ylabel('Training Time (seconds)', fontsize=12)
    ax2.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}s', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(cm, model_name):
    """Create confusion matrix heatmap"""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=10)
    ax.set_xlabel('Predicted Label', fontsize=10)
    plt.tight_layout()
    return fig

def plot_feature_importance(importance, feature_names, model_name):
    """Plot feature importance"""
    if importance is None:
        return None
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Sort features by importance
    indices = np.argsort(importance)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importance = importance[indices]
    
    bars = ax.barh(sorted_features, sorted_importance, color='#2ecc71')
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f'Feature Importance - {model_name}', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Add values on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.3f}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    return fig

# Sidebar
st.sidebar.header("⚙️ Navigation")
page = st.sidebar.radio("Go to:", ["📊 Data & Quality", "🎯 Model Training", "📈 Results", "🔮 Make Predictions", "📚 Learn Models"])

# PAGE 1: Data Upload & Quality Check
if page == "📊 Data & Quality":
    st.header("📊 Data Upload & Quality Check")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Option 1: Upload CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded! Shape: {df.shape}")
            st.session_state.df = df
        
    with col2:
        st.subheader("Option 2: Sample Data")
        if st.button("Generate Sample Data", type="primary"):
            df = generate_sample_data()
            st.session_state.df = df
            st.success("✅ Sample data generated!")
            st.info("💡 Sample data simulates loan approval predictions")
    
    if 'df' in st.session_state:
        df = st.session_state.df
        
        # Data preview
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        # Data quality check
        st.subheader("🔍 Data Quality Analysis")
        issues = check_data_quality(df)
        
        if not issues:
            st.success("✅ No data quality issues detected! Your data looks great.")
        else:
            for issue in issues:
                severity_colors = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
                st.warning(f"{severity_colors[issue['severity']]} **{issue['type']}** ({issue['severity']} Priority)\n\n{issue['details']}")
        
        # Statistics
        with st.expander("📊 Detailed Statistics"):
            st.write(df.describe())
        
        # Column selection
        st.subheader("🎯 Select Target Column")
        target_col = st.selectbox("What do you want to predict?", df.columns)
        st.session_state.target_col = target_col
        
        # Show class distribution
        if target_col:
            st.write("**Class Distribution:**")
            class_dist = df[target_col].value_counts()
            col1, col2 = st.columns(2)
            with col1:
                st.write(class_dist)
            with col2:
                fig, ax = plt.subplots(figsize=(6, 4))
                class_dist.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c'])
                ax.set_title('Target Variable Distribution')
                ax.set_xlabel('Class')
                ax.set_ylabel('Count')
                plt.xticks(rotation=0)
                st.pyplot(fig)

# PAGE 2: Model Training
elif page == "🎯 Model Training":
    st.header("🎯 Train Machine Learning Models")
    
    if 'df' not in st.session_state:
        st.warning("⚠️ Please upload data first in the 'Data & Quality' page!")
    else:
        # Model selection with explanations
        st.subheader("Select Models to Train")
        
        col1, col2 = st.columns(2)
        
        with col1:
            lr = st.checkbox("📊 Logistic Regression", value=True)
            dt = st.checkbox("🌳 Decision Tree", value=True)
            rf = st.checkbox("🌲 Random Forest", value=True)
        
        with col2:
            svm = st.checkbox("🎯 SVM", value=True)
            knn = st.checkbox("👥 K-Nearest Neighbors", value=False)
        
        selected_models = []
        if lr: selected_models.append('Logistic Regression')
        if dt: selected_models.append('Decision Tree')
        if rf: selected_models.append('Random Forest')
        if svm: selected_models.append('SVM')
        if knn: selected_models.append('K-Nearest Neighbors')
        
        # Training parameters
        st.subheader("⚙️ Training Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
        
        with col2:
            random_state = st.number_input("Random seed", 0, 100, 42)
        
        # Train button
        if st.button("🚀 Train Models", type="primary", use_container_width=True):
            if len(selected_models) == 0:
                st.error("Please select at least one model!")
            else:
                with st.spinner("Training models..."):
                    df = st.session_state.df
                    target_col = st.session_state.target_col
                    
                    X = df.drop(columns=[target_col])
                    y = df[target_col]
                    
                    # Store feature names
                    st.session_state.feature_names = list(X.columns)
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    )
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    st.session_state.scaler = scaler
                    
                    # Train models
                    results = train_models(X_train_scaled, X_test_scaled, y_train, y_test, 
                                         selected_models, st.session_state.feature_names)
                    
                    st.session_state.results = results
                    st.session_state.y_test = y_test
                    st.session_state.X_test = X_test
                    st.session_state.trained = True
                    
                st.success("✅ Training complete! Check the Results page.")
                st.balloons()

# PAGE 3: Results
elif page == "📈 Results":
    st.header("📈 Training Results")
    
    if not st.session_state.trained:
        st.info("ℹ️ Train some models first to see results here!")
    else:
        results = st.session_state.results
        
        # Performance overview
        st.subheader("🏆 Performance Overview")
        cols = st.columns(len(results))
        for idx, (model_name, result) in enumerate(results.items()):
            with cols[idx]:
                st.metric(
                    model_name, 
                    f"{result['accuracy']:.1%}",
                    f"{result['training_time']:.2f}s"
                )
        
        # Best model
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        st.success(f"🏆 **Best Model:** {best_model[0]} with {best_model[1]['accuracy']:.2%} accuracy")
        
        # Visualizations
        st.subheader("📊 Performance Visualizations")
        fig = plot_accuracy_comparison(results)
        st.pyplot(fig)
        
        # Feature importance
        st.subheader("🔍 Feature Importance Analysis")
        
        models_with_importance = {name: res for name, res in results.items() 
                                 if res['feature_importance'] is not None}
        
        if models_with_importance:
            selected_model = st.selectbox("Select model to view feature importance:", 
                                         list(models_with_importance.keys()))
            
            fig = plot_feature_importance(
                models_with_importance[selected_model]['feature_importance'],
                st.session_state.feature_names,
                selected_model
            )
            if fig:
                st.pyplot(fig)
                
                st.info("💡 **Feature Importance** shows which features have the most impact on predictions. Higher values mean the feature is more important for the model's decisions.")
        else:
            st.info("Feature importance is only available for tree-based models and Logistic Regression.")
        
        # Detailed results
        st.subheader("📋 Detailed Model Results")
        
        for model_name, result in results.items():
            with st.expander(f"{model_name} - {result['accuracy']:.2%} accuracy"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Accuracy", f"{result['accuracy']:.2%}")
                    st.metric("Training Time", f"{result['training_time']:.3f}s")
                    
                    report = classification_report(
                        st.session_state.y_test, 
                        result['predictions'],
                        output_dict=True
                    )
                    st.write("**Classification Report:**")
                    st.json(report)
                
                with col2:
                    fig = plot_confusion_matrix(result['confusion_matrix'], model_name)
                    st.pyplot(fig)
        
        # Download results
        st.subheader("💾 Export Results")
        
        results_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [results[m]['accuracy'] for m in results.keys()],
            'Training Time (s)': [results[m]['training_time'] for m in results.keys()]
        }).sort_values('Accuracy', ascending=False)
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="ml_results.csv",
            mime="text/csv"
        )

# PAGE 4: Make Predictions
elif page == "🔮 Make Predictions":
    st.header("🔮 Make Predictions with Your Models")
    
    if not st.session_state.trained:
        st.warning("⚠️ Please train models first!")
    else:
        st.markdown("Enter values for each feature to get predictions from all trained models.")
        
        # Select model
        model_name = st.selectbox("Choose a model:", list(st.session_state.results.keys()))
        
        st.subheader("📝 Enter Feature Values")
        
        # Create input fields for each feature
        feature_values = {}
        cols = st.columns(2)
        
        for idx, feature in enumerate(st.session_state.feature_names):
            with cols[idx % 2]:
                # Get sample statistics
                sample_data = st.session_state.df[feature]
                mean_val = float(sample_data.mean())
                min_val = float(sample_data.min())
                max_val = float(sample_data.max())
                
                feature_values[feature] = st.number_input(
                    f"{feature}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    help=f"Range: {min_val:.2f} to {max_val:.2f}"
                )
        
        if st.button("🎯 Predict", type="primary", use_container_width=True):
            # Prepare input
            input_data = pd.DataFrame([feature_values])
            input_scaled = st.session_state.scaler.transform(input_data)
            
            # Get prediction
            model = st.session_state.results[model_name]['model']
            prediction = model.predict(input_scaled)[0]
            
            # Get probability if available
            try:
                proba = model.predict_proba(input_scaled)[0]
                confidence = max(proba) * 100
            except:
                proba = None
                confidence = None
            
            # Display result
            st.markdown("---")
            st.subheader("🎉 Prediction Result")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Model Used", model_name)
            
            with col2:
                pred_label = "✅ Class 1" if prediction == 1 else "❌ Class 0"
                st.metric("Prediction", pred_label)
            
            with col3:
                if confidence:
                    st.metric("Confidence", f"{confidence:.1f}%")
            
            if proba is not None:
                st.write("**Probability Distribution:**")
                prob_df = pd.DataFrame({
                    'Class': [0, 1],
                    'Probability': proba
                })
                
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.barh(prob_df['Class'].astype(str), prob_df['Probability'], 
                       color=['#e74c3c', '#2ecc71'])
                ax.set_xlabel('Probability')
                ax.set_xlim(0, 1)
                ax.set_title('Prediction Probabilities')
                for i, v in enumerate(prob_df['Probability']):
                    ax.text(v + 0.02, i, f'{v:.2%}', va='center')
                st.pyplot(fig)
        
        # Batch prediction
        st.markdown("---")
        st.subheader("📦 Batch Predictions")
        st.markdown("Upload a CSV file with the same features to get predictions for multiple rows.")
        
        batch_file = st.file_uploader("Upload CSV for batch predictions", type=['csv'], key='batch')
        
        if batch_file is not None:
            batch_df = pd.read_csv(batch_file)
            
            # Check if columns match
            missing_cols = set(st.session_state.feature_names) - set(batch_df.columns)
            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
            else:
                batch_scaled = st.session_state.scaler.transform(batch_df[st.session_state.feature_names])
                
                predictions = model.predict(batch_scaled)
                batch_df['Prediction'] = predictions
                
                st.success(f"✅ Generated {len(predictions)} predictions!")
                st.dataframe(batch_df)
                
                # Download predictions
                csv = batch_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Predictions",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )

# PAGE 5: Learn About Models
elif page == "📚 Learn Models":
    st.header("📚 Understanding Machine Learning Models")
    st.markdown("Learn how each algorithm works and when to use them!")
    
    models_info = [
        'Logistic Regression',
        'Decision Tree',
        'Random Forest',
        'SVM',
        'K-Nearest Neighbors'
    ]
    
    for model_name in models_info:
        info = get_model_explanation(model_name)
        
        with st.expander(f"{info['emoji']} {model_name}", expanded=False):
            st.markdown(f"### How it works:")
            st.info(info['simple'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### ✅ Advantages")
                st.success(info['pros'])
                
            with col2:
                st.markdown("### ❌ Disadvantages")
                st.error(info['cons'])
            
            st.markdown("### 💡 When to use:")
            st.warning(info['when'])
    
    # Comparison table
    st.subheader("📊 Quick Comparison")
    
    comparison_data = {
        'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM', 'KNN'],
        'Speed': ['⚡⚡⚡', '⚡⚡', '⚡', '⚡⚡', '⚡'],
        'Accuracy': ['⭐⭐⭐', '⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐'],
        'Interpretability': ['⭐⭐⭐⭐', '⭐⭐⭐⭐⭐', '⭐⭐', '⭐', '⭐⭐'],
        'Best For': ['Linear patterns', 'Simple rules', 'Complex patterns', 'Clear boundaries', 'Similar patterns']
    }
    
    st.table(pd.DataFrame(comparison_data))
    
    st.info("""
    **Legend:**  
    ⚡ = Speed (more = faster training)  
    ⭐ = Rating (more = better performance)
    """)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.info("""
**🚀 Quick Start:**
1. Upload data or generate sample
2. Check data quality
3. Train models
4. View results & insights
5. Make predictions!

**✨ New Features:**
- 🔍 Data quality checker
- 📊 Feature importance
- 🔮 Make predictions
- 📚 Model explainer
- ⚡ Training time comparison
""")