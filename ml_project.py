"""
Machine Learning Educational Project
A comprehensive introduction to ML algorithms with practical examples.

This project demonstrates:
- Data preprocessing
- Multiple ML algorithms
- Model evaluation
- Visualization
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

class MLLearningProject:
    """
    A class to demonstrate various machine learning algorithms
    and their performance on a dataset.
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
    
    def generate_sample_data(self, n_samples=1000):
        """
        Generate a sample dataset for classification.
        In a real project, you'd load data from a CSV or API.
        """
        print("Generating sample data...")
        
        # Create features
        X = np.random.randn(n_samples, 5)
        
        # Create target (binary classification)
        # Simple rule: if sum of first 3 features > 0, class 1, else class 0
        y = (X[:, 0] + X[:, 1] + X[:, 2] > 0).astype(int)
        
        # Add some noise
        noise_idx = np.random.choice(n_samples, size=int(n_samples * 0.1))
        y[noise_idx] = 1 - y[noise_idx]
        
        return X, y
    
    def preprocess_data(self, X, y, test_size=0.2):
        """
        Split and scale the data.
        """
        print(f"\nPreprocessing data...")
        print(f"Total samples: {len(X)}")
        
        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features (important for many ML algorithms)
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"Training samples: {len(self.X_train)}")
        print(f"Testing samples: {len(self.X_test)}")
    
    def initialize_models(self):
        """
        Initialize different ML models to compare.
        """
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42)
        }
        print(f"\nInitialized {len(self.models)} models")
    
    def train_and_evaluate(self):
        """
        Train all models and evaluate their performance.
        """
        print("\n" + "="*50)
        print("TRAINING AND EVALUATION")
        print("="*50)
        
        for name, model in self.models.items():
            print(f"\n{name}:")
            print("-" * 30)
            
            # Train
            model.fit(self.X_train, self.y_train)
            
            # Predict
            y_pred = model.predict(self.X_test)
            
            # Evaluate
            accuracy = accuracy_score(self.y_test, y_pred)
            self.results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred))
    
    def visualize_results(self):
        """
        Create visualizations of model performance.
        """
        print("\nGenerating visualizations...")
        
        # Accuracy comparison
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        models = list(self.results.keys())
        accuracies = [self.results[m]['accuracy'] for m in models]
        
        bars = plt.bar(models, accuracies, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy Comparison')
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # Confusion matrix for best model
        plt.subplot(1, 2, 2)
        best_model = max(self.results.items(), key=lambda x: x[1]['accuracy'])[0]
        cm = confusion_matrix(self.y_test, self.results[best_model]['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {best_model}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('ml_results.png', dpi=300, bbox_inches='tight')
        print("Visualization saved as 'ml_results.png'")
        plt.show()
    
    def run_complete_pipeline(self):
        """
        Execute the complete ML pipeline.
        """
        print("\n" + "="*50)
        print("MACHINE LEARNING PIPELINE")
        print("="*50)
        
        # Generate data
        X, y = self.generate_sample_data()
        
        # Preprocess
        self.preprocess_data(X, y)
        
        # Initialize models
        self.initialize_models()
        
        # Train and evaluate
        self.train_and_evaluate()
        
        # Visualize
        self.visualize_results()
        
        print("\n" + "="*50)
        print("Pipeline complete!")
        print("="*50)

def main():
    """
    Main function to run the ML project.
    """
    # Create instance
    ml_project = MLLearningProject()
    
    # Run complete pipeline
    ml_project.run_complete_pipeline()

if __name__ == "__main__":
    main()