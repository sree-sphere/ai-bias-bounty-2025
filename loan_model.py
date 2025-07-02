#!/usr/bin/env python
"""
Loan Approval Prediction Model with bias analysis and fairness auditing
This script performs:
1. Data cleaning and preprocessing
2. Feature engineering with SHAP based selection
3. Model training with 'Logistic Regression'
4. Comprehensive fairness auditing and bias detection
5. Prediction on test dataset
6. Bias visualization generation
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, roc_auc_score
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from fairlearn.metrics import MetricFrame, selection_rate, false_negative_rate
import shap
import tabulate
import pickle
warnings.filterwarnings('ignore')
shap.initjs()

class LoanFairnessAnalyzer:
    """
    Complete pipeline for loan approval prediction with fairness analysis
    """
    
    def __init__(self):
        """Initialize the analyzer with configuration parameters"""
        self.sensitive_attrs = ['Gender', 'Race', 'Age_Group', 'Citizenship_Status', 'Disability_Status', 'Criminal_Record']
        self.categorical_cols = ['Gender', 'Race', 'Age_Group', 'Employment_Type', 
                                'Education_Level', 'Citizenship_Status', 'Language_Proficiency',
                                'Disability_Status', 'Criminal_Record', 'Zip_Code_Group']
        self.continuous_cols = ['Income', 'Credit_Score', 'Loan_Amount']
        
        # Model components
        self.model = None
        self.scaler = None
        self.imputer = None
        self.top_features = None
        self.explainer = None
        
        # output directories
        os.makedirs('images', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
    def load_and_preprocess_data(self, filepath):
        """
        Initial preprocessing of the dataset
        
        Args:
            filepath (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        print("Loading and preprocessing data...")
        
        df = pd.read_csv(filepath, encoding='latin1')
        
        # Convert target variable to binary
        if 'Loan_Approved' in df.columns:
            df['Loan_Approved'] = df['Loan_Approved'].map({'Approved': 1, 'Denied': 0})
        
        # Drop ID column if present
        df = df.drop(columns=['ID'], errors='ignore')
        
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
        
    def perform_eda(self, df, save_plots=True):
        """
        Perform exploratory data analysis and save visualization plots
        
        Args:
            df (pd.DataFrame): Input dataframe
            save_plots (bool): Whether to save plots to images folder
        """
        print("Performing EDA...")
        
        # A: Categorical variables analysis
        fig, axes = plt.subplots(nrows=len(self.categorical_cols), ncols=2, figsize=(12, 6*len(self.categorical_cols)))
        
        for i, col in enumerate(self.categorical_cols):
            # Distribution plot
            df[col].value_counts().plot(kind='barh', ax=axes[i, 0], color=sns.color_palette('Dark2'))
            axes[i, 0].set_title(f'Distribution of {col}')
            axes[i, 0].spines[['top', 'right']].set_visible(False)
            
            # Approval rate by category
            if 'Loan_Approved' in df.columns:
                approval_rates = df.groupby([col, 'Loan_Approved']).size().unstack()
                approval_rates.plot(kind='bar', ax=axes[i, 1], color=sns.color_palette('Dark2'))
                axes[i, 1].set_title(f'{col} vs Loan Approved')
                axes[i, 1].spines[['top', 'right']].set_visible(False)
                axes[i, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('images/categorical_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # B: Continuous variables analysis
        if 'Loan_Approved' in df.columns:
            fig, axes = plt.subplots(nrows=len(self.continuous_cols), ncols=2, figsize=(12, 6*len(self.continuous_cols)))
            
            for i, col in enumerate(self.continuous_cols):
                # Distribution plot
                sns.histplot(df[col], ax=axes[i, 0], kde=True)
                axes[i, 0].set_title(f'Distribution of {col}')
                axes[i, 0].spines[['top', 'right']].set_visible(False)
                
                # Box plot by approval status
                sns.boxplot(x='Loan_Approved', y=col, data=df, ax=axes[i, 1])
                axes[i, 1].set_title(f'{col} by Loan Approved')
                axes[i, 1].spines[['top', 'right']].set_visible(False)
            
            plt.tight_layout()
            if save_plots:
                plt.savefig('images/continuous_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def generate_bias_analysis_plots(self, df):
        """
        Generate specific bias analysis visualizations
        
        Args:
            df (pd.DataFrame): Input dataframe with predictions
        """
        print("Generating bias analysis plots...")
        
        # Approval rates by demographic groups
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, attr in enumerate(self.sensitive_attrs):
            if attr in df.columns:
                approval_rates = df.groupby(attr)['Loan_Approved'].mean()
                approval_rates.plot(kind='bar', ax=axes[i], color='skyblue', edgecolor='black')
                axes[i].set_title(f'Loan Approval Rate by {attr}', fontsize=12, fontweight='bold')
                axes[i].set_ylabel('Approval Rate')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].spines[['top', 'right']].set_visible(False)
                
                # Add value labels on bars
                for j, v in enumerate(approval_rates.values):
                    axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('images/bias_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def prepare_features(self, df):
        """
        Prepare features for model training
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            tuple: (X, y) feature matrix and target vector
        """
        print("Preparing features...")
        
        # Separate features and target
        drop_cols = ['ID', 'Zip_Code_Group', 'Loan_Approved']
        X = df.drop(columns=drop_cols, errors='ignore')
        y = df['Loan_Approved'] if 'Loan_Approved' in df.columns else None
        
        # Binary encode sensitive attributes with 2 levels
        for col in self.sensitive_attrs:
            if col in X.columns and X[col].nunique() == 2:
                vals = X[col].unique()
                X[col] = X[col].map({vals[0]: 0, vals[1]: 1})
        
        # Handle multi class sensitive attributes
        for col in self.sensitive_attrs:
            if col in X.columns and X[col].dtype == 'object' and X[col].nunique() > 2:
                # For sensitive attributes with more than 2 categories, use label encoding
                X[col] = pd.factorize(X[col])[0]
        
        # One hot encode remaining categorical columns
        categorical_to_encode = [col for col in X.select_dtypes(include='object').columns]
        if categorical_to_encode:
            X = pd.get_dummies(X, columns=categorical_to_encode, drop_first=True)
        
        return X, y
    
    def feature_selection_with_shap(self, X, y, n_features=20):
        """
        Perform feature selection using SHAP values
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target vector
            n_features (int): Number of top features to select
            
        Returns:
            list: Selected feature names
        """
        print("Performing SHAP-based feature selection...")
        
        # Impute missing values with KNN (none anyways)
        self.imputer = KNNImputer(n_neighbors=5)
        X_imputed = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
        
        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, stratify=y, random_state=42)
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train baseline model for SHAP analysis
        base_model = LogisticRegression(max_iter=200, penalty='l1', solver='liblinear', class_weight='balanced')
        base_model.fit(X_train_scaled, y_train)
        
        # SHAP explanation
        self.explainer = shap.LinearExplainer(base_model, X_train_scaled, feature_perturbation="correlation_dependent")
        shap_values = self.explainer.shap_values(X_train_scaled)
        
        # Finding feature importance based on SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({'feature': X.columns, 'importance': mean_abs_shap})
        shap_df = shap_df.sort_values('importance', ascending=False)
        
        # Get top features
        self.top_features = shap_df.head(n_features)['feature'].tolist()
        
        # Generate SHAP plots
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, features=X_train_scaled, feature_names=X.columns, max_display=10, show=False)
        plt.tight_layout()
        plt.savefig('images/shap_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Selected {len(self.top_features)} top features based on SHAP values")
        return self.top_features
    
    def train_model(self, X, y):
        """
        Train the final model on selected features
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target vector
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, predictions)
        """
        print("Training final model...")
        
        # Use only top features
        X_selected = X[self.top_features]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, stratify=y, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train final model
        self.model = LogisticRegression(max_iter=200, penalty='l1',solver='liblinear', class_weight='balanced')
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        print("Model Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
        print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return X_train, X_test, y_train, y_test, y_pred, y_pred_proba
    

    def perform_fairness_audit(self, df, X_test, y_test, y_pred):
        """
        Perform comprehensive fairness auditing
        
        Args:
            df (pd.DataFrame): Original dataframe
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): True labels
            y_pred (np.array): Predicted labels
        """
        print("Performing fairness audit...")
        
        # Making test dataframe with predictions and sensitive attributes
        test_df = pd.DataFrame({'y_true': y_test.reset_index(drop=True),'y_pred': y_pred})
        
        # Add sensitive attributes
        for attr in self.sensitive_attrs:
            if attr in df.columns:
                test_df[attr] = df.loc[X_test.index, attr].reset_index(drop=True)
        
        # A: Univariate fairness analysis
        print("\n" + "="*50)
        print("UNIVARIATE FAIRNESS ANALYSIS")
        print("="*50)
        
        fairness_results = {
            'univariate': [],
            'intersectional': {},
            'detailed_metrics': {},
            'disparate_impact': []
        }
        
        for attr in self.sensitive_attrs:
            if attr in test_df.columns:
                approval_rates = test_df.groupby(attr)['y_pred'].mean()
                print(f"\nApproval rates by {attr}:")
                print(tabulate.tabulate(approval_rates.to_frame(), headers=['Group', 'Approval Rate'], tablefmt='grid'))
                
                # Record fairness results
                for group, rate in approval_rates.items():
                    fairness_results['univariate'].append({
                        'Attribute': attr,
                        'Group': group,
                        'Approval_Rate': rate,
                        'Sample_Size': len(test_df[test_df[attr] == group])
                    })
        
        # B: Intersectional analysis
        print("\n" + "="*50)
        print("INTERSECTIONAL ANALYSIS")
        print("="*50)
        
        top_sensitive = ['Criminal_Record', 'Gender', 'Disability_Status']
        available_sensitive = [attr for attr in top_sensitive if attr in test_df.columns]
        
        if len(available_sensitive) >= 2:
            test_df['intersectional_group'] = test_df[available_sensitive].astype(str).agg('-'.join, axis=1)
            group_counts = test_df['intersectional_group'].value_counts()
            valid_groups = group_counts[group_counts >= 30].index
            
            if len(valid_groups) > 0:
                intersectional_rates = test_df[test_df['intersectional_group'].isin(valid_groups)].groupby('intersectional_group')['y_pred'].mean()
                print("Intersectional approval rates (groups with n >= 30):")
                print(tabulate.tabulate(intersectional_rates.to_frame(), headers=['Group', 'Approval Rate'], tablefmt='grid'))
                fairness_results['intersectional'] = intersectional_rates.to_dict()
        
        # C: MetricFrame analysis
        print("\n" + "="*50)
        print("DETAILED FAIRNESS METRICS")
        print("="*50)
        
        for attr in available_sensitive:
            if attr in test_df.columns:
                try:
                    mf = MetricFrame(
                        metrics={'selection_rate': selection_rate, 'fnr': false_negative_rate},
                        y_true=test_df['y_true'],
                        y_pred=test_df['y_pred'],
                        sensitive_features=test_df[attr]
                    )
                    
                    print(f"\nMetrics by {attr}:")
                    print(tabulate.tabulate(mf.by_group, headers='keys', tablefmt='grid'))
                    print(f"\nDifferences for {attr}:")
                    print(tabulate.tabulate(mf.difference().to_frame(), headers=['Metric', 'Difference'], tablefmt='grid'))
                    fairness_results['detailed_metrics'][attr] = {
                        'by_group': mf.by_group.to_dict(),
                        'differences': mf.difference().to_dict()
                    }
                except Exception as e:
                    print(f"Could not compute MetricFrame for {attr}: {e}")
        
        # D: AIF360 Disparate Impact Analysis
        print("\n" + "="*50)
        print("DISPARATE IMPACT ANALYSIS")
        print("="*50)
        
        try:
            # Prepare data for AIF360
            sf_test = df.loc[X_test.index, self.sensitive_attrs].reset_index(drop=True)
            sf_test_encoded = sf_test.copy()
            
            for col in self.sensitive_attrs:
                if col in sf_test_encoded.columns and sf_test_encoded[col].dtype == 'object':
                    sf_test_encoded[col] = pd.factorize(sf_test_encoded[col])[0]
            
            # Create AIF360 dataset
            df_for_aif = pd.concat([
                X_test.reset_index(drop=True),
                sf_test_encoded,
                y_test.reset_index(drop=True).rename("Loan_Approved")
            ], axis=1)
            
            aif_test = BinaryLabelDataset(
                df=df_for_aif,
                label_names=['Loan_Approved'],
                protected_attribute_names=[attr for attr in self.sensitive_attrs if attr in sf_test_encoded.columns]
            )
            
            # Create prediction dataset
            aif_pred = aif_test.copy()
            aif_pred.labels = y_pred.reshape(-1, 1)
            
            # Calculate disparate impact for each sensitive attribute
            for attr in self.sensitive_attrs:
                if attr in sf_test_encoded.columns:
                    try:
                        metric = BinaryLabelDatasetMetric(
                            aif_pred,
                            unprivileged_groups=[{attr: 0}],
                            privileged_groups=[{attr: 1}]
                        )
                        di = metric.disparate_impact()
                        fairness_results['disparate_impact'].append({'Attribute': attr, 'Disparate_Impact': di})
                        print(f"{attr} Disparate Impact: {di:.4f}")
                    except Exception as e:
                        print(f"Could not compute disparate impact for {attr}: {e}")
        except Exception as e:
            print(f"AIF360 analysis failed: {e}")
        
        # Save fairness results
        with open('results/fairness_analysis.json', 'w') as f:
            json.dump(fairness_results, f, indent=4)
            
    def predict_test_data(self, test_filepath):
        """
        Make predictions on test dataset
        
        Args:
            test_filepath (str): path to test CSV file
            
        Returns:
            pd.DataFrame: Predictions dataframe
        """
        print("Making predictions on test data...")
        
        test_df = pd.read_csv(test_filepath, encoding='latin1')
        
        # Store ID
        test_ids = test_df['ID']
        
        # Prepare test features using the same encoding as training
        X_test, _ = self.prepare_features(test_df)
        
        # Select top features
        X_test_selected = X_test[self.top_features]
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        # Make predictions
        predictions = self.model.predict(X_test_scaled)
        
        # Submission dataframe
        submission_df = pd.DataFrame({'ID': test_ids,'LoanApproved': predictions})
        submission_df.to_csv('results/submission.csv', index=False)
        print(f"Predictions saved to submission.csv ({len(predictions)} predictions)")
        
        return submission_df
    
    def save_training_results(self, X_test, y_test, y_pred, y_pred_proba):
        """
        Save training results for analysis
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): True labels
            y_pred (np.array): Predicted labels
            y_pred_proba (np.array): Predicted probabilities
        """
        print("Saving training results...")
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'True_Label': y_test.reset_index(drop=True),
            'Predicted_Label': y_pred,
            'Predicted_Probability': y_pred_proba
        })
        
        # Add test features
        for col in X_test.columns:
            results_df[f'Feature_{col}'] = X_test[col].reset_index(drop=True)
        
        results_df.to_csv('results/model_training_results.csv', index=False)
        print("Training results saved to model_training_results.csv")
    
    def generate_final_visualizations(self, y_test, y_pred_proba):
        """
        Generate final visualization plots for bias analysis
        
        Args:
            y_test (pd.Series): True labels
            y_pred_proba (np.array): Predicted probabilities
        """
        print("Generating final visualizations...")
        
        # Predicted probability distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(y_pred_proba, bins=30, kde=True, color='blue', stat='density', label='Predicted Probabilities')
        plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold (0.5)')
        plt.title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Probability of Loan Approval')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('images/probability_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("All visualizations saved to images/ folder")
    
    def run_complete_analysis(self, train_filepath, test_filepath=None):
        """
        Run the complete analysis pipeline
        
        Args:
            train_filepath (str): Path to training data CSV
            test_filepath (str): Path to test data CSV (optional)
        """
        print("Starting complete loan fairness analysis...")
        print("="*60)
        
        # Load and preprocess training data
        df = self.load_and_preprocess_data(train_filepath)
        
        # Perform EDA
        self.perform_eda(df)
        
        # Generate initial bias analysis
        self.generate_bias_analysis_plots(df)
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Feature selection with SHAP
        self.feature_selection_with_shap(X, y)
        
        # Train model
        X_train, X_test, y_train, y_test, y_pred, y_pred_proba = self.train_model(X, y)
        
        # Perform fairness audit
        self.perform_fairness_audit(df, X_test, y_test, y_pred)
        
        # Save training results
        self.save_training_results(X_test, y_test, y_pred, y_pred_proba)
        
        # Generate final visualizations
        self.generate_final_visualizations(y_test, y_pred_proba)
        
        # Predict on test data
        self.predict_test_data(test_filepath)
        
        # Save the trained model
        with open('results/trained_model.pkl', 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'imputer': self.imputer,
                'top_features': self.top_features
            }, f)


def main():
    """Main execution function"""
    # Initialize fairness analyzer
    analyzer = LoanFairnessAnalyzer()
    
    train_file = "datasets/loan_access_dataset.csv"
    test_file = "datasets/test.csv"
    
    # Run complete analysis
    analyzer.run_complete_analysis(train_file, test_file)


if __name__ == "__main__":
    main()