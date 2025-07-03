# Loan Approval Fairness Analysis

A comprehensive ML project that predicts loan approvals while detecting and analyzing algorithmic bias across demographic groups.

[Runthrough Video Link](https://drive.google.com/file/d/13BZb6elftSa1GwGODAS0uPpqdqdnTPTy/view?usp=sharing)

## Problem Statement

The financial sector has long faced scrutiny over systemic inequities embedded in its decision-making processes, particularly in loan approvals. Mortgage lending, in particular, reveals deeply entrenched disparities across race, gender, age, and geography. In response to increasing demand for fairness in AI-powered decision systems, this track is designed to simulate a real-world audit scenario, where participants step into the role of ethical model builders, bias investigators, and fairness advocates. 

Loan approval systems can perpetuate historical biases and discriminate against protected groups, leading to unfair lending practices. This project addresses the critical need to:

- **Detect algorithmic bias** in loan approval decisions
- **Quantify fairness disparities** across demographic groups (gender, race, disability status, etc.)
- **Build transparent models** that balance predictive accuracy with fairness considerations
- **Provide actionable insights** for bias mitigation in financial services

The challenge involves analyzing a loan dataset with sensitive attributes and building a model that not only predicts loan approvals accurately but also identifies and visualizes potential discriminatory patterns.

## Model Approach & Fairness Considerations

Note: More elaborated explanation on my approach and findings in Additional_Report.md

### Machine Learning Pipeline

1. **Data Preprocessing & Feature Engineering**
   - KNN imputation for missing values
   - Binary encoding for sensitive attributes
   - One-hot encoding for categorical variables
   - Standard scaling for numerical features

2. **Feature Selection**
   - SHAP (SHapley Additive exPlanations) based feature importance
   - Linear explainer for transparent feature ranking
   - Top 20 most important features selected for model training

3. **Model Training**
   - Logistic Regression with L1 regularization
   - Class balancing to handle imbalanced target variable
   - Cross-validation for robust performance estimation

### Fairness Analysis Framework

#### Bias Detection Methods
- **Univariate Analysis**: Approval rates across individual demographic groups
- **Intersectional Analysis**: Multi-dimensional bias detection (Criminal_Record x Gender x Disability_Status)
- **Disparate Impact**: Statistical parity assessment using AIF360 metrics
- **False Positive/Negative Rates**: Error rate disparities across groups

#### Fairness Metrics Implemented
- **Selection Rate**: Proportion of positive predictions per group
- **False Negative Rate**: Missed opportunities for each demographic
- **Disparate Impact Ratio**: Ratio of approval rates between groups
- **Statistical Parity**: Equal treatment regardless of sensitive attributes

#### Visualization Strategy
- Approval rate bar charts by demographic groups
- SHAP feature importance plots
- Prediction probability distributions
- Intersectional bias heatmaps

## Key Findings

The analysis reveals several concerning patterns:

- **Gender Disparities**: Significant differences in approval rates between male and female applicants
- **Racial Bias**: Systematic disadvantages for certain racial groups
- **Intersectional Effects**: Compounded discrimination for individuals with multiple protected characteristics
- **Feature Impact**: Criminal record and credit score show disproportionate influence on certain groups

## Tools & Libraries Used

### Core Machine Learning
- **scikit-learn**: Model training, preprocessing, and evaluation
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations

### Fairness & Bias Detection
- **AIF360**: IBM's fairness toolkit for computing disparate impact
- **Fairlearn**: Microsoft's fairness assessment library for evaluating selection rate and false negative rate across sensitive attributes
- **SHAP**: Model explainability and feature importance

### Visualization & Analysis
- **matplotlib**: Static plotting and visualizations
- **seaborn**: Statistical data visualization
- **tabulate**: Formatted table outputs

### Data Processing
- **KNNImputer**: Missing value imputation
- **StandardScaler**: Feature normalization

## Project Structure

```
ai-bias-bounty-2025/
├── dataset/
│   ├── loan_access_dataset.csv
│   └── test.csv
├── hackathon_report.md
├── images/
│   ├── bias_visualization.png
│   ├── categorical_analysis.png
│   ├── continuous_analysis.png
│   ├── probability_distribution.png
│   └── shap_feature_importance.png
├── loan_model.ipynb
├── loan_model.py
├── README.md
├── requirements.txt
└── results/
    ├── fairness_analysis.csv
    ├── submission.csv
    ├── training_results.csv
    └── trained_model.pkl
```

## Usage

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sree-sphere/ai-bias-bounty-2025.git
   cd ai-bias-bounty-2025
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install 'aif360[AdversarialDebiasing]' 'aif360[inFairness]'
   ```

3. **Run the model**
    ```bash
    python loan_model.py
    ```
    ```Note: Might require to close the image files if opened automatically```

## Output Files Description

### Primary Deliverables

1. **submission.csv**  
   - Contains test‐set predictions in the competition format  
   - Columns:  
     - `ID` — identifier for each test record  
     - `LoanApproved` — model prediction (0 = Denied, 1 = Approved)

2. **model_training_results.csv**  
   - Detailed model outputs on the training set  
   - Columns include:  
     - True labels and predicted labels  
     - Predicted probabilities  
     - Values of the SHAP‑selected features

### Additional Analysis Files

- **results/fairness_analysis.json**  
  - Fairness metrics computed for each sensitive group (selection rate, false negative rate, disparate impact)

- **results/trained_model.pkl**  
  - Serialized `LogisticRegression` model and preprocessing pipeline components (`StandardScaler`, `KNNImputer`) for reproducibility

- **images/**  
  - A collection of visualizations capturing:
     - EDA for categorical and continuous features
     - Loan approval rate disparities by sensitive attributes
     - SHAP-based feature importance
     - Predicted probability distribution

---

## Interpreting Results

### Model Performance

- **Accuracy**: The final logistic regression model achieved an accuracy of **63.15%**, which reflects a moderately reliable classification of loan approvals.
- **Balanced Accuracy**: At **63.4%**, it suggests the model handles both classes reasonably well despite some class imbalance.
- **AUC‑ROC**: With an AUC of **0.6777**, the model shows fair discrimination capability between approved and denied loans.

### Fairness Metrics & Bias Insights

- **Disparate Impact**:
  - Applicants with disabilities faced a **disparate impact ratio of 2.04**, suggesting their approval rate is significantly lower than those without disabilities.
  - Individuals with a criminal record experienced the most extreme disparity, with a **disparate impact of 2.81**. This may reflect embedded policies but still indicates a significant access gap.

- **Univariate Approval Gaps**:
  - **Gender**: Males had noticeably higher predicted approval rates than females and non-binary applicants.
  - **Race**: Black and Hispanic applicants had the lowest predicted approval rates, raising fairness concerns.
  - **Citizenship**: Approval likelihood decreased consistently from Citizens to Visa Holders.
  - **Disability & Criminal Record**: Disabled applicants and those with a criminal record had much lower approval probabilities—disparities unlikely to be explained by feature differences alone.

- **Intersectional Analysis**:
  - Intersection of multiple attributes compounds bias. For example, **'No-Male-No'** had a predicted approval rate of **66%**, while **'Yes-Female-No'** dropped to **16%**—a stark contrast.
  - **Non-binary disabled or criminal-record-holding applicants** saw the lowest predicted rates, often below **27%**.

- **MetricFrame Disparities**:
  - **Gender**: Males had both higher selection rates (**0.59**) and lower false negative rates (**0.25**) compared to others.
  - **Criminal Record**: Those with a record saw a drastic drop in selection rate (**0.19**) and a spike in false negatives (**0.73**).
  - **Disability Status**: Similarly, those with disabilities had significantly lower selection rates and higher false negative rates.

### Key Behavioral Findings

- **Criminal record-related disparities are most strongly aligned with race**, not gender.
- **Loan amounts** across demographic groups did not differ significantly—indicating that denial gaps are not solely driven by financial asks.
- **Low-income applicants** (bottom 20%) had uniformly low approval rates (<30%), regardless of gender.
- The intersection of **Disability and Criminal Record** status represents the most marginalized subgroup—with just **24.5%** approval.

### Visualization Insights

- **Approval Rate Charts**: Highlight demographic-based disparities and help surface unbalanced model behavior.
- **SHAP Feature Importance**: Revealed `Credit Score`, `Income`, and `Gender_Male` as top predictors—making the model's decision-making traceable.
- **Prediction Probability Distributions**: Provided clarity into model confidence and decision thresholds, especially around marginalized groups.

These outputs together provide a transparent view of both **how well** the model predicts and **how fairly** it treats different segments of the population.

---

## References

- [Bellamy et al. (2018). AI Fairness 360: An extensible toolkit for detecting and mitigating algorithmic bias.](https://arxiv.org/abs/1810.01943)
- [Weerts et al. (2023). Fairlearn: Assessing and Improving Fairness of AI Systems](https://arxiv.org/abs/2303.16626)
- [Lundberg et al. (2017). A unified approach to interpreting model predictions.](https://arxiv.org/abs/1705.07874)
