# 📂 AI Bias Bounty Hackathon

Welcome to the **2025 AI Bias Bounty Hackathon**!

Join us as we explore, detect, and report biases in AI models and datasets. This is an exciting opportunity to contribute to ethical AI while building your skills and network.


# 🚀 Overview

The **AI Bias Bounty Hackathon** challenges participants to build machine learning models and generate technical reports that identifies bias within provided datasets. The goal is to encourage the development of fair and responsible AI systems.

**Devpost**: [AI Bias Bounty Hackathon on Devpost](https://ai-bias-bounty-hackathon.devpost.com/)

**Official Website**: [Hack the Fest](https://hackthefest.com/)


# 🗓️ Key Dates

| Schedule                        | Date                               |
| ------------------------------- | ---------------------------------- |
| Registration Period             | *\[June 4 – June 27, 2025]*        |
| Kickoff Event                   | *\[June 28, 2025]*                 |
| Onboarding Period               | *\[June 28 – June 30, 2025]*       |
| Hackathon Launch                | *\[July 1 – July 3, 2025]*   |
| Submission Deadline             | *\[July 3, 2025 11:59pm CST]*      |
| Judging Period                  | *\[July 5 – July 15, 2025]*        |
| Winners Announced               | *\[July 17, 2025]*                 |


# 🎯 Objectives

Participants will:

- Build AI models to analyze and detect bias in provided datasets.

- Generate detailed, well-structured technical reports documenting bias detection.

- Present solutions that contribute to fairness and accountability in AI.


# 🛠️ Getting Started

### 1. Register

- Sign up on our [official website](https://hackthefest.com/) or [Devpost](https://ai-bias-bounty-hackathon.devpost.com/) to officially enter the hackathon.

### 2. Dataset

- You will receive access to the dataset upon registration.

### 3. Deliverables

- `loan_model.py` — Python script containing
    
    - Data cleaning and preprocessing steps
    - Feature engineering (e.g., encoding, binning)
    - Model training (you may use Logistic Regression, Random Forest, XGBoost, or other classification models)
    - Fairness auditing and bias detection
    - Well-commented, readable code

- `submission.csv` — Model's output on the provided test dataset

    - A 2-column CSV:
      - `ID` – test set identifier
      - `LoanApproved` – predicted value (0 or 1)

- Detailed technical report (`ai_risk_report.docx` or `.pdf` or `.md`) using the provided AI Risk Report template

- Visual Evidence of Bias: Submit one or both of the following:

    - `bias_visualization.png` — a clear, readable graphic that illustrates discovered bias
    - Or a `chart/folder` containing:
      - Approval rate bar plots by demographic
      - SHAP/LIME feature importance charts
      - False positive/negative disparities
      - Any visual insights related to model behavior or group fairness
    
    Label every chart clearly. These visuals will help judges understand your biased insights at a glance.

- (Optional) `loan_model.ipynb` — Clean and reproducible Jupyter Notebook

    - Include EDA, model pipeline, audits, and final results
    - Clear markdown explanations and cell comments encouraged

- `README.md` — Describes:

    - The problem you addressed
    - Summary of your model approach and fairness considerations
    - Instructions to run the project & tools/libraries used
    - GitHub repo should be public and well-structured

# 📑 Submission Guidelines

1. Submissions is made on GitHub.

2. Include:

    - Source code

    - Output file

    - Technical report

    - Demo video (required)

3. Follow all provided instructions and deadlines.

4. After completing your GitHub repo, you must submit the repository link using the official [Final Submission Form](https://forms.gle/ES3CY59jEjdaqCvBA). This is how your entry is registered for judging.


# 🏆 Judging Criteria

Our judging panel is made up of industry leaders, data scientists, AI ethics professionals, and engineers across tech firms like Meta, Google, Amazon, Visa, JPMorgan, and Walmart, who will evaluate your work based on the following key areas. Each area reflects both the technical quality of your work and your ability to think critically about fairness, impact, and communication.

1. Bias Identification (30 points)
   We’re looking for how well you detected and explained patterns of bias in the dataset or model predictions.
Strong entries will show clear evidence of bias across multiple demographic groups (e.g., gender, race, income) and thoughtfully discuss false positives/negatives and their real-world implications.

2. Model Design & Justification (30 points)
   Your model doesn’t need to be perfect, but your choices should be intentional. This includes the algorithms you used, the features you engineered, and how you approached fairness. We value models that are interpretable and grounded in thoughtful design, not just performance.

3. Interpretability & Insights (20 points)
   Judges will be looking at how well you explain your results.
Use charts, plots, tools like SHAP/LIME, or group breakdowns to show what’s happening inside your model, especially when it behaves unfairly.

4. Presentation & Clarity (20 points)
   Clear communication is key. Your README, demo video, and any supporting materials should help others understand your work without confusion. The best submissions will be organized, polished, and easy to follow, even for non-technical reviewers.


# 📣 Stay Connected

Follow us for updates and highlights:

- **Website**: [Hack the Fest](https://hackthefest.com/)

- **Devpost**: [AI Bias Bounty Hackathon](https://ai-bias-bounty-hackathon.devpost.com/)


# 📚 Code of Conduct

We are committed to providing a welcoming and inclusive environment for all participants. Please review our [Code of Conduct](./CODE_OF_CONDUCT.md).


# 💬 Communication

Join our community via our [Slack Invite Link](https://join.slack.com/t/hackthefest/shared_invite/zt-380la7fd3-xk~zDvk~kZIrqr_HznLHbQ)
