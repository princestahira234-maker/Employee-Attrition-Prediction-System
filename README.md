# Employee Attrition Intelligence Platform

![Logo](assets/company_logo.png)

## AI-Powered Workforce Analytics & Retention Intelligence System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![HR Analytics](https://img.shields.io/badge/HR-Analytics-teal)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.857-success)

---

# Executive Summary

Employee Attrition Intelligence Platform is a Machine Learning-powered HR Analytics solution designed to identify employees at risk of leaving an organization before attrition occurs.

The platform combines Predictive Analytics, Workforce Intelligence, and Explainable AI principles to help HR leaders improve employee retention, reduce turnover costs, and support data-driven workforce planning.

Built using an Extra Trees Classifier and deployed through Streamlit, the solution generates probability-based attrition predictions and actionable workforce insights through an interactive dashboard experience.

---

# Business Problem

Employee turnover represents a major challenge for organizations due to:

* Recruitment and onboarding costs
* Productivity losses
* Workforce instability
* Knowledge drain
* Reduced employee morale
* Increased HR operational burden

Organizations often struggle to identify high-risk employees before resignation occurs.

This platform addresses that challenge by enabling proactive intervention through predictive workforce analytics.

---

# Business Impact

The solution helps organizations:

✅ Predict employee attrition risk

✅ Improve employee retention planning

✅ Reduce turnover-related costs

✅ Support data-driven HR decisions

✅ Identify high-risk employees early

✅ Improve workforce stability

✅ Enable proactive talent management

✅ Strengthen workforce planning initiatives

---

# Platform Screenshots

## Dashboard Overview

![Dashboard](assets/dashboard.png)

---

## Prediction Dashboard

![Prediction Dashboard](assets/prediction.png)

---

## Feature Importance Analysis

![Feature Importance](assets/feature_importance.png)

---

## ROC Curve

![ROC Curve](assets/roc_curve.png)

---

## Confusion Matrix

![Confusion Matrix](assets/confusion_matrix.png)

---

# Project Highlights

| Metric            | Value                         |
| ----------------- | ----------------------------- |
| Dataset Size      | 15,000 Employees              |
| Algorithm         | Extra Trees Classifier        |
| Selected Features | 10                            |
| Class Balancing   | SMOTE                         |
| Accuracy          | 79%                           |
| ROC-AUC           | 0.857                         |
| Deployment        | Streamlit                     |
| Prediction Type   | Binary Classification         |
| Use Case          | Employee Attrition Prediction |

---

# Platform Capabilities

| Capability           | Description                        |
| -------------------- | ---------------------------------- |
| Attrition Prediction | Predict employee turnover risk     |
| Probability Scoring  | Generate attrition probability     |
| Workforce Analytics  | HR-focused employee insights       |
| Explainable AI       | Feature-based prediction reasoning |
| Risk Classification  | Low, Medium, High Risk Categories  |
| What-If Analysis     | Simulate employee scenarios        |
| Executive Alerts     | Identify high-risk employees       |
| HR Playbooks         | Retention recommendations          |
| Prediction History   | Store prediction records           |
| PDF Reporting        | Export executive reports           |
| Streamlit Dashboard  | Interactive web application        |

---

# Dataset Information

### Dataset

IBM HR Analytics Employee Attrition Dataset (Enhanced Version)

### Target Variable

| Value | Meaning         |
| ----- | --------------- |
| 0     | Employee Stays  |
| 1     | Employee Leaves |

### Dataset Contains

* Employee Demographics
* Compensation Information
* Job Satisfaction
* Work-Life Balance Metrics
* Performance Ratings
* Career Development Indicators
* Training Records
* Workplace Factors
* Commute Information
* Employee Engagement Metrics

---

# Machine Learning Pipeline

```text
Employee Dataset
       │
       ▼
Data Cleaning
       │
       ▼
Categorical Encoding
       │
       ▼
Feature Engineering
       │
       ▼
Feature Selection
       │
       ▼
Train-Test Split
       │
       ▼
SMOTE Balancing
       │
       ▼
Extra Trees Classifier
       │
       ▼
Model Evaluation
       │
       ▼
Probability Prediction
       │
       ▼
Streamlit Dashboard
       │
       ▼
HR Decision Support
```

---

# Selected Features

The final model was trained using the top-performing features identified through feature importance analysis.

1. Attrition_Risk_Score
2. Job_Satisfaction
3. Work_Life_Balance
4. Stock_Option
5. Manager_Rating
6. Commute_Time_Minutes
7. Monthly_Income
8. Salary_Hike_Percent
9. Performance_Rating
10. Training_Hours_Last_Year

---

# Machine Learning Model

### Algorithm

ExtraTreesClassifier

### Configuration

```python
ExtraTreesClassifier(
    n_estimators=500,
    class_weight="balanced",
    random_state=42
)
```

---

# Class Imbalance Handling

### SMOTE

The dataset was balanced using:

**SMOTE (Synthetic Minority Oversampling Technique)**

Benefits:

* Better minority class detection
* Improved recall performance
* Reduced prediction bias
* Enhanced employee risk identification

---

# Model Performance

| Metric                     | Score |
| -------------------------- | ----- |
| Accuracy                   | 79%   |
| ROC-AUC Score              | 0.857 |
| Recall (Attrition Class)   | 69%   |
| F1 Score (Attrition Class) | 68%   |

### Performance Summary

The model demonstrates:

* Strong predictive capability
* Good class discrimination
* Balanced classification performance
* Effective attrition risk detection
* Reliable probability estimation

The ROC-AUC score of **0.857** indicates excellent capability to distinguish employees likely to leave from employees likely to stay.

---

# Feature Importance Analysis

The model identified the following factors as the strongest predictors of attrition:

* Attrition Risk Score
* Job Satisfaction
* Work-Life Balance
* Stock Option Availability
* Manager Rating
* Commute Time
* Monthly Income
* Salary Hike Percentage
* Performance Rating
* Training Hours

### Visualization

![Feature Importance](assets/feature_importance.png)

---

# ROC Curve Analysis

ROC-AUC Score: **0.857**

### Visualization

![ROC Curve](assets/roc_curve.png)

The ROC curve demonstrates strong classification performance and separation capability across prediction thresholds.

---

# Model Governance

The model was developed following industry-standard machine learning practices:

* Stratified Train-Test Split
* Feature Importance Validation
* SMOTE Class Balancing
* ROC-AUC Evaluation
* Reproducible Pipeline Design
* Controlled Random State Configuration

---

# Model Compression

The trained model was compressed using Joblib serialization before deployment.

Compression was performed solely to reduce storage requirements and repository size.

The compression process does not impact:

* Model Accuracy
* Prediction Results
* ROC-AUC Performance
* Probability Scores
* Business Outcomes

The deployed model generates identical predictions to the original trained model.

---

# Streamlit Application

The application provides a modern HR Analytics interface for business users and HR teams.

Users can:

* Enter employee information
* Generate attrition predictions
* View probability scores
* Assess retention risk
* Analyze workforce insights
* Download executive reports
* Track prediction history
* Support HR decision-making

---

# Project Structure

```text
employee-attrition-intelligence-platform/
│
├── app.py
├── employee_attrition_model.pkl
├── requirements.txt
├── prediction_history.csv
├── README.md
├── .gitignore
│
├── assets/
│   ├── company_logo.png
│   ├── dashboard.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── prediction.png
│   └── roc_curve.png
```

---

# Installation

Clone repository:

```bash
git clone https://github.com/yourusername/employee-attrition-intelligence-platform.git
```

Navigate into project:

```bash
cd employee-attrition-intelligence-platform
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run application:

```bash
streamlit run app.py
```

---

# Deployment Options

The platform can be deployed on:

* Streamlit Community Cloud
* Render
* Railway
* AWS
* Microsoft Azure
* Google Cloud Platform

---

# Data Privacy & Ethical Use

This solution is designed to support HR decision-making and should not be used as the sole basis for employment-related decisions.

Organizations should ensure compliance with internal HR policies and applicable data privacy regulations when deploying predictive workforce analytics solutions.

---

# Model Limitations

* Predictions depend on data quality
* Performance may vary across organizations
* External economic conditions are not directly modeled
* Workforce behavior may evolve over time
* Periodic retraining is recommended

---

# Future Enhancements

* SHAP Explainability
* Advanced Workforce Segmentation
* Hyperparameter Optimization
* XGBoost Benchmarking
* REST API Deployment
* Automated Model Retraining
* Real-Time Monitoring Dashboard
* Cloud-Native Architecture
* Enterprise Authentication
* Multi-Department Workforce Analytics

---

# Conclusion

Employee Attrition Intelligence Platform demonstrates how Machine Learning can be effectively applied to workforce analytics and employee retention challenges.

By identifying employees at risk of attrition before turnover occurs, organizations can implement targeted retention strategies, reduce operational costs, and make informed HR decisions through predictive analytics.

The platform achieved a ROC-AUC score of **0.857** and delivers reliable employee attrition predictions through a modern Streamlit-based HR Intelligence Dashboard.

---

## Author

### Employee Attrition Intelligence Platform

Machine Learning • HR Analytics • Workforce Intelligence • Predictive Analytics • Streamlit Deployment
