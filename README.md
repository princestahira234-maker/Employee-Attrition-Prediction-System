# 👥 Employee Attrition Intelligence Platform

> **AI-powered workforce analytics and employee retention intelligence system built with Machine Learning and Streamlit.**

![Logo](assets/company_logo.png)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![HR Analytics](https://img.shields.io/badge/HR-Analytics-teal)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.857-success)

---

## 🚀 Project Overview

Employee Attrition Intelligence Platform is an end-to-end **Machine Learning HR Analytics application** designed to assess employee attrition risk using workforce, compensation, satisfaction, performance, and workplace-related factors.

The platform combines:

**Data Processing → Feature Selection → SMOTE → Extra Trees → Risk Prediction → Explainability → Streamlit Dashboard**

The trained model generates **attrition predictions and probability scores**, while the interactive Streamlit application provides a practical interface for exploring employee risk and workforce insights.

---

## 📊 Project Highlights

| Metric               |                      Value |
| -------------------- | -------------------------: |
| 👥 Dataset Size      |       **15,000 Employees** |
| 🤖 Model             | **Extra Trees Classifier** |
| 🧩 Selected Features |                     **10** |
| ⚖️ Class Balancing   |                  **SMOTE** |
| 🎯 Task              |  **Binary Classification** |
| 📈 ROC-AUC           |                  **0.857** |
| ✅ Accuracy           |                    **79%** |
| 🔍 Attrition Recall  |                    **69%** |
| 🚀 Deployment        |              **Streamlit** |

---

## 🌐 Interactive Application

The trained Machine Learning model is integrated into an interactive Streamlit HR Intelligence Dashboard.

### What the application provides

* 👤 Employee information input
* 🎯 Attrition prediction
* 📊 Probability scoring
* 🚦 Risk classification
* 🔎 Workforce insights
* 🔄 What-if analysis
* 📋 Prediction history
* 📄 Executive report generation
* 💼 HR decision-support functionality

---

# 💼 Business Problem

Employee turnover can create significant operational challenges for organizations, including:

* Recruitment and onboarding costs
* Productivity losses
* Workforce instability
* Knowledge and experience loss
* Employee morale challenges
* Increased HR workload

Traditional HR processes may identify employee turnover risks only after warning signs become visible.

This project demonstrates a **proactive predictive analytics approach**, using available employee information to identify patterns associated with attrition risk.

---

# 🎯 Business Value

The platform is designed to support organizations in:

* Identifying employees with elevated attrition risk
* Supporting proactive retention planning
* Analyzing workforce-related risk factors
* Supporting data-driven HR decisions
* Understanding factors associated with employee attrition
* Improving workforce planning workflows

> **Important:** Model predictions should be treated as decision-support information rather than as the sole basis for employment-related decisions.

---

# 🖥️ Platform Screenshots

## 📊 Dashboard Overview

![Dashboard](assets/dashboard.png)

The main dashboard provides an overview of the HR intelligence platform and its available analytics capabilities.

---

## 🎯 Prediction Dashboard

![Prediction Dashboard](assets/prediction.png)

The prediction interface allows users to enter employee information and obtain an attrition prediction with probability-based risk information.

---

## 🔎 Feature Importance Analysis

![Feature Importance](assets/feature_importance.png)

Feature importance analysis provides insight into the variables contributing most strongly to the model's predictions.

---

## 📈 ROC Curve

![ROC Curve](assets/roc_curve.png)

The ROC curve illustrates the classifier's discrimination performance across different classification thresholds.

---

## 🧮 Confusion Matrix

![Confusion Matrix](assets/confusion_matrix.png)

The confusion matrix provides a detailed view of classification outcomes across the two target classes.

---

# ✨ Platform Capabilities

| Capability              | Description                               |
| ----------------------- | ----------------------------------------- |
| 🎯 Attrition Prediction | Predict employee turnover risk            |
| 📊 Probability Scoring  | Generate probability-based risk estimates |
| 🚦 Risk Classification  | Categorize employee risk levels           |
| 🔎 Feature Analysis     | Examine important predictive factors      |
| 🧠 Explainable Insights | Provide feature-based prediction context  |
| 🔄 What-If Analysis     | Explore hypothetical employee scenarios   |
| 🚨 Executive Alerts     | Highlight higher-risk cases               |
| 📋 HR Playbooks         | Support retention planning                |
| 🕒 Prediction History   | Store prediction records                  |
| 📄 PDF Reporting        | Generate executive reports                |
| 🖥️ Streamlit Dashboard | Interactive web-based interface           |

---

# 🗂️ Dataset

## Dataset Information

**IBM HR Analytics Employee Attrition Dataset (Enhanced Version)**

The dataset contains **15,000 employee records** and **32 columns** covering demographic, employment, compensation, satisfaction, performance, workplace, and attrition-related information.

### Target Variable

| Value | Meaning         |
| ----: | --------------- |
|   `0` | Employee Stays  |
|   `1` | Employee Leaves |

### Dataset Categories

* 👤 Employee demographics
* 🎓 Education information
* 🏢 Department and job role
* 💼 Employment information
* 💰 Compensation
* ⭐ Job satisfaction
* ⚖️ Work-life balance
* 📈 Performance
* 🎓 Training
* 🚗 Commute information
* 🏠 Remote-work information
* ✈️ Business travel
* 📦 Stock options
* 📊 Attrition risk information

---

# 🧩 Selected Features

The final model uses the following **10 selected features** identified through feature importance analysis:

|  # | Feature                    |
| -: | -------------------------- |
|  1 | `Attrition_Risk_Score`     |
|  2 | `Job_Satisfaction`         |
|  3 | `Work_Life_Balance`        |
|  4 | `Stock_Option`             |
|  5 | `Manager_Rating`           |
|  6 | `Commute_Time_Minutes`     |
|  7 | `Monthly_Income`           |
|  8 | `Salary_Hike_Percent`      |
|  9 | `Performance_Rating`       |
| 10 | `Training_Hours_Last_Year` |

---

# ⚙️ Machine Learning Pipeline

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
SMOTE Class Balancing
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

# 🤖 Machine Learning Model

## Algorithm

**ExtraTreesClassifier**

Extra Trees is an ensemble tree-based classification algorithm used here to model relationships between employee characteristics and attrition outcomes.

### Model Configuration

```python
ExtraTreesClassifier(
    n_estimators=500,
    class_weight="balanced",
    random_state=42
)
```

### Learning Task

* **Learning Type:** Supervised Learning
* **Task:** Binary Classification
* **Target:** `Attrition`
* **Model:** Extra Trees Classifier
* **Output:** Attrition class + probability score

---

# ⚖️ Class Imbalance Handling

## SMOTE

The training data was balanced using **SMOTE (Synthetic Minority Oversampling Technique)**.

SMOTE was incorporated to improve representation of the minority class during model training.

### Objective

* Improve minority-class detection
* Support better recall
* Reduce class imbalance effects
* Improve attrition-risk identification

---

# 📈 Model Performance

| Metric                     |     Score |
| -------------------------- | --------: |
| Accuracy                   |   **79%** |
| ROC-AUC                    | **0.857** |
| Recall — Attrition Class   |   **69%** |
| F1 Score — Attrition Class |   **68%** |

### Performance Interpretation

The model achieved a **ROC-AUC of 0.857** on the evaluated data, indicating good class-discrimination performance across classification thresholds.

The model also achieved:

* **79% accuracy**
* **69% recall** for the attrition class
* **68% F1 score** for the attrition class

These metrics provide multiple perspectives on classification performance rather than relying on accuracy alone.

---

# 🔎 Feature Importance Analysis

Feature importance analysis was used to identify the variables contributing most strongly to the model's predictions.

### Key Predictive Features

* `Attrition_Risk_Score`
* `Job_Satisfaction`
* `Work_Life_Balance`
* `Stock_Option`
* `Manager_Rating`
* `Commute_Time_Minutes`
* `Monthly_Income`
* `Salary_Hike_Percent`
* `Performance_Rating`
* `Training_Hours_Last_Year`

### Visualization

![Feature Importance](assets/feature_importance.png)

---

# 📈 ROC Curve Analysis

### ROC-AUC: **0.857**

![ROC Curve](assets/roc_curve.png)

The ROC curve visualizes classifier performance across different probability thresholds and complements the reported ROC-AUC metric.

---

# 🧮 Confusion Matrix

![Confusion Matrix](assets/confusion_matrix.png)

The confusion matrix provides a class-by-class view of the model's predictions, including correctly and incorrectly classified employees.

---

# 🧠 Model Governance

The project incorporates several reproducibility and evaluation practices:

* Stratified train-test splitting
* Feature importance analysis
* SMOTE-based class balancing
* ROC-AUC evaluation
* Controlled random state
* Reproducible model configuration
* Separate model evaluation metrics

These practices help make the training and evaluation workflow more structured and reproducible.

---

# 📦 Model Serialization

The trained model is stored using **Joblib serialization** for deployment.

The serialized model is loaded by the Streamlit application during inference, allowing the deployed application to use the trained classifier without retraining.

---

# 🚀 Streamlit Application

The Streamlit application turns the trained model into an interactive HR analytics tool.

### Application Workflow

```text
Enter Employee Information
          ↓
Validate Input
          ↓
Load Trained Model
          ↓
Generate Prediction
          ↓
Calculate Probability
          ↓
Display Risk Information
          ↓
Generate Insights / Reports
          ↓
Store Prediction History
```

### Application Features

* 👤 Employee data input
* 🎯 Attrition prediction
* 📊 Probability scores
* 🚦 Risk classification
* 🔎 Workforce insights
* 🔄 What-if analysis
* 📋 Prediction history
* 📄 PDF executive reports

---

# 📁 Project Structure

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
└── assets/
    ├── company_logo.png
    ├── dashboard.png
    ├── confusion_matrix.png
    ├── feature_importance.png
    ├── prediction.png
    └── roc_curve.png
```

---

# 💻 Installation & Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/princestahira234-maker/Employee-Attrition-Prediction-System.git
```

### 2. Navigate to the Project

```bash
cd Employee-Attrition-Prediction-System
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

---

# ☁️ Deployment

The application is designed for Streamlit deployment and can also be adapted for cloud platforms such as:

* Streamlit Community Cloud
* Render
* Railway
* AWS
* Microsoft Azure
* Google Cloud Platform

---

# 🔐 Data Privacy & Ethical Use

Employee analytics involves sensitive organizational information. This system should therefore be used as **decision-support technology**, not as an automated employment decision-maker.

Organizations should:

* Protect employee information
* Apply appropriate access controls
* Follow applicable privacy requirements
* Review model outputs with qualified HR professionals
* Avoid using predictions as the sole basis for employment decisions
* Monitor model performance after deployment

---

# ⚠️ Model Limitations

The model has several practical limitations:

* Predictions depend on the quality and representativeness of the training data.
* Model performance may differ across organizations and workforce populations.
* External economic and organizational factors are not directly modeled.
* Employee behavior and workforce conditions can change over time.
* Periodic evaluation and retraining may be required for continued performance.

---

# 🔮 Future Enhancements

Potential future improvements include:

* 🧠 SHAP-based explainability
* 👥 Advanced workforce segmentation
* ⚙️ Hyperparameter optimization
* 🔬 XGBoost model benchmarking
* 🌐 REST API deployment
* 🔄 Automated model retraining
* 📡 Real-time model monitoring
* ☁️ Cloud-native architecture
* 🔐 Enterprise authentication
* 🏢 Multi-department workforce analytics

---

# 🏁 Conclusion

Employee Attrition Intelligence Platform demonstrates an end-to-end application of **Machine Learning to workforce analytics and employee attrition prediction**.

The project combines:

**15,000 employee records → feature selection → SMOTE → Extra Trees Classification → model evaluation → probability scoring → interactive Streamlit deployment**

The final model achieved a **ROC-AUC of 0.857**, with **79% accuracy, 69% attrition-class recall, and 68% attrition-class F1 score** on the evaluated data.

Beyond model training, the project demonstrates practical implementation across:

* Data preparation
* Feature selection
* Class imbalance handling
* Classification
* Model evaluation
* Feature importance analysis
* Probability-based prediction
* Interactive dashboard development
* Reporting
* Deployment

This makes the project a practical demonstration of building a **complete, user-facing Machine Learning solution rather than a model-only experiment**.

---

## 👩‍💻 Author

### Employee Attrition Intelligence Platform

**Machine Learning • HR Analytics • Workforce Intelligence • Predictive Analytics • Streamlit Deployment**
