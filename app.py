# =====================================================
# PART 1
# Employee Attrition Intelligence Platform
# Imports + Theme + Dashboard + Navigation
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime
import os

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Employee Attrition Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load("employee_attrition_model.pkl")

model = load_model()

# =====================================================
# CSV HISTORY FILE
# =====================================================

HISTORY_FILE = "prediction_history.csv"

if not os.path.exists(HISTORY_FILE):
    history_df = pd.DataFrame(
        columns=[
            "Timestamp",
            "Probability",
            "Risk_Level",
            "Prediction"
        ]
    )

    history_df.to_csv(
        HISTORY_FILE,
        index=False
    )

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    background-color: #F8FAFC;
}

.block-container {
    padding-top: 1rem;
}

.hero-card {
    background: linear-gradient(
        135deg,
        #1E3A8A,
        #0F766E
    );

    padding: 25px;
    border-radius: 18px;
    color: white;
    margin-bottom: 20px;
}

.metric-card {
    background: white;
    padding: 18px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    text-align:center;
}

.section-card {
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.06);
}

.low-risk {
    background:#DCFCE7;
    color:#166534;
    padding:15px;
    border-radius:12px;
}

.medium-risk {
    background:#FED7AA;
    color:#9A3412;
    padding:15px;
    border-radius:12px;
}

.high-risk {
    background:#FEE2E2;
    color:#991B1B;
    padding:15px;
    border-radius:12px;
}

.sidebar-title {
    font-size:22px;
    font-weight:bold;
}

hr {
    margin-top:10px;
    margin-bottom:10px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.image(
        "images/company_logo.png",
        width=180
    )

    st.markdown(
        "## Employee Attrition Intelligence"
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Executive Dashboard",
            "🔮 Employee Prediction",
            "📊 Model Insights",
            "📈 Workforce Analytics",
            "🤖 AI Recommendations",
            "📄 Report Center",
            "ℹ️ About Solution"
        ]
    )

    st.markdown("---")

    st.markdown("### Model Information")

    st.info(
        """
Algorithm: ExtraTreesClassifier

Balancing: SMOTE

ROC-AUC: 0.857

Accuracy: 79%

Version: 1.0
"""
    )

# =====================================================
# HERO HEADER
# =====================================================

st.markdown("""
<div class="hero-card">

<h1>
📊 Employee Attrition Intelligence Platform
</h1>

<h4>
Predict employee turnover risk using Machine Learning and HR Analytics
</h4>

<p>
✓ Machine Learning Powered
&nbsp;&nbsp;&nbsp;
✓ ROC-AUC 0.857
&nbsp;&nbsp;&nbsp;
✓ Accuracy 79%
&nbsp;&nbsp;&nbsp;
✓ Executive Dashboard
</p>

</div>
""", unsafe_allow_html=True)

# =====================================================
# KPI CARDS
# =====================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
    <h4>🎯 Accuracy</h4>
    <h2>79%</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
    <h4>📈 ROC-AUC</h4>
    <h2>0.857</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
    <h4>🧠 Features</h4>
    <h2>10</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
    <h4>⚡ Algorithm</h4>
    <h2>ExtraTrees</h2>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# =====================================================
# EXECUTIVE DASHBOARD PAGE
# =====================================================

if page == "🏠 Executive Dashboard":

    st.subheader(
        "Executive Summary"
    )

    col1, col2 = st.columns([2,1])

    with col1:

        st.markdown("""
        <div class="section-card">

        <h3>Business Overview</h3>

        Employee attrition can significantly impact
        productivity, recruitment costs, employee morale,
        and workforce stability.

        This platform uses Machine Learning to identify
        employees at risk of leaving and provides
        actionable HR recommendations.

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="section-card">

        <h3>Business Impact</h3>

        ✓ Reduce Turnover Costs

        ✓ Improve Retention

        ✓ Workforce Planning

        ✓ HR Decision Intelligence

        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.subheader(
        "Top Attrition Drivers"
    )

    driver1, driver2, driver3, driver4, driver5 = st.columns(5)

    driver1.metric(
        "Attrition Risk Score",
        "#1"
    )

    driver2.metric(
        "Job Satisfaction",
        "#2"
    )

    driver3.metric(
        "Work-Life Balance",
        "#3"
    )

    driver4.metric(
        "Stock Option",
        "#4"
    )

    driver5.metric(
        "Manager Rating",
        "#5"
    )

    st.write("")

    st.markdown("""
    ### Why These Factors Matter

    - Attrition Risk Score is the strongest predictor.
    - Job Satisfaction influences employee engagement.
    - Work-Life Balance affects retention and burnout.
    - Stock Options improve long-term commitment.
    - Manager Rating reflects leadership quality.

    """)
# =====================================================
# PART 2
# Prediction Engine + Explainable AI
# =====================================================

if page == "🔮 Employee Prediction":

    st.subheader("Employee Risk Assessment")

    col1, col2 = st.columns(2)

    # ==========================================
    # INPUT FORM
    # ==========================================

    with col1:

        st.markdown("### Risk Assessment")

        attrition_risk_score = st.slider(
            "Attrition Risk Score",
            0,
            100,
            46
        )

        st.markdown("### Employee Engagement")

        job_satisfaction = st.slider(
            "Job Satisfaction",
            1,
            5,
            3
        )

        work_life_balance = st.slider(
            "Work Life Balance",
            1,
            5,
            3
        )

        manager_rating = st.slider(
            "Manager Rating",
            1,
            5,
            3
        )

    with col2:

        st.markdown("### Compensation")

        monthly_income = st.number_input(
            "Monthly Income",
            min_value=1907,
            max_value=24412,
            value=8170
        )

        salary_hike = st.slider(
            "Salary Hike %",
            3.0,
            28.0,
            15.0
        )

        stock_option = st.selectbox(
            "Stock Option",
            [0, 1]
        )

        st.markdown("### Performance")

        performance_rating = st.slider(
            "Performance Rating",
            1,
            5,
            3
        )

        training_hours = st.slider(
            "Training Hours Last Year",
            0,
            100,
            35
        )

        st.markdown("### Workplace Factors")

        commute_time = st.slider(
            "Commute Time (Minutes)",
            5,
            150,
            40
        )

    st.write("")

    # ==========================================
    # PREDICT BUTTON
    # ==========================================

    if st.button(
        "🔮 Predict Attrition Risk",
        use_container_width=True
    ):

        with st.spinner(
            "Analyzing employee profile..."
        ):

            input_data = pd.DataFrame(
                [[
                    attrition_risk_score,
                    job_satisfaction,
                    work_life_balance,
                    stock_option,
                    manager_rating,
                    commute_time,
                    monthly_income,
                    salary_hike,
                    performance_rating,
                    training_hours
                ]],
                columns=[
                    "Attrition_Risk_Score",
                    "Job_Satisfaction",
                    "Work_Life_Balance",
                    "Stock_Option",
                    "Manager_Rating",
                    "Commute_Time_Minutes",
                    "Monthly_Income",
                    "Salary_Hike_Percent",
                    "Performance_Rating",
                    "Training_Hours_Last_Year"
                ]
            )

            prediction = model.predict(
                input_data
            )[0]

            probability = model.predict_proba(
                input_data
            )[0][1]

        probability_percent = round(
            probability * 100,
            2
        )

        confidence_score = round(
            max(
                probability,
                1 - probability
            ) * 100,
            2
        )

        # ======================================
        # RISK LEVEL
        # ======================================

        if probability_percent < 40:

            risk_level = "Low Risk"

            card_class = "low-risk"

            recommendation = (
                "Maintain engagement and retention strategy."
            )

        elif probability_percent < 70:

            risk_level = "Medium Risk"

            card_class = "medium-risk"

            recommendation = (
                "Monitor employee satisfaction and career development."
            )

        else:

            risk_level = "High Risk"

            card_class = "high-risk"

            recommendation = (
                "Immediate retention intervention recommended."
            )

        # ======================================
        # SAVE HISTORY
        # ======================================

        history = pd.read_csv(
            HISTORY_FILE
        )

        new_row = pd.DataFrame(
            [{
                "Timestamp":
                datetime.now(),

                "Probability":
                probability_percent,

                "Risk_Level":
                risk_level,

                "Prediction":
                prediction
            }]
        )

        history = pd.concat(
            [history, new_row],
            ignore_index=True
        )

        history.to_csv(
            HISTORY_FILE,
            index=False
        )

        # ======================================
        # RESULTS
        # ======================================

        st.markdown(
            f"""
            <div class="{card_class}">
            <h2>{risk_level}</h2>
            <h3>{probability_percent}% Probability</h3>
            <p>{recommendation}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ======================================
        # GAUGE CHART
        # ======================================

        st.subheader(
            "Attrition Risk Meter"
        )

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability_percent,

                title={
                    "text":
                    "Attrition Risk %"
                },

                gauge={
                    "axis":
                    {
                        "range":[0,100]
                    },

                    "bar":
                    {
                        "color":"#1E3A8A"
                    },

                    "steps":[
                        {
                            "range":[0,40],
                            "color":"#16A34A"
                        },
                        {
                            "range":[40,70],
                            "color":"#EA580C"
                        },
                        {
                            "range":[70,100],
                            "color":"#DC2626"
                        }
                    ]
                }
            )
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

        # ======================================
        # CONFIDENCE SCORE
        # ======================================

        st.subheader(
            "Prediction Confidence"
        )

        st.progress(
            confidence_score / 100
        )

        st.success(
            f"Confidence Score: {confidence_score}%"
        )

        # ======================================
        # EXECUTIVE ALERTS
        # ======================================

        if probability_percent >= 80:

            st.error(
                """
                🚨 Critical Alert

                Employee exceeds
                80% attrition probability.

                Immediate HR review recommended.
                """
            )

        # ======================================
        # EXPLAINABLE AI
        # ======================================

        st.subheader(
            "Explainable AI"
        )

        drivers = []

        if attrition_risk_score > 60:
            drivers.append(
                "High Attrition Risk Score"
            )

        if job_satisfaction < 3:
            drivers.append(
                "Low Job Satisfaction"
            )

        if work_life_balance < 3:
            drivers.append(
                "Poor Work-Life Balance"
            )

        if manager_rating < 3:
            drivers.append(
                "Low Manager Rating"
            )

        if salary_hike < 10:
            drivers.append(
                "Limited Salary Growth"
            )

        if commute_time > 60:
            drivers.append(
                "Long Commute Time"
            )

        if len(drivers) == 0:
            drivers.append(
                "No significant risk drivers detected."
            )

        for item in drivers:

            st.info(
                f"✓ {item}"
            )

        # ======================================
        # HR PLAYBOOK
        # ======================================

        st.subheader(
            "HR Action Playbook"
        )

        if risk_level == "High Risk":

            st.warning("""
1. Conduct retention interview

2. Review compensation package

3. Assess workload balance

4. Discuss career progression

5. Improve manager communication
""")

        elif risk_level == "Medium Risk":

            st.info("""
1. Monitor engagement

2. Review development opportunities

3. Improve recognition

4. Schedule feedback discussion
""")

        else:

            st.success("""
1. Maintain engagement

2. Continue recognition programs

3. Monitor periodically
""")

    # ==========================================
    # WHAT IF ANALYSIS
    # ==========================================

    st.write("")
    st.write("")

    st.subheader(
        "What-If Simulator"
    )

    st.info(
        """
Adjust key variables and
evaluate potential impact
on attrition risk.
"""
    )

    st.write(
        "Future version can perform real-time scenario comparison and workforce planning analysis."
    )
# =====================================================
# PART 3
# Insights + Reports + Analytics + About
# =====================================================

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

# =====================================================
# MODEL INSIGHTS PAGE
# =====================================================

if page == "📊 Model Insights":

    st.subheader("Model Insights")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Feature Importance")

        st.image(
            "images/feature_importance.png",
            use_container_width=True
        )

        st.info(
            """
Attrition Risk Score is the most influential feature.

Other important factors include:

• Job Satisfaction

• Work Life Balance

• Stock Option

• Manager Rating

• Monthly Income
"""
        )

    with col2:

        st.markdown("### ROC Curve")

        st.image(
            "images/roc_curve.png",
            use_container_width=True
        )

        st.success(
            """
ROC-AUC Score = 0.857

This indicates strong discrimination capability between employees likely to stay and employees likely to leave.
"""
        )

    st.write("")

    st.subheader("Future Explainability")

    st.info(
        """
SHAP integration can be added in future releases for advanced local and global model explanations.
"""
    )

# =====================================================
# WORKFORCE ANALYTICS
# =====================================================

if page == "📈 Workforce Analytics":

    st.subheader("Workforce Analytics")

    history = pd.read_csv(
        HISTORY_FILE
    )

    total_predictions = len(history)

    if total_predictions > 0:

        low_count = len(
            history[
                history["Risk_Level"] == "Low Risk"
            ]
        )

        medium_count = len(
            history[
                history["Risk_Level"] == "Medium Risk"
            ]
        )

        high_count = len(
            history[
                history["Risk_Level"] == "High Risk"
            ]
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Low Risk",
            low_count
        )

        col2.metric(
            "Medium Risk",
            medium_count
        )

        col3.metric(
            "High Risk",
            high_count
        )

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=[
                        "Low Risk",
                        "Medium Risk",
                        "High Risk"
                    ],

                    values=[
                        low_count,
                        medium_count,
                        high_count
                    ]
                )
            ]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.info(
            "No prediction history available."
        )

    st.write("")

    st.subheader(
        "Top Attrition Drivers"
    )

    st.markdown("""
### Key Drivers

1. Attrition Risk Score

2. Job Satisfaction

3. Work Life Balance

4. Stock Option

5. Manager Rating

These variables contribute most significantly to attrition prediction outcomes.
""")

# =====================================================
# AI RECOMMENDATIONS
# =====================================================

if page == "🤖 AI Recommendations":

    st.subheader(
        "AI-Powered HR Recommendations"
    )

    st.success("""
### Employee Retention Strategies

• Conduct periodic engagement surveys

• Improve work-life balance initiatives

• Strengthen manager-employee communication

• Enhance career development opportunities

• Review compensation and rewards

• Increase recognition programs

• Reduce employee burnout risks

• Promote internal mobility
""")

    st.warning("""
### High-Risk Employee Actions

1. Retention Interview

2. Compensation Review

3. Career Progression Discussion

4. Leadership Coaching

5. Workload Assessment
""")

# =====================================================
# PDF REPORT CENTER
# =====================================================

if page == "📄 Report Center":

    st.subheader(
        "Executive Report Center"
    )

    st.write(
        "Generate downloadable workforce reports."
    )

    def create_pdf():

        buffer = BytesIO()

        doc = SimpleDocTemplate(
            buffer
        )

        styles = getSampleStyleSheet()

        content = []

        content.append(
            Paragraph(
                "Employee Attrition Intelligence Report",
                styles["Title"]
            )
        )

        content.append(
            Spacer(
                1,
                20
            )
        )

        content.append(
            Paragraph(
                """
This report summarizes employee attrition analytics generated by the Employee Attrition Intelligence Platform.
""",
                styles["BodyText"]
            )
        )

        content.append(
            Spacer(
                1,
                20
            )
        )

        content.append(
            Paragraph(
                "Model: ExtraTreesClassifier",
                styles["BodyText"]
            )
        )

        content.append(
            Paragraph(
                "Accuracy: 79%",
                styles["BodyText"]
            )
        )

        content.append(
            Paragraph(
                "ROC-AUC: 0.857",
                styles["BodyText"]
            )
        )

        doc.build(
            content
        )

        buffer.seek(0)

        return buffer

    pdf_file = create_pdf()

    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_file,
        file_name="employee_attrition_report.pdf",
        mime="application/pdf"
    )

# =====================================================
# PREDICTION HISTORY
# =====================================================

st.write("")
st.write("")

st.subheader(
    "Prediction History"
)

history = pd.read_csv(
    HISTORY_FILE
)

if len(history) > 0:

    st.dataframe(
        history,
        use_container_width=True
    )

    csv_data = history.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇ Download History CSV",
        csv_data,
        "prediction_history.csv",
        "text/csv"
    )

    if st.button(
        "🗑 Clear History"
    ):

        empty_df = pd.DataFrame(
            columns=[
                "Timestamp",
                "Probability",
                "Risk_Level",
                "Prediction"
            ]
        )

        empty_df.to_csv(
            HISTORY_FILE,
            index=False
        )

        st.success(
            "History cleared successfully."
        )

else:

    st.info(
        "No history available."
    )

# =====================================================
# ABOUT PAGE
# =====================================================

if page == "ℹ️ About Solution":

    st.subheader(
        "About Employee Attrition Intelligence Platform"
    )

    st.markdown("""
### Project Overview

This platform predicts employee attrition risk using Machine Learning and provides actionable HR recommendations.

### Dataset

IBM HR Analytics Employee Attrition Dataset

### Machine Learning Pipeline

• Data Cleaning

• Label Encoding

• Feature Selection

• SMOTE Balancing

• ExtraTreesClassifier

### Performance

• Accuracy: 79%

• ROC-AUC: 0.857

### Technology Stack

• Python

• Pandas

• NumPy

• Scikit-Learn

• Streamlit

• Plotly

• ReportLab

• Joblib

### Business Value

• Workforce Intelligence

• Employee Retention

• Turnover Cost Reduction

• HR Decision Support
""")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    """
<center>

Employee Attrition Intelligence Platform

Machine Learning | HR Analytics | Workforce Intelligence

Version 1.0

</center>
""",
    unsafe_allow_html=True
)