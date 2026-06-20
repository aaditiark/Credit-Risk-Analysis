import streamlit as st
st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

h1, h2, h3 {
    color: #1f4e79;
}

/* 🔥 FIXED METRIC CARD */
[data-testid="metric-container"] {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 2px 2px 12px rgba(0,0,0,0.1);
}

/* 🔥 VALUE (big number) */
[data-testid="metric-container"] > div {
    color: #000000 !important;
    font-weight: bold;
}

/* 🔥 LABEL (small text) */
[data-testid="metric-container"] label {
    color: #555555 !important;
}
</style>
""", unsafe_allow_html=True)
import numpy as np
import pandas as pd
import os
import joblib
import plotly.graph_objects as go
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image
from reportlab.lib.units import inch
import tempfile
import joblib

#model = joblib.load(model_path)
#scaler = joblib.load("scaler.pkl")

# 🔥 ADD HERE (exactly here)
from io import BytesIO   # 🔥 ADD THIS IMPORT
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors

def generate_pdf(probability, decision, input_data, feature_names, top_features, shap_fig, gauge_fig):

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = []

    # -------------------------------
    # 🏦 HEADER (BANK STYLE)
    # -------------------------------
    content.append(Paragraph("<b>NovaTrust Bank - Credit Risk Department</b>", styles['Title']))
    content.append(Spacer(1, 8))

    content.append(Paragraph("Credit Risk Assessment Report", styles['Heading2']))
    content.append(Spacer(1, 5))
    
    #After title
    content.append(Paragraph("Confidential Credit Evaluation Document", styles['Italic']))
    content.append(Spacer(1, 5))


    # -------------------------------
    # CUSTOMER TABLE
    # -------------------------------
    table_data = [["Feature", "Value"]]

    for i, name in enumerate(feature_names):
        table_data.append([name, str(input_data[0][i])])

    table = Table(table_data, colWidths=[200, 200])

    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),

        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('BACKGROUND',(0,1),(-1,-1),colors.beige),
    ]))

    content.append(Paragraph("<b>1. Customer Profile</b>", styles['Heading3']))
    content.append(Spacer(1, 10))
    content.append(table)
    content.append(Spacer(1, 20))

    # -------------------------------
    # RISK RESULT
    # -------------------------------
    content.append(Paragraph("<b>2. Risk Evaluation</b>", styles['Heading3']))
    content.append(Spacer(1, 10))
    
    #New 
    content.append(Paragraph("<b>Executive Summary</b>", styles['Heading3']))
    content.append(Spacer(1, 5))

    summary_text = f"""
    The applicant has been evaluated using multiple financial indicators.
    The computed risk probability is <b>{probability:.2f}</b>, indicating a <b>{decision}</b>.

    Key contributing factors include credit behavior, income stability,
    existing liabilities, and repayment history.

    This decision aligns with internal risk policies of NovaTrust Bank.
    """

    content.append(Paragraph(summary_text, styles['Normal']))
    content.append(Spacer(1, 10))





    content.append(Paragraph(f"Risk Probability: {probability:.2f}", styles['Normal']))
    content.append(Paragraph(f"<b>Final Decision:</b> {decision}", styles['Normal']))
    content.append(Spacer(1, 20))

    # -------------------------------
    # SAVE SHAP FIG TEMP
    # -------------------------------
    shap_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    shap_fig.savefig(shap_temp.name, bbox_inches='tight')

    content.append(Paragraph("<b>3. Explainability Insights</b>", styles['Heading3']))
    content.append(Spacer(1, 10))
    content.append(Image(shap_temp.name, width=4.5*inch, height=2.5*inch))
    content.append(Spacer(1, 20))

    # -------------------------------
    # SAVE GAUGE FIG TEMP
    # -------------------------------
    gauge_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    gauge_fig.write_image(gauge_temp.name)

    content.append(Paragraph("<b>4. Risk Visualization</b>", styles['Heading3']))
    content.append(Spacer(1, 10))
    content.append(Image(gauge_temp.name, width=4.5*inch, height=2.5*inch))
    content.append(Spacer(1, 20))

    # -------------------------------
    # TOP FACTORS
    # -------------------------------
    content.append(Paragraph("<b>5. Key Decision Drivers</b>", styles['Heading3']))
    content.append(Spacer(1, 10))

    for name, val in top_features:
        if val > 0:
            text = f"{name} increased risk"
        else:
            text = f"{name} reduced risk"

        content.append(Paragraph(text, styles['Normal']))
        content.append(Spacer(1, 5))

    content.append(Spacer(1, 20))

    content.append(Paragraph(
        "This report is system-generated using advanced Machine Learning models and Explainable AI (SHAP).",
        styles['Italic']
    ))

    doc.build(content)
    buffer.seek(0)

    return buffer

# -------------------------------
# LOAD MODEL (CORRECT)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "credit_risk_model.pkl")


scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)



# -------------------------------
# UI TITLE
# -------------------------------
st.markdown("## 💳 Credit Risk Prediction Dashboard")
st.markdown("### Smart Loan Approval System using Machine Learning")
st.divider()

# -------------------------------
# SIDEBAR INPUTS
# -------------------------------
st.sidebar.header("Enter Customer Details")

fea_1 = st.sidebar.slider("Credit Score", 1, 10, 5,help="Higher score means better repayment history and lower risk.")
fea_2 = st.sidebar.slider("Monthly Income (₹)", 10000, 300000, 50000,help="Stable and higher income improves loan approval chances.")
employment_map = {"Salaried": 1, "Self-Employed": 2, "Unemployed": 3}
default_map = {"No": 0, "Yes": 1}

# Dropdowns (user sees labels)
fea_3_label = st.sidebar.selectbox(
    "Employment Type",
    list(employment_map.keys()),
    help="Salaried jobs are more stable. Unemployed increases risk."
)

fea_5_label = st.sidebar.selectbox(
    "Previous Default",
    list(default_map.keys()),
    help="Past loan defaults significantly increase risk."
)

# Convert to numeric (model needs numbers)
fea_3 = employment_map[fea_3_label]
fea_5 = default_map[fea_5_label]
fea_4 = st.sidebar.slider("Loan Amount (₹)",50000, 2000000, 300000,help="Higher loan amounts increase risk.")
#fea_5 = st.sidebar.selectbox("Previous Default", ["No", "Yes"])
fea_6 = st.sidebar.slider("Number of Loans", 0, 10, 5,help="Too many active loans increase financial burden.")
fea_7 = st.sidebar.slider("Loan Duration (years)", 0, 10, 3,help="Longer durations may indicate higher repayment risk.")
fea_8 = st.sidebar.slider("Credit Cards", 0, 100, 20,help="Too many credit cards may indicate financial stress.")
fea_9 = st.sidebar.slider("Late Payments", 1, 5, 2,help="Frequent late payments negatively impact creditworthiness.")
fea_10 = st.sidebar.slider("Total Assets (₹)", 50000, 5000000, 300000,help="Higher assets provide financial security.")
fea_11 = st.sidebar.slider("Debt-to-Income Ratio (%)", 1.0, 100.0, 30.0,help="Higher ratio means more debt compared to income → risky.")


# -------------------------------
# INPUT ARRAY
# -------------------------------
input_data = np.array([[fea_1, fea_2, fea_3, fea_4,
                        fea_5, fea_6, fea_7,
                        fea_8, fea_9, fea_10, fea_11]])

input_data_scaled = scaler.transform(input_data)

# -------------------------------
# PREDICTION
# -------------------------------
st.write("### Click below to predict")

import shap
import matplotlib.pyplot as plt

# 🔹 Background data (VERY IMPORTANT)
background = np.zeros((1, 11))  # 11 features

# 🔹 Create explainer
# -------------------------------
# LOAD SCALER
# -------------------------------
model = joblib.load(model_path)
scaler = joblib.load("scaler.pkl")

# -------------------------------
# SHAP EXPLAINER
# -------------------------------
explainer = shap.Explainer(model)


if st.button("Predict", key="predict_btn"):


    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]
    
    # 👇 ADD THEM HERE
    #st.write(f"Monthly Income: ₹{fea_2:,}")
    #st.write(f"Loan Amount: ₹{fea_4:,}")

     # 👇 ADD YOUR METRIC CODE HERE
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("💰 Monthly Income", f"₹{fea_2:,}")

    with col2:
        st.metric("🏦 Loan Amount", f"₹{fea_4:,}")

    with col3:
        st.metric("📊 Risk Score", f"{probability:.2f}")

    st.divider()  # optional but recommended
    #
    st.subheader("🎯 Risk Analysis Meter")

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability,
    
        title={'text': "Credit Risk Score", 'font': {'size': 22}},
    
        delta={'reference': 0.5, 'increasing': {'color': "red"}},

        gauge={
            'axis': {
                'range': [0, 1],
                'tickwidth': 1,
                'tickcolor': "#333"
            },

            'bar': {'color': "#1f77b4", 'thickness': 0.3},

            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#ccc",

            'steps': [
                {'range': [0, 0.3], 'color': "#2ecc71"},   # green
                {'range': [0.3, 0.6], 'color': "#f39c12"}, # orange
                {'range': [0.6, 1], 'color': "#e74c3c"},   # red
            ],

            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': probability
            }
        }
    ))

    st.plotly_chart(fig, use_container_width=True)
    gauge_fig = fig   # ✅ ADD THIS LINE


    st.markdown("📊 Prediction Result")
    st.write(f"Risk Probability: {probability:.2f}")
    st.write("Risk Level Indicator")
    st.progress(float(probability))

    if probability > 0.6:
        st.error(f"❌ HIGH RISK ({probability:.2f}) - Loan Rejected")

    elif probability > 0.3:
        st.warning(f"⚠ MEDIUM RISK ({probability:.2f}) - Needs Manual Review")

    else:
        st.success(f"✅ LOW RISK ({probability:.2f}) - Loan Approved")
    #
    # 👇 ADD HERE (exactly here)
    st.subheader("🧠 Decision Summary")

    if probability > 0.6:
        st.write("The loan was rejected because the system detected high risk based on multiple financial factors.")

    elif probability > 0.3:
        st.write("The customer falls in a medium risk category. This requires manual verification by a bank officer.")

    else:
        st.write("The loan is approved because the customer shows strong financial stability and low risk.")
    
    st.subheader("🔍 SHAP Explanation")

    feature_names = [
        "Credit Score","Monthly Income","Employment Type","Loan Amount",
        "Previous Default","Number of Loans","Loan Duration",
        "Credit Cards","Late Payments",
        "Total Assets","Debt-to-Income Ratio"
    ]

    # SHAP VALUES
    shap_values = explainer(input_scaled)    

    # CREATE FIGURE
    # ✅ SMALL CLEAN SHAP GRAPH
    plt.figure(figsize=(8,4))

    # WATERFALL PLOT
    shap.plots.waterfall(
        shap_values[0, :, 1],
        max_display=11,
        show=False
    )

    # ✅ REMOVE EXTRA WHITE SPACE
    plt.tight_layout()

    # ✅ GET CURRENT FIGURE
    fig = plt.gcf()

    # ✅ SHOW IN STREAMLIT
    st.pyplot(fig)

    # ✅ SAVE FOR PDF
    shap_fig = fig
    
    # 🔹 DASHBOARD (comes AFTER SHAP)

    st.subheader("📊 Dashboard")

    feature_names = [
        "Credit Score","Monthly Income","Employment Type","Loan Amount","Previous Default",
        "Number of Loans","Loan Duration","Credit Cards","Late Payments",
        "Total Assets","Debt-to-Income Ratio"
    ]
    

    input_df = pd.DataFrame(input_data, columns=feature_names)

    st.write("### Input Data")
    st.dataframe(input_df)

    st.write("### Feature Distribution")
    st.bar_chart(input_df.T)

    st.write("### Risk Level")
    if probability > 0.6:
        st.error(f"High Risk ({probability:.2f})")
    elif probability > 0.3:
        st.warning(f"Medium Risk ({probability:.2f})")
    else:
        st.success(f"Low Risk ({probability:.2f})")
    #
    # -------------------------------
    # HUMAN FRIENDLY EXPLANATION
    # -------------------------------
    st.subheader("🧠 Why this decision?")

    # Extract numeric SHAP values for High Risk class
    shap_vals = shap_values.values[0][:, 1]

    # Convert all values to float
    shap_vals = [float(v) for v in shap_vals]

    feature_names = [
        "Credit Score","Monthly Income","Employment Type","Loan Amount","Previous Default",
        "Number of Loans","Loan Duration","Credit Cards","Late Payments",
        "Total Assets","Debt-to-Income Ratio"
    ]

    # Get top important features
    importance = list(zip(feature_names, shap_vals))

    importance = sorted(
        importance,
        key=lambda x: abs(x[1]),
        reverse=True
    )

    top_features = importance[:3]

    st.write("### 🔍 Key Factors Influencing Decision:")

    for name, val in importance[:3]:
        if val > 0:
            st.write(f"🔴 {name} increased risk")
        else:
            st.write(f"🟢 {name} reduced risk")
    #
    st.write("### 📌 Final Decision Reason:")


    if probability > 0.6:
        st.write("Loan rejected because multiple factors indicate HIGH risk.")
    elif probability > 0.3:
        st.write("Customer falls in MEDIUM risk zone. Needs manual review.")
    else:
        st.write("Loan approved because most factors indicate LOW risk.")
    #
    
    st.write("📌 Final Decision Reason:")
    # 🔥 PDF GENERATION (ADD HERE)

    if probability > 0.6:
        decision_text = "Loan Rejected (High Risk)"
    elif probability > 0.3:
        decision_text = "Manual Review Required (Medium Risk)"
    else:
        decision_text = "Loan Approved (Low Risk)"

    #
    pdf_file = generate_pdf(
        probability,
        decision_text,
        input_data,
        feature_names,
        top_features,shap_fig,gauge_fig
    )

    st.download_button(
        label="📥 Download Credit Report",
        data=pdf_file,
        file_name="Credit_Risk_Report.pdf",
        mime="application/pdf"
    )




    #
    # -------------------------------
    # ACTIONABLE INSIGHTS
    # -------------------------------

    # 🔴 For Rejected or Medium Risk
    if probability > 0.3:

        st.subheader("📉 How to Improve Approval Chances")

        improvements = []

        if fea_1 < 5:
            improvements.append("Improve your Credit Score")

        if fea_5 == 1:
            improvements.append("Avoid loan defaults and maintain good repayment history")

        if fea_11 > 300:
            improvements.append("Reduce your Debt-to-Income Ratio")

        if fea_6 > 5:
            improvements.append("Reduce number of active loans")

        if fea_2 < 80:
            improvements.append("Increase your monthly income")

        if len(improvements) == 0:
            st.write("No major issues detected, but slight improvements can reduce risk further.")
        else:
            for imp in improvements:
                st.write(f"🔧 {imp}")


    # 🟢 For Approved
    else:

        st.subheader("📈 Why You Are Eligible")

        strengths = []

        if fea_1 >= 7:
            strengths.append("Strong Credit Score")

        if fea_5 == 0:
            strengths.append("No Previous Defaults")

        if fea_2 > 120:
            strengths.append("Stable Monthly Income")

        if fea_11 < 200:
            strengths.append("Healthy Debt-to-Income Ratio")

        if len(strengths) == 0:
            st.write("You meet basic eligibility criteria.")
        else:
            for s in strengths:
                st.write(f"✅ {s}")