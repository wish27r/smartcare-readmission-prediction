import streamlit as st
import pandas as pd
import joblib

#Loading saved model and preprocessing objects
model = joblib.load('readmission_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')
numeric_features = joblib.load('numeric_features.pkl')

st.set_page_config(page_title="SmartCare Readmission Predictor", page_icon="🏥", layout="centered")

#Image URLs
LOGIN_BG = "https://raw.githubusercontent.com/wish27r/smartcare-readmission-prediction/main/login_bg.png"
PREDICT_HEADER = "https://raw.githubusercontent.com/wish27r/smartcare-readmission-prediction/main/predict_header.png"
INSIGHTS_HEADER = "https://raw.githubusercontent.com/wish27r/smartcare-readmission-prediction/main/insights_header.png"
ABOUT_HEADER = "https://raw.githubusercontent.com/wish27r/smartcare-readmission-prediction/main/about_header.png"

#Background image helper (with readability overlay)
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(255,255,255,0.75), rgba(255,255,255,0.75)), url("{image_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

#Simple password protection
def check_password():
    def password_entered():
        if st.session_state["password"] == "smartcare2026":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        set_background(LOGIN_BG)
        st.text_input("Enter password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        set_background(LOGIN_BG)
        st.text_input("Enter password", type="password", on_change=password_entered, key="password")
        st.error("Incorrect password")
        return False
    else:
        return True

if not check_password():
    st.stop()

#App UI
st.title("SmartCare Hospital")
st.subheader("30-Day Patient Readmission Risk Predictor")

tab1, tab2, tab3 = st.tabs(["Predict", "Model Insights", "About"])

#TAB 1: PREDICT
with tab1:
    st.image(PREDICT_HEADER, use_container_width=True)
    st.write("Enter patient details below to predict the likelihood of readmission within 30 days of discharge.")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=45)
        gender = st.selectbox("Gender", ["Male", "Female"])
        department = st.selectbox("Department", ["Cardiology", "Neurology", "Orthopedics", "Pediatrics", "General Medicine", "Oncology"])
        previous_admissions = st.number_input("Previous Admissions", min_value=0, max_value=20, value=1)
        length_of_stay_days = st.number_input("Length of Stay (days)", min_value=1, max_value=60, value=5)
        room_type = st.selectbox("Room Type", ["General Ward", "Private Room", "ICU", "Unknown"])
        systolic_bp = st.number_input("Systolic BP", min_value=70, max_value=220, value=120)
        diastolic_bp = st.number_input("Diastolic BP", min_value=40, max_value=140, value=80)

    with col2:
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0)
        cholesterol_mg_dl = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=180)
        blood_sugar_mg_dl = st.number_input("Blood Sugar (mg/dL)", min_value=50, max_value=400, value=100)
        lab_tests_count = st.number_input("Lab Tests Count", min_value=0, max_value=30, value=3)
        treatments_count = st.number_input("Treatments Count", min_value=0, max_value=30, value=2)
        missed_previous_appointments = st.number_input("Missed Previous Appointments", min_value=0, max_value=20, value=0)
        previous_appointments = st.number_input("Previous Appointments", min_value=0, max_value=50, value=3)
        total_bill_lkr = st.number_input("Total Bill (LKR)", min_value=0, value=50000)

    appointment_month = st.slider("Appointment Month", 1, 12, 6)

    st.divider()

    if st.button("Predict Readmission Risk", type="primary"):

        missed_appointment_rate = missed_previous_appointments / previous_appointments if previous_appointments > 0 else 0
        cost_per_day = total_bill_lkr / length_of_stay_days if length_of_stay_days > 0 else total_bill_lkr
        prior_utilisation = previous_admissions + previous_appointments

        if systolic_bp >= 140 or diastolic_bp >= 90:
            bp_category = "High"
        elif systolic_bp >= 130 or diastolic_bp >= 80:
            bp_category = "Elevated"
        else:
            bp_category = "Normal"

        raw_input = {
            'age': age, 'gender': gender, 'department': department,
            'previous_admissions': previous_admissions, 'length_of_stay_days': length_of_stay_days,
            'room_type': room_type, 'systolic_bp': systolic_bp, 'diastolic_bp': diastolic_bp,
            'bmi': bmi, 'cholesterol_mg_dl': cholesterol_mg_dl, 'blood_sugar_mg_dl': blood_sugar_mg_dl,
            'lab_tests_count': lab_tests_count, 'treatments_count': treatments_count,
            'missed_previous_appointments': missed_previous_appointments,
            'previous_appointments': previous_appointments, 'total_bill_lkr': total_bill_lkr,
            'appointment_month': appointment_month, 'missed_appointment_rate': missed_appointment_rate,
            'cost_per_day': cost_per_day, 'prior_utilisation': prior_utilisation,
            'bp_category': bp_category
        }

        input_df = pd.DataFrame([raw_input])
        input_encoded = pd.get_dummies(input_df)
        input_final = input_encoded.reindex(columns=feature_columns, fill_value=0)
        input_final[numeric_features] = scaler.transform(input_final[numeric_features])

        prediction = model.predict(input_final)[0]
        probability = model.predict_proba(input_final)[0][1]

        st.divider()
        if prediction == 1:
            st.error(f"⚠️ **High Risk of Readmission** — Predicted probability: {probability:.1%}")
        else:
            st.success(f"✅ **Low Risk of Readmission** — Predicted probability: {probability:.1%}")

        st.progress(float(probability))
        st.caption("This prediction is generated by an XGBoost model trained on the SmartCare Hospital dataset. "
                   "It is a coursework prototype and should not be used for real clinical decision-making.")

#TAB 2: MODEL INSIGHTS
with tab2:
    st.image(INSIGHTS_HEADER, use_container_width=True)
    st.write("### Which factors drive the model's predictions?")
    st.write("Based on XGBoost feature importance across all patients in the dataset.")

    importance_data = pd.DataFrame({
        'Feature': ['appointment_month', 'previous_admissions', 'missed_appointment_rate',
                    'bp_category_Normal', 'blood_group_A-', 'cholesterol_mg_dl', 'bmi',
                    'blood_group_O-', 'lab_tests_count', 'diastolic_bp',
                    'department_Pediatrics', 'total_bill_lkr', 'treatments_count',
                    'length_of_stay_days', 'medicine_charge_lkr'],
        'Importance': [0.0697, 0.0680, 0.0547, 0.0522, 0.0512, 0.0491, 0.0443,
                        0.0421, 0.0402, 0.0383, 0.0379, 0.0357, 0.0344, 0.0336, 0.0334]
    })

    st.bar_chart(importance_data.set_index('Feature'))

    st.write("---")
    st.write("**Key takeaways:**")
    st.markdown("""
    - **Appointment month** is the strongest overall driver, suggesting a seasonal pattern in readmissions.
    - **Previous admissions** strongly influences risk — patients with a history of frequent admissions are more likely to return.
    - **Missed appointment rate** reflects patient engagement with follow-up care.
    - Clinical vitals (cholesterol, BMI, blood pressure) matter meaningfully in combination, even though no single vital strongly predicts readmission alone.
    """)

#TAB 3: ABOUT
with tab3:
    st.image(ABOUT_HEADER, use_container_width=True)
    st.write("### About This Project")
    st.markdown("""
    **SmartCare Hospital AI — 30-Day Readmission Predictor**

    This prototype was built as part of the CCS3440 Artificial Intelligence coursework. 
    It predicts whether a patient is likely to be readmitted to hospital within 30 days 
    of discharge, based on demographic, clinical, and operational data.

    **Model:** XGBoost Classifier (selected based on ROC-AUC after comparing Logistic 
    Regression, Random Forest, and XGBoost)

    **Dataset:** SmartCare Hospital AI Dataset — 330 admitted patient records

    **Explainability:** Feature importance and SHAP were used to interpret model predictions

    ⚠️
    **Disclaimer:** This is an educational coursework prototype only. It is not a 
    certified medical device and should never be used for real clinical decision-making.

    ---
    © 2026 SmartCare AI · CCS3440 Coursework Project · SLTC
    """)
