import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SmartCare Hospital",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# LOAD MODEL AND PREPROCESSING OBJECTS
# ============================================================

model = joblib.load("readmission_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")
numeric_features = joblib.load("numeric_features.pkl")


# ============================================================
# IMAGE URLS
# ============================================================

LOGIN_BG = (
    "https://raw.githubusercontent.com/wish27r/"
    "smartcare-readmission-prediction/main/login_bg.png"
)

PREDICT_HEADER = (
    "https://raw.githubusercontent.com/wish27r/"
    "smartcare-readmission-prediction/main/predict_header.png"
)

INSIGHTS_HEADER = (
    "https://raw.githubusercontent.com/wish27r/"
    "smartcare-readmission-prediction/main/insights_header.png"
)

ABOUT_HEADER = (
    "https://raw.githubusercontent.com/wish27r/"
    "smartcare-readmission-prediction/main/about_header.png"
)


# ============================================================
# GLOBAL UI / BACKGROUND / ANIMATIONS
# ============================================================

st.markdown(
    f"""
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    html,
    body,
    [data-testid="stAppViewContainer"] {{
        margin: 0;
        padding: 0;
        background: transparent !important;
    }}

    [data-testid="stAppViewContainer"] {{
        background: transparent !important;
    }}

    [data-testid="stHeader"] {{
        background: transparent !important;
    }}

    /* ========================================================
       FIXED BLURRED LOGIN IMAGE BACKGROUND
       ======================================================== */

    [data-testid="stAppViewContainer"]::before {{
        content: "";

        position: fixed;

        top: -35px;
        left: -35px;

        width: calc(100% + 70px);
        height: calc(100% + 70px);

        background-image: url("{LOGIN_BG}");

        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;

        filter: blur(14px);

        transform: scale(1.06);

        z-index: -3;

        pointer-events: none;
    }}

    /* ========================================================
       BACKGROUND READABILITY OVERLAY
       ======================================================== */

    [data-testid="stAppViewContainer"]::after {{
        content: "";

        position: fixed;

        inset: 0;

        background:
            linear-gradient(
                135deg,
                rgba(255,255,255,0.78),
                rgba(236,248,253,0.72)
            );

        z-index: -2;

        pointer-events: none;
    }}

    /* ========================================================
       MAIN CONTENT
       ======================================================== */

    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;

        padding-left: 4rem;
        padding-right: 4rem;

        max-width: 1450px;

        animation:
            pageFadeIn 0.65s ease-out;
    }}

    /* ========================================================
       PAGE ANIMATION
       ======================================================== */

    @keyframes pageFadeIn {{

        0% {{
            opacity: 0;
            transform: translateY(18px);
        }}

        100% {{
            opacity: 1;
            transform: translateY(0);
        }}

    }}

    @keyframes softFade {{

        0% {{
            opacity: 0;
            transform: translateY(14px);
        }}

        100% {{
            opacity: 1;
            transform: translateY(0);
        }}

    }}

    @keyframes tabTransition {{

        0% {{
            opacity: 0;
            transform: translateY(15px);
            filter: blur(2px);
        }}

        60% {{
            opacity: 0.75;
            filter: blur(0.5px);
        }}

        100% {{
            opacity: 1;
            transform: translateY(0);
            filter: blur(0);
        }}

    }}

    /* ========================================================
       MAIN TITLE
       ======================================================== */

    h1 {{
        font-size: 2.7rem !important;

        font-weight: 800 !important;

        letter-spacing: -1px;

        color: #123b55 !important;

        text-shadow:
            0 2px 12px rgba(255,255,255,0.75);
    }}

    h2,
    h3 {{
        color: #163d58 !important;
    }}

    p {{
        color: #294b60;
    }}

    /* ========================================================
       GENERAL GLASS CARD
       ======================================================== */

    .glass-card {{

        background:
            rgba(255,255,255,0.62);

        border:
            1px solid rgba(255,255,255,0.82);

        border-radius: 24px;

        padding: 28px;

        box-shadow:
            0 18px 50px rgba(20,80,110,0.13),
            inset 0 1px 0 rgba(255,255,255,0.85);

        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);

        animation:
            softFade 0.55s ease-out;
    }}

    /* ========================================================
       TAB NAVIGATION
       ======================================================== */

    [data-baseweb="tab-list"] {{

        gap: 10px;

        background:
            rgba(255,255,255,0.48);

        padding: 8px;

        border-radius: 18px;

        border:
            1px solid rgba(255,255,255,0.70);

        box-shadow:
            0 8px 30px rgba(30,90,120,0.10);

        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);

        margin-bottom: 20px;
    }}

    [data-baseweb="tab"] {{

        height: 48px;

        padding: 0 28px;

        border-radius: 13px;

        color: #31566d !important;

        font-weight: 600;

        transition:
            all 0.30s ease;
    }}

    [data-baseweb="tab"]:hover {{

        background:
            rgba(255,255,255,0.72);

        transform:
            translateY(-1px);
    }}

    [aria-selected="true"] {{

        background:
            rgba(255,255,255,0.92) !important;

        color:
            #087ea4 !important;

        box-shadow:
            0 5px 18px rgba(30,120,150,0.15);
    }}

    [data-baseweb="tab-highlight"] {{

        background:
            #18a8c9 !important;

        height:
            3px !important;

        border-radius:
            10px;
    }}

    /* ========================================================
       TAB CONTENT ANIMATION
       ======================================================== */

    [data-baseweb="tab-panel"] {{

        animation:
            tabTransition 0.55s ease-out;
    }}

    /* ========================================================
       HEADER IMAGES
       ======================================================== */

    img {{

        border-radius: 20px;

        box-shadow:
            0 12px 35px rgba(20,70,90,0.14);
    }}

    /* ========================================================
       INPUT FIELDS
       ======================================================== */

    div[data-baseweb="input"],
    div[data-baseweb="select"] {{
        border-radius: 12px !important;
    }}

    input {{

        background:
            rgba(255,255,255,0.75) !important;

        border-radius:
            12px !important;

        border:
            1px solid rgba(100,170,195,0.25) !important;
    }}

    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {{

        border-radius:
            13px;

        min-height:
            46px;

        font-weight:
            700;

        border:
            1px solid rgba(255,255,255,0.75);

        background:
            linear-gradient(
                135deg,
                rgba(21,157,190,0.96),
                rgba(25,121,165,0.96)
            );

        color:
            white;

        box-shadow:
            0 8px 22px rgba(20,120,160,0.22);

        transition:
            transform 0.25s ease,
            box-shadow 0.25s ease,
            filter 0.25s ease;
    }}

    .stButton > button:hover {{

        transform:
            translateY(-2px);

        box-shadow:
            0 12px 28px rgba(20,120,160,0.30);

        filter:
            brightness(1.05);
    }}

    .stButton > button:active {{
        transform: translateY(0);
    }}

    /* ========================================================
       DIVIDERS
       ======================================================== */

    hr {{

        border:
            none;

        height:
            1px;

        background:
            linear-gradient(
                90deg,
                transparent,
                rgba(50,130,160,0.30),
                transparent
            );

        margin:
            25px 0;
    }}

    /* ========================================================
       ALERTS
       ======================================================== */

    [data-testid="stAlert"] {{

        border-radius:
            15px;

        backdrop-filter:
            blur(12px);

        -webkit-backdrop-filter:
            blur(12px);
    }}

    /* ========================================================
       PROGRESS BAR
       ======================================================== */

    [data-testid="stProgressBar"] > div > div {{
        border-radius: 20px;
    }}


    /* ========================================================
       ========================================================
       ABOUT PAGE
       ========================================================
       ======================================================== */

    .about-container {{

        padding:
            28px 38px 38px 38px;

        animation:
            softFade 0.55s ease-out;
    }}


    /* --------------------------------------------------------
       ABOUT INTRO
       -------------------------------------------------------- */

    .about-intro {{

        text-align:
            center;

        max-width:
            800px;

        margin:
            38px auto 15px auto;

        animation:
            aboutIntro 0.7s ease-out;
    }}

    @keyframes aboutIntro {{

        0% {{
            opacity: 0;
            transform:
                translateY(20px)
                scale(0.98);
        }}

        100% {{
            opacity: 1;
            transform:
                translateY(0)
                scale(1);
        }}

    }}

    .about-icon {{

        width:
            76px;

        height:
            76px;

        margin:
            0 auto 18px auto;

        display:
            flex;

        align-items:
            center;

        justify-content:
            center;

        border-radius:
            23px;

        background:
            linear-gradient(
                135deg,
                rgba(21,157,190,0.16),
                rgba(25,121,165,0.08)
            );

        border:
            1px solid rgba(21,157,190,0.18);

        font-size:
            36px;

        box-shadow:
            0 12px 32px rgba(20,120,160,0.13);
    }}

    .about-title {{

        margin:
            0 !important;

        font-size:
            2.5rem !important;

        font-weight:
            800 !important;

        color:
            #123b55 !important;

        letter-spacing:
            -1px;
    }}

    .about-subtitle {{

        margin-top:
            7px;

        font-size:
            1.15rem;

        font-weight:
            600;

        color:
            #1292b4;
    }}

    .about-description {{

        max-width:
            650px;

        margin:
            18px auto 0 auto;

        font-size:
            1rem;

        line-height:
            1.7;

        color:
            #526f7f;
    }}


    /* --------------------------------------------------------
       INFORMATION CARDS
       -------------------------------------------------------- */

    .info-card {{

        min-height:
            225px;

        padding:
            27px 23px;

        text-align:
            center;

        border-radius:
            20px;

        background:
            rgba(255,255,255,0.57);

        border:
            1px solid rgba(255,255,255,0.82);

        box-shadow:
            0 12px 30px rgba(30,90,120,0.10);

        backdrop-filter:
            blur(15px);

        -webkit-backdrop-filter:
            blur(15px);

        transition:
            transform 0.30s ease,
            box-shadow 0.30s ease;

        animation:
            cardAppear 0.7s ease-out;
    }}

    .info-card:hover {{

        transform:
            translateY(-6px);

        box-shadow:
            0 18px 38px rgba(30,100,130,0.16);
    }}

    @keyframes cardAppear {{

        0% {{
            opacity: 0;
            transform:
                translateY(15px);
        }}

        100% {{
            opacity: 1;
            transform:
                translateY(0);
        }}

    }}

    .info-icon {{

        font-size:
            30px;

        margin-bottom:
            14px;
    }}

    .info-label {{

        font-size:
            0.70rem;

        font-weight:
            800;

        letter-spacing:
            1.2px;

        color:
            #648494;

        margin-bottom:
            8px;
    }}

    .info-value {{

        font-size:
            1.05rem;

        font-weight:
            750;

        color:
            #164b66;

        margin-bottom:
            12px;
    }}

    .info-description {{

        font-size:
            0.86rem;

        line-height:
            1.55;

        color:
            #607b89;
    }}


    /* --------------------------------------------------------
       PROJECT SECTION
       -------------------------------------------------------- */

    .project-section {{

        max-width:
            900px;

        margin:
            35px auto 25px auto;

        padding:
            28px 35px;

        text-align:
            center;

        border-radius:
            20px;

        background:
            rgba(255,255,255,0.48);

        border:
            1px solid rgba(255,255,255,0.70);

        backdrop-filter:
            blur(12px);

        -webkit-backdrop-filter:
            blur(12px);

        box-shadow:
            0 10px 30px rgba(30,90,120,0.07);
    }}

    .section-small-title {{

        font-size:
            0.72rem;

        letter-spacing:
            1.8px;

        font-weight:
            800;

        color:
            #1599b8;

        margin-bottom:
            12px;
    }}

    .project-text {{

        font-size:
            0.95rem;

        line-height:
            1.75;

        color:
            #4f6d7c;
    }}

    .project-text strong {{
        color:
            #164c66;
    }}


    /* --------------------------------------------------------
       DISCLAIMER
       -------------------------------------------------------- */

    .disclaimer-card {{

        max-width:
            900px;

        margin:
            25px auto 0 auto;

        padding:
            20px 25px;

        display:
            flex;

        align-items:
            flex-start;

        gap:
            17px;

        text-align:
            left;

        border-radius:
            18px;

        background:
            rgba(255,248,225,0.70);

        border:
            1px solid rgba(240,190,70,0.30);

        box-shadow:
            0 10px 25px rgba(150,110,30,0.08);
    }}

    .disclaimer-icon {{

        font-size:
            27px;

        flex-shrink:
            0;
    }}

    .disclaimer-title {{

        font-size:
            0.95rem;

        font-weight:
            800;

        color:
            #856c2c;

        margin-bottom:
            4px;
    }}

    .disclaimer-text {{

        font-size:
            0.83rem;

        line-height:
            1.55;

        color:
            #786d51;
    }}


    /* --------------------------------------------------------
       ABOUT FOOTER
       -------------------------------------------------------- */

    .about-footer {{

        text-align:
            center;

        margin-top:
            35px;

        padding-top:
            10px;
    }}

    .footer-line {{

        width:
            100px;

        height:
            2px;

        margin:
            0 auto 17px auto;

        border-radius:
            10px;

        background:
            linear-gradient(
                90deg,
                transparent,
                #1599b8,
                transparent
            );
    }}

    .footer-text {{

        font-size:
            0.82rem;

        font-weight:
            600;

        color:
            #65818f;
    }}

    .footer-text span {{

        margin:
            0 7px;

        color:
            #1599b8;
    }}

    .footer-subtext {{

        margin-top:
            7px;

        font-size:
            0.72rem;

        color:
            #8ca0aa;
    }}


    /* ========================================================
       LOGIN PAGE
       ======================================================== */

    .login-container {{

        min-height:
            72vh;

        display:
            flex;

        align-items:
            center;

        justify-content:
            center;

        animation:
            pageFadeIn 0.8s ease-out;
    }}

    .login-card {{

        width:
            430px;

        padding:
            42px;

        text-align:
            center;

        border-radius:
            28px;

        background:
            rgba(255,255,255,0.64);

        border:
            1px solid rgba(255,255,255,0.85);

        box-shadow:
            0 25px 70px rgba(20,70,100,0.18);

        backdrop-filter:
            blur(22px);

        -webkit-backdrop-filter:
            blur(22px);
    }}

    .login-icon {{

        font-size:
            52px;

        margin-bottom:
            10px;
    }}

    .login-title {{

        font-size:
            2rem;

        font-weight:
            800;

        color:
            #123b55;

        margin-bottom:
            6px;
    }}

    .login-subtitle {{

        font-size:
            0.95rem;

        color:
            #5d7785;

        line-height:
            1.5;
    }}


    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 768px) {{

        .main .block-container {{

            padding-left:
                1rem;

            padding-right:
                1rem;
        }}

        h1 {{
            font-size:
                2rem !important;
        }}

        [data-baseweb="tab"] {{

            padding:
                0 12px;

            font-size:
                0.85rem;
        }}

        .about-container {{

            padding:
                20px 15px 25px 15px;
        }}

        .about-title {{

            font-size:
                2rem !important;
        }}

        .about-description {{

            font-size:
                0.9rem;
        }}

        .info-card {{

            margin-bottom:
                15px;
        }}

        .project-section {{

            padding:
                23px 20px;
        }}

        .disclaimer-card {{

            padding:
                18px;
        }}

        .login-card {{

            width:
                auto;

            margin:
                10px;
        }}

    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PASSWORD PROTECTION
# ============================================================

def check_password():

    def password_entered():

        if st.session_state["password"] == "smartcare2026":

            st.session_state["password_correct"] = True

            del st.session_state["password"]

        else:

            st.session_state["password_correct"] = False


    # --------------------------------------------------------
    # FIRST LOGIN
    # --------------------------------------------------------

    if "password_correct" not in st.session_state:

        st.markdown(
            """
            <div class="login-container">

                <div class="login-card">

                    <div class="login-icon">
                        🏥
                    </div>

                    <div class="login-title">
                        SmartCare Hospital
                    </div>

                    <div class="login-subtitle">
                        AI-Powered 30-Day Patient
                        Readmission Prediction
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.text_input(
            "Enter password",
            type="password",
            on_change=password_entered,
            key="password"
        )

        return False


    # --------------------------------------------------------
    # INCORRECT PASSWORD
    # --------------------------------------------------------

    elif not st.session_state["password_correct"]:

        st.markdown(
            """
            <div class="login-container">

                <div class="login-card">

                    <div class="login-icon">
                        🔐
                    </div>

                    <div class="login-title">
                        SmartCare Hospital
                    </div>

                    <div class="login-subtitle">
                        Please enter the correct password
                        to continue.
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.text_input(
            "Enter password",
            type="password",
            on_change=password_entered,
            key="password"
        )

        st.error("Incorrect password")

        return False


    # --------------------------------------------------------
    # PASSWORD CORRECT
    # --------------------------------------------------------

    else:

        return True


# ============================================================
# STOP UNTIL LOGIN
# ============================================================

if not check_password():

    st.stop()


# ============================================================
# MAIN APPLICATION HEADER
# ============================================================

st.markdown(
    """
    <div
        style="
            text-align:center;
            margin-bottom:20px;
            animation:softFade 0.6s ease-out;
        "
    >

        <h1>
            SmartCare Hospital
        </h1>

        <p
            style="
                font-size:1.15rem;
                color:#527487;
                margin-top:-15px;
            "
        >
            30-Day Patient Readmission Risk Predictor
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# NAVIGATION
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🩺  Predict",
        "📊  Model Insights",
        "ℹ️  About"
    ]
)


# ============================================================
# TAB 1 — PREDICT
# ============================================================

with tab1:

    st.markdown(
        '<div class="glass-card">',
        unsafe_allow_html=True
    )

    st.image(
        PREDICT_HEADER,
        use_container_width=True
    )

    st.write(
        "Enter patient details below to predict the likelihood "
        "of readmission within 30 days of discharge."
    )

    st.divider()


    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # LEFT COLUMN
    # --------------------------------------------------------

    with col1:

        age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            value=45
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        department = st.selectbox(
            "Department",
            [
                "Cardiology",
                "Neurology",
                "Orthopedics",
                "Pediatrics",
                "General Medicine",
                "Oncology"
            ]
        )

        previous_admissions = st.number_input(
            "Previous Admissions",
            min_value=0,
            max_value=20,
            value=1
        )

        length_of_stay_days = st.number_input(
            "Length of Stay (days)",
            min_value=1,
            max_value=60,
            value=5
        )

        room_type = st.selectbox(
            "Room Type",
            [
                "General Ward",
                "Private Room",
                "ICU",
                "Unknown"
            ]
        )

        systolic_bp = st.number_input(
            "Systolic BP",
            min_value=70,
            max_value=220,
            value=120
        )

        diastolic_bp = st.number_input(
            "Diastolic BP",
            min_value=40,
            max_value=140,
            value=80
        )


    # --------------------------------------------------------
    # RIGHT COLUMN
    # --------------------------------------------------------

    with col2:

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=24.0
        )

        cholesterol_mg_dl = st.number_input(
            "Cholesterol (mg/dL)",
            min_value=100,
            max_value=400,
            value=180
        )

        blood_sugar_mg_dl = st.number_input(
            "Blood Sugar (mg/dL)",
            min_value=50,
            max_value=400,
            value=100
        )

        lab_tests_count = st.number_input(
            "Lab Tests Count",
            min_value=0,
            max_value=30,
            value=3
        )

        treatments_count = st.number_input(
            "Treatments Count",
            min_value=0,
            max_value=30,
            value=2
        )

        missed_previous_appointments = st.number_input(
            "Missed Previous Appointments",
            min_value=0,
            max_value=20,
            value=0
        )

        previous_appointments = st.number_input(
            "Previous Appointments",
            min_value=0,
            max_value=50,
            value=3
        )

        total_bill_lkr = st.number_input(
            "Total Bill (LKR)",
            min_value=0,
            value=50000
        )


    appointment_month = st.slider(
        "Appointment Month",
        1,
        12,
        6
    )


    st.divider()


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    if st.button(
        "🔍 Predict Readmission Risk",
        type="primary",
        use_container_width=True
    ):

        missed_appointment_rate = (
            missed_previous_appointments /
            previous_appointments
            if previous_appointments > 0
            else 0
        )

        cost_per_day = (
            total_bill_lkr /
            length_of_stay_days
            if length_of_stay_days > 0
            else total_bill_lkr
        )

        prior_utilisation = (
            previous_admissions +
            previous_appointments
        )


        # ----------------------------------------------------
        # BLOOD PRESSURE CATEGORY
        # ----------------------------------------------------

        if (
            systolic_bp >= 140
            or diastolic_bp >= 90
        ):

            bp_category = "High"

        elif (
            systolic_bp >= 130
            or diastolic_bp >= 80
        ):

            bp_category = "Elevated"

        else:

            bp_category = "Normal"


        # ----------------------------------------------------
        # INPUT DATA
        # ----------------------------------------------------

        raw_input = {

            "age":
                age,

            "gender":
                gender,

            "department":
                department,

            "previous_admissions":
                previous_admissions,

            "length_of_stay_days":
                length_of_stay_days,

            "room_type":
                room_type,

            "systolic_bp":
                systolic_bp,

            "diastolic_bp":
                diastolic_bp,

            "bmi":
                bmi,

            "cholesterol_mg_dl":
                cholesterol_mg_dl,

            "blood_sugar_mg_dl":
                blood_sugar_mg_dl,

            "lab_tests_count":
                lab_tests_count,

            "treatments_count":
                treatments_count,

            "missed_previous_appointments":
                missed_previous_appointments,

            "previous_appointments":
                previous_appointments,

            "total_bill_lkr":
                total_bill_lkr,

            "appointment_month":
                appointment_month,

            "missed_appointment_rate":
                missed_appointment_rate,

            "cost_per_day":
                cost_per_day,

            "prior_utilisation":
                prior_utilisation,

            "bp_category":
                bp_category
        }


        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

        input_df = pd.DataFrame(
            [raw_input]
        )

        input_encoded = pd.get_dummies(
            input_df
        )

        input_final = input_encoded.reindex(
            columns=feature_columns,
            fill_value=0
        )

        input_final[numeric_features] = (
            scaler.transform(
                input_final[numeric_features]
            )
        )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            input_final
        )[0]

        probability = model.predict_proba(
            input_final
        )[0][1]


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.divider()


        if prediction == 1:

            st.error(
                f"⚠️ **High Risk of Readmission** — "
                f"Predicted probability: "
                f"{probability:.1%}"
            )

        else:

            st.success(
                f"✅ **Low Risk of Readmission** — "
                f"Predicted probability: "
                f"{probability:.1%}"
            )


        st.progress(
            float(probability)
        )


        st.caption(
            "This prediction is generated by an XGBoost model "
            "trained on the SmartCare Hospital dataset. "
            "It is a coursework prototype and should not be "
            "used for real clinical decision-making."
        )


        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        st.divider()

        st.write(
            "#### 🔎 Why did the model predict this?"
        )


        explainer = shap.TreeExplainer(
            model
        )

        shap_values_patient = (
            explainer.shap_values(
                input_final
            )
        )


        contrib_df = pd.DataFrame(
            {
                "Feature":
                    input_final.columns,

                "Contribution":
                    shap_values_patient[0]
            }
        )


        contrib_df["abs_val"] = (
            contrib_df["Contribution"].abs()
        )


        contrib_df = (
            contrib_df
            .sort_values(
                "abs_val",
                ascending=False
            )
            .head(8)
        )


        contrib_df = (
            contrib_df
            .sort_values(
                "Contribution"
            )
        )


        colors = [
            "#D64550"
            if val > 0
            else "#3E7CB1"
            for val in
            contrib_df["Contribution"]
        ]


        fig, ax = plt.subplots(
            figsize=(7, 4)
        )


        ax.barh(
            contrib_df["Feature"],
            contrib_df["Contribution"],
            color=colors
        )


        ax.set_xlabel(
            "Impact on prediction (SHAP value)"
        )


        ax.axvline(
            0,
            color="black",
            linewidth=0.8
        )


        fig.patch.set_alpha(0)

        ax.set_facecolor("none")


        st.pyplot(
            fig,
            use_container_width=True
        )


        st.caption(
            "🔴 Red bars pushed the prediction toward "
            "Readmitted. 🔵 Blue bars pushed it toward "
            "Not Readmitted."
        )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# TAB 2 — MODEL INSIGHTS
# ============================================================

with tab2:

    st.markdown(
        '<div class="glass-card">',
        unsafe_allow_html=True
    )


    st.image(
        INSIGHTS_HEADER,
        use_container_width=True
    )


    st.write(
        "### 📊 Which factors drive the model's predictions?"
    )


    st.write(
        "Based on XGBoost feature importance across all "
        "patients in the dataset."
    )


    st.divider()


    importance_data = pd.DataFrame(
        {
            "Feature": [

                "appointment_month",

                "previous_admissions",

                "missed_appointment_rate",

                "bp_category_Normal",

                "blood_group_A-",

                "cholesterol_mg_dl",

                "bmi",

                "blood_group_O-",

                "lab_tests_count",

                "diastolic_bp",

                "department_Pediatrics",

                "total_bill_lkr",

                "treatments_count",

                "length_of_stay_days",

                "medicine_charge_lkr"
            ],

            "Importance": [

                0.0697,

                0.0680,

                0.0547,

                0.0522,

                0.0512,

                0.0491,

                0.0443,

                0.0421,

                0.0402,

                0.0383,

                0.0379,

                0.0357,

                0.0344,

                0.0336,

                0.0334
            ]
        }
    )


    st.bar_chart(
        importance_data.set_index(
            "Feature"
        )
    )


    st.divider()


    st.write(
        "### 💡 Key Takeaways"
    )


    st.markdown(
        """
        - **Appointment month** is the strongest overall
          driver, suggesting a seasonal pattern in
          readmissions.

        - **Previous admissions** strongly influences risk —
          patients with a history of frequent admissions are
          more likely to return.

        - **Missed appointment rate** reflects patient
          engagement with follow-up care.

        - Clinical vitals such as cholesterol, BMI, and blood
          pressure matter meaningfully in combination, even
          though no single vital strongly predicts readmission
          alone.
        """
    )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# TAB 3 — ABOUT
# ============================================================

with tab3:

    st.markdown(
        '<div class="glass-card about-container">',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # ABOUT HEADER IMAGE
    # --------------------------------------------------------

    st.image(
        ABOUT_HEADER,
        use_container_width=True
    )


    # --------------------------------------------------------
    # CENTERED INTRO
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="about-intro">

            <div class="about-icon">
                🏥
            </div>

            <div class="about-title">
                SmartCare Hospital AI
            </div>

            <div class="about-subtitle">
                30-Day Patient Readmission Predictor
            </div>

            <div class="about-description">

                An AI-powered healthcare prototype designed
                to predict whether a patient is likely to be
                readmitted to hospital within 30 days of
                discharge.

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # THREE INFORMATION CARDS
    # --------------------------------------------------------

    info_col1, info_col2, info_col3 = st.columns(3)


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    with info_col1:

        st.markdown(
            """
            <div class="info-card">

                <div class="info-icon">
                    🤖
                </div>

                <div class="info-label">
                    MACHINE LEARNING MODEL
                </div>

                <div class="info-value">
                    XGBoost Classifier
                </div>

                <div class="info-description">

                    Selected based on ROC-AUC after
                    comparing Logistic Regression,
                    Random Forest, and XGBoost.

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    with info_col2:

        st.markdown(
            """
            <div class="info-card">

                <div class="info-icon">
                    🗂️
                </div>

                <div class="info-label">
                    DATASET
                </div>

                <div class="info-value">
                    SmartCare Hospital AI
                </div>

                <div class="info-description">

                    Dataset containing 330 admitted
                    patient records used for developing
                    the prediction model.

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # EXPLAINABILITY
    # --------------------------------------------------------

    with info_col3:

        st.markdown(
            """
            <div class="info-card">

                <div class="info-icon">
                    🔍
                </div>

                <div class="info-label">
                    EXPLAINABILITY
                </div>

                <div class="info-value">
                    SHAP + Feature Importance
                </div>

                <div class="info-description">

                    Model explanations help identify
                    which features influenced individual
                    predictions.

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # PROJECT INFORMATION
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="project-section">

            <div class="section-small-title">
                ABOUT THE PROJECT
            </div>

            <div class="project-text">

                This prototype was developed as part of the
                <strong>CCS3440 Artificial Intelligence</strong>
                coursework.

                It combines machine learning and explainable
                AI techniques to provide an interpretable
                prediction of patient readmission risk.

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="disclaimer-card">

            <div class="disclaimer-icon">
                ⚠️
            </div>

            <div>

                <div class="disclaimer-title">
                    Educational Prototype
                </div>

                <div class="disclaimer-text">

                    This application is developed for
                    educational and coursework purposes only.

                    It is not a certified medical device and
                    should never be used as a substitute for
                    professional medical decision-making.

                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="about-footer">

            <div class="footer-line"></div>

            <div class="footer-text">

                © 2026 SmartCare AI

                <span>•</span>

                CCS3440 Artificial Intelligence

                <span>•</span>

                SLTC

            </div>

            <div class="footer-subtext">

                AI-assisted healthcare technology prototype

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )
