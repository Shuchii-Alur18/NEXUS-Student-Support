
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ==========================================================
# 1. LOAD TRAINED MACHINE LEARNING MODEL
# ==========================================================

model = joblib.load("final_student_support_model.pkl")
model_features = joblib.load("final_model_features.pkl")
metadata = joblib.load("model_metadata.pkl")

MODULES = metadata.get("modules", ["AAA", "BBB", "CCC"])
PRESENTATIONS = metadata.get(
    "presentations", ["2013B", "2013J", "2014B", "2014J"]
)

# ==========================================================
# 2. STREAMLIT PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="NEXUS | Student Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# 3. PREMIUM DARK CSS
# ==========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 85% 0%, #18234A 0%, transparent 35%),
        #080D1B;
    color: #F1F5F9;
}

.block-container {
    max-width: 1550px;
    padding: 2rem 2.5rem 3rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

h1, h2, h3, h4 {
    color: #F8FAFC !important;
}

h1 {
    font-size: 36px !important;
    font-weight: 800 !important;
}

h2 {
    font-size: 27px !important;
    font-weight: 750 !important;
}

h3 {
    font-size: 20px !important;
    font-weight: 700 !important;
}

p, label, li {
    color: #CBD5E1;
    font-size: 15px;
    line-height: 1.7;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: #0C1427;
    border-right: 1px solid #263653;
}

[data-testid="stSidebar"] * {
    color: #E2E8F0;
}

/* HERO */

.hero-tag {
    display: inline-block;
    padding: 8px 14px;
    border-radius: 30px;
    background: rgba(139,92,246,.20);
    border: 1px solid rgba(196,181,253,.4);
    color: #DDD6FE;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
}

.hero-title {
    font-size: 38px;
    font-weight: 800;
    color: white;
    margin: 18px 0 12px;
}

.hero-description {
    color: #D5DEFF;
    font-size: 16px;
    line-height: 1.8;
    max-width: 700px;
}

/* KPI CARDS */

.kpi-card {
    background: linear-gradient(145deg, #172541, #111A2E);
    border: 1px solid #2C4268;
    border-radius: 17px;
    padding: 22px;
    min-height: 155px;
    box-shadow: 0 8px 25px rgba(0,0,0,.15);
}

.kpi-icon {
    font-size: 27px;
    margin-bottom: 12px;
}

.kpi-label {
    color: #A5B4CC;
    font-size: 14px;
    font-weight: 600;
}

.kpi-value {
    color: white;
    font-size: 31px;
    font-weight: 800;
    margin-top: 7px;
}

.kpi-note {
    color: #A78BFA;
    font-size: 12px;
    margin-top: 5px;
}

/* WORKFLOW CARDS */

.workflow-card {
    background: linear-gradient(145deg, #172541, #111A2E);
    border: 1px solid #304467;
    border-radius: 16px;
    padding: 22px;
    min-height: 155px;
}

.workflow-icon {
    font-size: 30px;
    margin-bottom: 14px;
}

.workflow-title {
    color: #E9D5FF;
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 8px;
}

.workflow-description {
    color: #AFC0D8;
    font-size: 14px;
    line-height: 1.7;
}

/* PREDICTION RESULT */

.risk-card {
    background: linear-gradient(120deg, #421E38, #21172F);
    border: 1px solid #F472B6;
    border-radius: 18px;
    padding: 28px;
}

.safe-card {
    background: linear-gradient(120deg, #103B39, #142A31);
    border: 1px solid #34D399;
    border-radius: 18px;
    padding: 28px;
}

/* INPUTS */

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background: #17233C !important;
    color: white !important;
    border: 1px solid #394C6B !important;
    border-radius: 9px !important;
}

div[data-baseweb="select"] > div {
    background: #17233C !important;
    border-color: #394C6B !important;
}

/* BUTTONS */

div.stButton > button,
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(100deg, #2563EB, #7C3AED);
    color: white !important;
    border: none;
    border-radius: 10px;
    min-height: 47px;
    font-size: 15px;
    font-weight: 700;
}

div.stButton > button:hover {
    background: linear-gradient(100deg, #3B82F6, #8B5CF6);
}

/* TABS */

button[data-baseweb="tab"] {
    color: #CBD5E1 !important;
    font-size: 15px !important;
    font-weight: 700 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #C4B5FD !important;
}

[data-testid="stMetric"] {
    background: #17233C;
    border: 1px solid #344767;
    border-radius: 12px;
    padding: 15px;
}

[data-testid="stMetricValue"] {
    color: white !important;
}

[data-testid="stMetricLabel"] {
    color: #A5B4CC !important;
}

/* DATA TABLE */

[data-testid="stDataFrame"] {
    border: 1px solid #344767;
    border-radius: 12px;
}

/* FOOTER */

</style>
""", unsafe_allow_html=True)


# ==========================================================
# 4. REUSABLE FUNCTIONS
# ==========================================================

def show_hero(title, description, tag):

    st.html(f"""
    <div style="
        display:flex;
        justify-content:space-between;
        align-items:center;
        gap:25px;
        flex-wrap:wrap;
        padding:36px;
        border-radius:22px;
        margin-bottom:28px;
        background:
        radial-gradient(circle at 80% 50%,#603BC5 0%,transparent 36%),
        linear-gradient(115deg,#14244B,#27194E,#34216D);
        border:1px solid #51449B;
        box-shadow:0 12px 40px rgba(70,50,160,.20);
    ">

        <div style="flex:1;min-width:260px;">
            <div class="hero-tag">{tag}</div>

            <div class="hero-title">{title}</div>

            <div class="hero-description">
                {description}
            </div>

            <div style="
                color:#C4B5FD;
                font-size:13px;
                font-weight:700;
                margin-top:22px;
                letter-spacing:1px;
            ">
                MACHINE LEARNING • EDUCATION ANALYTICS • EARLY SUPPORT
            </div>
        </div>

        <div style="width:190px;max-width:100%;">
            <svg viewBox="0 0 220 190"
                 width="100%"
                 xmlns="http://www.w3.org/2000/svg">

                <defs>
                    <linearGradient id="nexusgrad">
                        <stop offset="0%" stop-color="#60A5FA"/>
                        <stop offset="100%" stop-color="#A78BFA"/>
                    </linearGradient>
                </defs>

                <circle cx="110" cy="88" r="78"
                        fill="#FFFFFF" opacity=".05"/>

                <rect x="35" y="40" width="150" height="105"
                      rx="15" fill="#101A37"
                      stroke="#8170D8" stroke-width="2"/>

                <rect x="48" y="53" width="124" height="75"
                      rx="8" fill="#1C2A50"/>

                <path d="M60 108 L85 91 L108 98 L137 70 L159 80"
                      fill="none" stroke="url(#nexusgrad)"
                      stroke-width="4" stroke-linecap="round"/>

                <circle cx="85" cy="91" r="4" fill="#60A5FA"/>
                <circle cx="108" cy="98" r="4" fill="#A78BFA"/>
                <circle cx="137" cy="70" r="4" fill="#22D3EE"/>

                <path d="M90 145 L130 145 L138 160 L82 160 Z"
                      fill="#A78BFA"/>

                <circle cx="174" cy="35" r="18"
                        fill="#7C3AED" stroke="#C4B5FD"/>

                <path d="M174 25 L177 32 L184 35 L177 38 L174 45 L171 38 L164 35 L171 32 Z"
                      fill="white"/>
            </svg>
        </div>
    </div>
    """)


def show_kpi(icon, label, value, note):

    st.html(f"""
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-note">{note}</div>
    </div>
    """)


def create_dark_chart():

    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    fig.patch.set_facecolor("#111A2E")
    ax.set_facecolor("#111A2E")

    ax.tick_params(colors="#B8C5D9", labelsize=10)

    for spine in ax.spines.values():
        spine.set_color("#344767")

    ax.grid(
        axis="y",
        color="#344767",
        alpha=.45,
        linestyle="--"
    )

    return fig, ax


def chart_title(ax, title, ylabel):

    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        color="#F8FAFC",
        pad=18
    )

    ax.set_ylabel(ylabel, color="#A5B4CC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def show_footer():

    # Use st.html so HTML is rendered instead of displayed as code.

    st.html("""
    <div style="
        text-align:center;
        padding:28px 15px;
        margin-top:30px;
        border-top:1px solid #293A59;
        color:#94A3B8;
        font-family:Arial,sans-serif;
    ">
        <div style="
            color:#A78BFA;
            font-size:22px;
            font-weight:800;
            letter-spacing:2px;
            margin-bottom:8px;
        ">✦ NEXUS</div>

        <div style="
            font-size:14px;
            font-weight:600;
            margin-bottom:8px;
        ">STUDENT INTELLIGENCE PLATFORM</div>

        <div style="font-size:12px;color:#8292AC;">
            Machine Learning • Education Analytics • Early Support
        </div>

        <div style="
            font-size:11px;
            color:#64748B;
            margin-top:15px;
        ">Built with Python and Machine Learning</div>
    </div>
    """)



with st.sidebar:

    st.html("""
    <div style="
        padding:15px 0 25px;
        font-family:Arial,sans-serif;
    ">
        <div style="
            display:flex;
            align-items:center;
            gap:12px;
            color:#A78BFA;
            font-size:34px;
            font-weight:800;
            letter-spacing:3px;
        ">
            <span style="font-size:38px;">✦</span>
            <span>NEXUS</span>
        </div>

        <div style="
            color:#94A3B8;
            font-size:12px;
            font-weight:600;
            letter-spacing:1.5px;
            margin-top:8px;
            padding-left:4px;
        ">
            STUDENT INTELLIGENCE
        </div>
    </div>
    """)

    st.divider()

    st.markdown("### NAVIGATION")

    page = st.radio(
        "Choose a section",
        [
            "Overview",
            "Student Prediction",
            "Project Information"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### SYSTEM STATUS")

    st.markdown(
        "<span style='color:#34D399;font-weight:700;'>● MODEL ACTIVE</span>",
        unsafe_allow_html=True
    )

    st.caption("Algorithm: Logistic Regression")
    st.caption("Input features: 11")
    st.caption("Platform: Streamlit")

    st.divider()

    st.caption("NEXUS v1.0")
    st.caption("Academic Analytics Platform")


# ==========================================================
# 6. OVERVIEW PAGE
# ==========================================================

if page == "Overview":

    show_hero(
        "Student Early Support System",
        "Understand student engagement, detect changes in "
        "learning behaviour and identify students who may "
        "benefit from additional academic support.",
        "✦ AI POWERED EDUCATION ANALYTICS"
    )

    st.markdown("## Platform Overview")

    k1, k2, k3, k4 = st.columns(4, gap="medium")

    with k1:
        show_kpi("👨‍🎓", "Students Analysed", "32,593", "OULAD dataset")

    with k2:
        show_kpi("🎯", "Model Accuracy", "70.76%", "Held-out test data")

    with k3:
        show_kpi("🧠", "ML Features", "11", "Student activity inputs")

    with k4:
        show_kpi("⚙️", "Algorithms Tested", "03", "Model comparison")

    st.write("")

    # ------------------------------------------------------
    # MODEL PERFORMANCE
    # ------------------------------------------------------

    left, right = st.columns(2, gap="large")

    with left:

        with st.container(border=True):

            st.markdown("### 📊 Model Performance")
            st.caption("Final Logistic Regression test metrics")

            fig, ax = create_dark_chart()

            names = ["Accuracy", "Precision", "Recall", "F1 Score"]
            values = [70.76, 73.84, 68.63, 71.14]

            colors = ["#3B82F6", "#8B5CF6", "#06B6D4", "#A78BFA"]

            bars = ax.barh(names, values, color=colors, height=.58)

            ax.set_xlim(0, 100)

            for bar, value in zip(bars, values):
                ax.text(
                    value + 2,
                    bar.get_y() + bar.get_height()/2,
                    f"{value:.1f}%",
                    va="center",
                    color="white",
                    fontweight="bold"
                )

            ax.invert_yaxis()

            chart_title(ax, "Final Model Evaluation", "Score (%)")

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.caption("Precision, recall and F1-score refer to the Potential Support class.")

    with right:

        with st.container(border=True):

            st.markdown("### 📈 Learning Engagement")
            st.caption("Dataset-wide average daily clicks across four weeks")

            weeks = ["Week 1", "Week 2", "Week 3", "Week 4"]
            activity = [8.83, 7.88, 10.11, 7.10]

            fig, ax = create_dark_chart()

            ax.plot(
                weeks,
                activity,
                color="#A78BFA",
                linewidth=3.5,
                marker="o",
                markersize=10,
                markerfacecolor="#22D3EE"
            )

            ax.fill_between(
                range(4),
                activity,
                color="#8B5CF6",
                alpha=.17
            )

            chart_title(ax, "Weekly Activity Trend", "Average Clicks Per Day")

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.caption("Exploratory dataset-wide averages, not individual student activity.")

    # ------------------------------------------------------
    # SUPPORT DISTRIBUTION
    # ------------------------------------------------------

    st.write("")

    with st.container(border=True):

        st.markdown("### 🎯 Support Category Distribution")

        d1, d2 = st.columns([1, 1.4])

        with d1:

            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor("#111A2E")
            ax.set_facecolor("#111A2E")

            values = [15385, 17208]
            colors = ["#3B82F6", "#A78BFA"]

            wedges, _ = ax.pie(
                values,
                colors=colors,
                startangle=90,
                wedgeprops={"width": .38, "edgecolor": "#111A2E"}
            )

            ax.text(
                0, 0,
                "32,593\nStudents",
                ha="center",
                va="center",
                color="white",
                fontsize=15,
                fontweight="bold"
            )

            ax.legend(
                wedges,
                ["On Track", "Potential Support"],
                loc="lower center",
                bbox_to_anchor=(.5, -.15),
                ncol=2,
                frameon=False,
                labelcolor="#CBD5E1"
            )

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with d2:

            st.metric("On Track Students", "15,385", "47.2% of dataset")
            st.metric("Potential Support Students", "17,208", "52.8% of dataset")

            st.caption(
                "These are prepared dataset label counts, "
                "not live predictions or confirmed cases."
            )

    # ======================================================
    # 7. INTERACTIVE NEXUS WORKFLOW
    # ======================================================

    st.divider()

    st.markdown("## 🚀 Explore How NEXUS Works")

    st.caption(
        "Select each stage to explore the actual analysis, "
        "machine learning and support process."
    )

    tab1, tab2, tab3 = st.tabs([
        "📊 01 Activity Analysis",
        "🧠 02 Intelligence Engine",
        "🎯 03 Early Support"
    ])

    # ------------------------------------------------------
    # TAB 1: ACTIVITY ANALYSIS
    # ------------------------------------------------------

    with tab1:

        st.markdown("### Student Activity Analysis")

        st.write(
            "The system analyses student learning behaviour "
            "using engagement data collected from the OULAD dataset."
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("Students Analysed", "32,593")
        c2.metric("Learning Features", "11")
        c3.metric("Support Categories", "2")

        st.markdown("#### Learning Activity Parameters")

        activity_data = pd.DataFrame({
            "Parameter": [
                "Total Learning Clicks",
                "Active Learning Days",
                "Resources Accessed",
                "Weekly Engagement",
                "Activity Change"
            ],
            "Purpose": [
                "Measures student interaction",
                "Measures learning consistency",
                "Tracks resource usage",
                "Identifies weekly patterns",
                "Measures change in engagement"
            ]
        })

        st.dataframe(
            activity_data,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("#### Weekly Learning Engagement")

        fig, ax = create_dark_chart()

        ax.plot(
            weeks,
            activity,
            color="#A78BFA",
            marker="o",
            linewidth=3,
            markersize=9
        )

        ax.fill_between(
            range(4),
            activity,
            color="#8B5CF6",
            alpha=.15
        )

        chart_title(
            ax,
            "Average Weekly Learning Activity",
            "Average Daily Clicks"
        )

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ------------------------------------------------------
    # TAB 2: INTELLIGENCE ENGINE
    # ------------------------------------------------------

    with tab2:

        st.markdown("### Machine Learning Intelligence Engine")

        st.write(
            "Three classification algorithms were evaluated. "
            "Logistic Regression was selected as the final model."
        )

        comparison = pd.DataFrame({
            "Algorithm": [
                "Logistic Regression",
                "Decision Tree",
                "Random Forest"
            ],
            "Accuracy (%)": [70.76, 69.25, 68.24],
            "Precision (%)": [73.84, 75.90, 70.90],
            "Recall (%)": [68.63, 60.71, 67.04],
            "F1 Score (%)": [71.14, 67.46, 68.92]
        })

        st.markdown("#### Algorithm Comparison")

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("#### Model Accuracy Comparison")

        fig, ax = create_dark_chart()

        bars = ax.bar(
            comparison["Algorithm"],
            comparison["Accuracy (%)"],
            color=["#8B5CF6", "#3B82F6", "#06B6D4"],
            width=.55
        )

        ax.set_ylim(0, 100)

        for bar, value in zip(bars, comparison["Accuracy (%)"]):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                value + 2,
                f"{value:.2f}%",
                ha="center",
                color="white",
                fontweight="bold"
            )

        chart_title(
            ax,
            "Accuracy of Tested Algorithms",
            "Accuracy (%)"
        )

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.success("Selected final model: Logistic Regression")

        st.info(
            "The saved Logistic Regression model is used "
            "by the Student Prediction section."
        )

    # ------------------------------------------------------
    # TAB 3: EARLY SUPPORT
    # ------------------------------------------------------

    with tab3:

        st.markdown("### Early Academic Support")

        st.write(
            "After analysing learning activity, the model "
            "classifies students into two predicted categories."
        )

        c1, c2 = st.columns(2)

        with c1:

            st.markdown("#### 🟢 On Track")

            st.success(
                "The model predicts the On Track category."
            )

            st.write(
                "Continue monitoring student engagement "
                "and learning consistency."
            )

        with c2:

            st.markdown("#### 🟠 Potential Support")

            st.warning(
                "The model predicts the Potential Support category."
            )

            st.write(
                "Recommend educator review to determine "
                "whether additional academic assistance is appropriate."
            )

        st.markdown("#### Support Recommendation Workflow")

        workflow = pd.DataFrame({
            "Stage": [
                "1. Collect Activity",
                "2. Analyse Behaviour",
                "3. Generate Prediction",
                "4. Educator Review",
                "5. Support Intervention"
            ],
            "Description": [
                "Collect student learning activity",
                "Identify changes in engagement",
                "Run the trained ML model",
                "Review the student's individual situation",
                "Provide appropriate academic assistance"
            ]
        })

        st.dataframe(
            workflow,
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "Predictions are early-warning indicators, "
            "not confirmed academic difficulties. "
            "Educator judgement is essential."
        )

        if st.button(
            "Go to Student Prediction",
            use_container_width=True
        ):
            st.session_state["open_prediction"] = True
            st.success(
                "Now select Student Prediction from the sidebar."
            )

    # ------------------------------------------------------
    # END OF INTERACTIVE WORKFLOW
    # ------------------------------------------------------


# ==========================================================
# 8. STUDENT PREDICTION PAGE
# ==========================================================

elif page == "Student Prediction":

    show_hero(
        "Student Risk Assessment",
        "Enter student learning engagement information "
        "to generate a prediction using the trained ML model.",
        "✦ AI POWERED STUDENT ANALYSIS"
    )

    st.markdown("## Student Information")

    st.caption("Provide the student's course and learning activity details.")

    with st.container(border=True):

        with st.form("student_prediction_form"):

            col1, col2 = st.columns(2, gap="large")

            with col1:

                st.markdown("### Basic Information")

                student_id = st.text_input(
                    "Student ID",
                    value="STU001"
                )

                module = st.selectbox(
                    "Course Module",
                    MODULES
                )

                presentation = st.selectbox(
                    "Course Presentation",
                    PRESENTATIONS
                )

                total_clicks = st.number_input(
                    "Total Learning Clicks",
                    min_value=0,
                    value=250
                )

                active_days = st.number_input(
                    "Active Learning Days",
                    min_value=0,
                    max_value=31,
                    value=15
                )

            with col2:

                st.markdown("### Learning Engagement")

                resources = st.number_input(
                    "Resources Accessed",
                    min_value=0,
                    value=15
                )

                week1 = st.number_input(
                    "Week 1 Average Clicks Per Day",
                    min_value=0.0,
                    value=10.0
                )

                week2 = st.number_input(
                    "Week 2 Average Clicks Per Day",
                    min_value=0.0,
                    value=9.0
                )

                week3 = st.number_input(
                    "Week 3 Average Clicks Per Day",
                    min_value=0.0,
                    value=8.0
                )

                week4 = st.number_input(
                    "Week 4 Average Clicks Per Day",
                    min_value=0.0,
                    value=5.0
                )

            submitted = st.form_submit_button(
                "✦ GENERATE STUDENT PREDICTION",
                use_container_width=True
            )

    # INDIVIDUAL ACTIVITY GRAPH

    st.write("")

    st.markdown("## 📈 Student Engagement Analytics")

    weekly_data = pd.DataFrame({
        "Week": ["Week 1", "Week 2", "Week 3", "Week 4"],
        "Average Clicks": [week1, week2, week3, week4]
    })

    with st.container(border=True):

        fig, ax = create_dark_chart()

        ax.plot(
            weekly_data["Week"],
            weekly_data["Average Clicks"],
            marker="o",
            markersize=10,
            linewidth=3,
            color="#A78BFA",
            markerfacecolor="#22D3EE"
        )

        ax.fill_between(
            range(4),
            weekly_data["Average Clicks"],
            color="#8B5CF6",
            alpha=.15
        )

        chart_title(
            ax,
            "Individual Weekly Learning Trend",
            "Average Clicks Per Day"
        )

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    if week4 < week1:

        st.warning(
            f"Activity decreased by {week1 - week4:.2f} "
            "clicks per day between Week 1 and Week 4."
        )

    elif week4 > week1:

        st.success(
            "Activity increased between Week 1 and Week 4."
        )

    else:

        st.info("Activity remained unchanged.")

    # PREDICTION

    if submitted:

        if not student_id.strip():

            st.error("Please enter a valid Student ID.")

        else:

            clicks_per_day = (
                total_clicks / active_days
                if active_days > 0 else 0
            )

            activity_change = week4 - week1

            numeric_values = {
                "total_clicks": total_clicks,
                "active_days": active_days,
                "resources_accessed": resources,
                "clicks_per_active_day": clicks_per_day,
                "week1_rate": week1,
                "week2_rate": week2,
                "week3_rate": week3,
                "week4_rate": week4,
                "activity_change": activity_change
            }

            # Match the exact training feature order

            row = {}

            for feature in model_features:

                if feature in numeric_values:

                    row[feature] = numeric_values[feature]

                elif feature.startswith("code_module_"):

                    category = feature[len("code_module_"):]

                    row[feature] = int(category == str(module))

                elif feature.startswith("code_presentation_"):

                    category = feature[len("code_presentation_"):]

                    row[feature] = int(
                        category == str(presentation)
                    )

                else:

                    row[feature] = 0

            input_data = pd.DataFrame(
                [row],
                columns=model_features
            )

            # MODEL PREDICTION

            prediction = model.predict(input_data)[0]

            probabilities = model.predict_proba(input_data)[0]

            classes = list(model.classes_)

            support_probability = (
                probabilities[classes.index(1)] * 100
            )

            on_track_probability = (
                probabilities[classes.index(0)] * 100
            )

            st.divider()

            st.markdown("## 🎯 Prediction Result")

            st.write(f"**Student ID:** {student_id}")

            if prediction == 1:

                st.html(f"""
                <div class="risk-card">

                    <div style="
                        color:#FDA4AF;
                        font-size:13px;
                        font-weight:700;
                        letter-spacing:1px;
                    ">EARLY WARNING INDICATOR</div>

                    <div style="
                        color:white;
                        font-size:27px;
                        font-weight:800;
                        margin-top:12px;
                    ">⚠️ Potential Support Recommended</div>

                    <div style="
                        color:#F9A8D4;
                        font-size:42px;
                        font-weight:800;
                        margin:12px 0;
                    ">{support_probability:.1f}%</div>

                    <p style="color:#FCE7F3;">
                        Model probability for Potential Support.
                        Educator review is recommended.
                    </p>

                </div>
                """)

            else:

                st.html(f"""
                <div class="safe-card">

                    <div style="
                        color:#6EE7B7;
                        font-size:13px;
                        font-weight:700;
                        letter-spacing:1px;
                    ">CURRENT STATUS</div>

                    <div style="
                        color:white;
                        font-size:27px;
                        font-weight:800;
                        margin-top:12px;
                    ">✓ Currently On Track</div>

                    <div style="
                        color:#6EE7B7;
                        font-size:42px;
                        font-weight:800;
                        margin:12px 0;
                    ">{support_probability:.1f}%</div>

                    <p style="color:#D1FAE5;">
                        Model probability for Potential Support.
                        Continue monitoring engagement.
                    </p>

                </div>
                """)

            st.markdown("### Model Probability Distribution")

            prob_df = pd.DataFrame({
                "Category": ["On Track", "Potential Support"],
                "Probability": [
                    on_track_probability,
                    support_probability
                ]
            })

            with st.container(border=True):

                fig, ax = create_dark_chart()

                bars = ax.barh(
                    prob_df["Category"],
                    prob_df["Probability"],
                    color=["#3B82F6", "#A78BFA"],
                    height=.5
                )

                ax.set_xlim(0, 110)

                for bar, value in zip(
                    bars,
                    prob_df["Probability"]
                ):

                    ax.text(
                        value + 2,
                        bar.get_y() + bar.get_height()/2,
                        f"{value:.1f}%",
                        color="white",
                        va="center",
                        fontweight="bold"
                    )

                ax.invert_yaxis()

                chart_title(
                    ax,
                    "Predicted Class Probabilities",
                    "Probability (%)"
                )

                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            st.markdown("### Student Activity Summary")

            a, b, c, d = st.columns(4)

            a.metric("Total Clicks", total_clicks)
            b.metric("Active Days", active_days)
            c.metric("Resources Accessed", resources)
            d.metric("Clicks Per Active Day", f"{clicks_per_day:.1f}")

            st.caption(
                "Model probabilities are estimates, not confirmed "
                "measures of actual academic difficulty. "
                "Educator review is necessary."
            )


# ==========================================================
# 9. PROJECT INFORMATION PAGE
# ==========================================================

elif page == "Project Information":

    show_hero(
        "About NEXUS",
        "A machine learning based early-warning platform "
        "designed to analyse student engagement and "
        "highlight potential academic support needs.",
        "✦ PROJECT INFORMATION"
    )

    st.markdown("## 🎯 Problem Statement")

    with st.container(border=True):

        st.markdown("""
        **Detect students who may need additional support
        from changes in learning activity.**
        """)

        st.write(
            "The system uses learning activity patterns, "
            "weekly engagement and course information "
            "to identify potential support categories."
        )

    st.markdown("## 🧠 Machine Learning Algorithms")

    x, y, z = st.columns(3)

    with x:
        st.info("**Logistic Regression**\n\nSelected final model.")

    with y:
        st.info("**Decision Tree**\n\nAlternative classification model.")

    with z:
        st.info("**Random Forest**\n\nEnsemble classification model.")

    st.markdown("## ⚙️ Technologies Used")

    t1, t2 = st.columns(2)

    with t1:
        st.markdown("""
        - Python
        - Pandas
        - NumPy
        - Scikit-learn
        """)

    with t2:
        st.markdown("""
        - Matplotlib
        - Streamlit
        - Joblib
        - OULAD Dataset
        """)

    st.markdown("## 📚 Project Objective")

    st.write(
        "To identify changes in student learning engagement "
        "and provide educators with an early-warning indicator "
        "to help review students who may benefit from "
        "additional academic support."
    )

    st.info(
        "NEXUS is an academic decision-support prototype. "
        "It supports educator judgement and does not replace it."
    )


# ==========================================================
# 10. FOOTER - HTML RENDERED CORRECTLY
# ==========================================================

show_footer()
