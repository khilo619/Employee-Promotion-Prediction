import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.base import BaseEstimator, TransformerMixin

# --- CUSTOM TRANSFORMER DEFINITION ---
# This class must be present in the namespace for pickle to load the pipeline successfully.
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, log_cols=None):
        self.log_cols = log_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        log_cols = self.log_cols or []
        for col in log_cols:
            if col in X.columns:
                X[col] = np.log1p(X[col])

        education_mapping = {'Bachelor': 1, 'Master': 2, 'PhD': 3}
        city_tier_mapping = {'Tier1': 3, 'Tier2': 2, 'Tier3': 1}
        
        # Safe mapping for Streamlit inputs
        if 'education_level' in X.columns:
            X['education_level'] = X['education_level'].map(education_mapping)
        if 'city_tier' in X.columns:
            X['city_tier'] = X['city_tier'].map(city_tier_mapping)

        # Ration features
        X['promotion_rate'] = 1 / (X['years_since_last_promotion'] + 1)
        X['performance_efficiency'] = X['performance_score'] / (X['avg_monthly_hours'] + 1)
        X['leadership_tenure_ratio'] = X['leadership_score'] / (X['years_at_company'] + 1)
        X['skill_cert_ratio'] = X['skill_assessment_score'] / (X['certifications_count'] + 1)
        return X

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Employee Promotion Predictor", layout="wide")

@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    artifact = load_model()
    pipeline = artifact['pipeline']
    threshold = artifact['threshold']
except FileNotFoundError:
    st.error("Model artifact not found. Please ensure 'model_and_preprocessing_artifacts.pkl' is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- UI LAYOUT ---
st.title("🚀 Employee Promotion Prediction")
st.markdown("""
Predict the likelihood of an employee's promotion based on their performance, demographics, and behavioral metrics.
This model uses a Linear SVM with a custom decision threshold optimized for F1-score.
""")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["f", "m"])
        marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
        age = st.number_input("Age", 18, 65, 30)
        education_level = st.selectbox("Education Level", ["Bachelor", "Master", "PhD"])
        city_tier = st.selectbox("City Tier", ["Tier1", "Tier2", "Tier3"])

    with col2:
        st.subheader("Company Stats")
        department = st.selectbox("Department", ["Sales", "HR", "Development", "Data Science", "Finance", "Legal", "Marketing"])
        employment_type = st.selectbox("Employment Type", ["Full-time", "Contract", "Freelance"])
        years_at_company = st.number_input("Years at Company", 0, 40, 3)
        years_in_current_role = st.number_input("Years in Role", 0, 20, 2)
        years_since_last_promotion = st.number_input("Years Since Last Promotion", 0, 20, 1)
        salary = st.number_input("Monthly Salary", 1000, 20000, 5000)
        salary_increase_percent = st.slider("Salary Increase (%)", 0, 50, 5)

    with col3:
        st.subheader("Performance Indicators")
        performance_score = st.slider("Current Performance Score", 1, 10, 5)
        manager_rating = st.slider("Manager Rating", 1, 5, 3)
        kpi_achievement_percent = st.slider("KPI Achievement (%)", 0, 100, 80)
        attendance_rate = st.slider("Attendance Rate (%)", 50, 100, 95)
        projects_completed = st.number_input("Projects Completed", 0, 50, 5)
        avg_monthly_hours = st.number_input("Avg Monthly Hours", 80, 300, 160)
        
    st.subheader("Additional Metrics")
    c1, c2, c3 = st.columns(3)
    with c1:
        innovation_score = st.slider("Innovation Score", 1, 10, 5)
        leadership_score = st.slider("Leadership Score", 1, 10, 5)
        problem_solving_score = st.slider("Problem Solving Score", 1, 10, 5)
    with c2:
        peer_feedback_score = st.slider("Peer Feedback Score", 1, 5, 3)
        training_hours_last_year = st.number_input("Training Hours", 0, 200, 20)
        certifications_count = st.number_input("Certifications", 0, 10, 1)
    with c3:
        team_size = st.number_input("Team Size", 1, 50, 10)
        late_days = st.number_input("Late Days", 0, 30, 0)
        job_satisfaction_score = st.slider("Job Satisfaction", 1, 5, 4)

    st.divider()
    b1, b2, b3, b4 = st.columns(4)
    with b1: bonus_last_year = st.checkbox("Received Bonus Last Year?", value=False)
    with b2: stock_options = st.checkbox("Has Stock Options?", value=False)
    with b3: remote_work_ratio = st.slider("Remote Work Ratio", 0.0, 1.0, 0.4)
    with b4: internal_mobility_score = st.slider("Internal Mobility Score", 1, 10, 5)

    # Fixed defaults for non-exposed features to match model input shape
    input_data = {
        'gender': gender,
        'marital_status': marital_status,
        'age': age,
        'education_level': education_level,
        'city_tier': city_tier,
        'department': department,
        'employment_type': employment_type,
        'years_at_company': years_at_company,
        'years_in_current_role': years_in_current_role,
        'years_since_last_promotion': years_since_last_promotion,
        'salary': salary,
        'salary_increase_percent': salary_increase_percent,
        'performance_score': performance_score,
        'manager_rating': manager_rating,
        'kpi_achievement_percent': kpi_achievement_percent,
        'attendance_rate': attendance_rate,
        'projects_completed': projects_completed,
        'avg_monthly_hours': avg_monthly_hours,
        'innovation_score': innovation_score,
        'leadership_score': leadership_score,
        'problem_solving_score': problem_solving_score,
        'peer_feedback_score': peer_feedback_score,
        'training_hours_last_year': training_hours_last_year,
        'certifications_count': certifications_count,
        'team_size': team_size,
        'late_days': late_days,
        'job_satisfaction_score': job_satisfaction_score,
        'bonus_last_year': 1 if bonus_last_year else 0,
        'stock_options': 1 if stock_options else 0,
        'remote_work_ratio': remote_work_ratio,
        'internal_mobility_score': internal_mobility_score,
        # Defaults for features not in form
        'performance_last_year': 3, 
        'performance_two_years_ago': 3,
        'overtime_hours': 10,
        'tasks_completed': 50,
        'deadline_adherence_rate': 90,
        'meeting_hours_per_month': 20,
        'skill_assessment_score': 70,
        'cross_department_projects': 1,
        'mentoring_sessions': 2,
        'employee_engagement_score': 4,
    }

    submit = st.form_submit_button("Predict Promotion Likelihood")

# --- PREDICTION LOGIC ---
if submit:
    df = pd.DataFrame([input_data])
    
    # Get decision score
    decision_score = pipeline.decision_function(df)[0]
    is_promoted = decision_score > threshold

    st.divider()
    if is_promoted:
        st.success(f"### Result: PROMOTED 🎉")
        st.info(f"The employee is likely to be promoted based on current metrics. (Decision Score: {decision_score:.3f} | Threshold: {threshold:.3f})")
    else:
        st.warning(f"### Result: NOT PROMOTED")
        st.info(f"No promotion predicted. (Decision Score: {decision_score:.3f} | Threshold: {threshold:.3f})")

    st.info("Note: Prediction is based on a Linear SVC model with optimized F1-score thresholding.")
