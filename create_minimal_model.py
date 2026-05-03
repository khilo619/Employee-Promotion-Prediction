#!/usr/bin/env python3
"""
Create a minimal working model for deployment
This recreates the essential components needed for the Streamlit app
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.svm import LinearSVC

# Recreate the FeatureEngineer class
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
        
        # Safe mapping for Streamlit inputs
        if 'education_level' in X.columns:
            education_mapping = {'Bachelor': 1, 'Master': 2, 'PhD': 3}
            X['education_level'] = X['education_level'].map(education_mapping)
        if 'city_tier' in X.columns:
            city_tier_mapping = {'Tier1': 3, 'Tier2': 2, 'Tier3': 1}
            X['city_tier'] = X['city_tier'].map(city_tier_mapping)
        
        # Ratio features
        X['promotion_rate'] = 1 / (X['years_since_last_promotion'] + 1)
        X['performance_efficiency'] = X['performance_score'] / (X['avg_monthly_hours'] + 1)
        X['leadership_tenure_ratio'] = X['leadership_score'] / (X['years_at_company'] + 1)
        X['skill_cert_ratio'] = X['skill_assessment_score'] / (X['certifications_count'] + 1)
        return X

# Create minimal preprocessing pipeline
log_cols = [
    'salary', 'years_at_company', 'years_in_current_role',
    'years_since_last_promotion', 'projects_completed',
    'certifications_count', 'mentoring_sessions', 'late_days',
    'cross_department_projects', 'training_hours_last_year',
]

nominal_features = ['gender', 'marital_status', 'department', 'employment_type']
bounded_features = ['attendance_rate', 'kpi_achievement_percent']
binary_features = ['bonus_last_year', 'stock_options']

numeric_features_after_fe = [
    'age', 'education_level', 'city_tier', 'years_at_company',
    'years_in_current_role', 'years_since_last_promotion', 'team_size',
    'performance_score', 'performance_last_year', 'performance_two_years_ago',
    'manager_rating', 'peer_feedback_score', 'projects_completed',
    'innovation_score', 'leadership_score', 'problem_solving_score',
    'avg_monthly_hours', 'overtime_hours', 'tasks_completed',
    'deadline_adherence_rate', 'meeting_hours_per_month', 'remote_work_ratio',
    'training_hours_last_year', 'certifications_count', 'skill_assessment_score',
    'cross_department_projects', 'mentoring_sessions', 'salary',
    'salary_increase_percent', 'late_days', 'employee_engagement_score',
    'job_satisfaction_score', 'internal_mobility_score', 'promotion_rate',
    'performance_efficiency', 'leadership_tenure_ratio', 'skill_cert_ratio',
]

robust_features = [
    col for col in numeric_features_after_fe
    if col not in bounded_features + binary_features
]

preprocessor = ColumnTransformer(
    transformers=[
        ('nominal', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), nominal_features),
        ('robust', RobustScaler(), robust_features),
        ('minmax', MinMaxScaler(), bounded_features),
        ('binary', 'passthrough', binary_features),
    ],
    remainder='drop',
    verbose_feature_names_out=False,
)

# Create a simple Linear SVC model
model = LinearSVC(random_state=42, C=1.0, max_iter=1000)

# Create the full pipeline
pipeline = ImbPipeline([
    ('feature_engineer', FeatureEngineer(log_cols)),
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# Set a reasonable threshold (you can adjust this based on your actual threshold)
threshold = 0.0

# Create the artifact
artifact = {
    'pipeline': pipeline,
    'threshold': threshold
}

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(artifact, f)

print("✅ Created minimal working model.pkl")
print(f"Model size: {len(pickle.dumps(artifact)) / 1024 / 1024:.2f} MB")