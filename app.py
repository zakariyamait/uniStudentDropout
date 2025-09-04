# --- Library Version Notes ---
# This simplified app removes the need for the 'imbalanced-learn' library,
# which can cause installation issues.
# Recommended versions:
# pip install scikit-learn
# pip install shap
# pip install "numpy<2.3"
# pip install streamlit-shap (for better SHAP integration)
# -----------------------------

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import os
import streamlit.components.v1 as components

# Set matplotlib backend for better Streamlit compatibility
plt.switch_backend('Agg')

# --- Define columns to exclude from UI ---
EXCLUDED_COLUMNS = [
    'Unemployment rate',
    'Inflation rate', 
    'GDP',
    'Marital status',
    'Application mode',
    'Daytime/evening attendance',
    "Mother's qualification",
    "Father's qualification",
    'Debtor',
    'Tuition fees up to date',
    'Displaced'
]

# --- Define Input Choices for Each Variable ---
INPUT_CHOICES = {
    # Categorical Variables
    'Marital status': {1: 'Single', 2: 'Married', 3: 'Widower', 4: 'Divorced', 5: 'Facto union', 6: 'Legally separated'},
    'Application mode': {
        1: '1st phase - general contingent',
        2: 'Ordinance No. 612/93',
        5: '1st phase - special contingent (Azores Island)',
        7: 'Holders of other higher courses',
        10: 'Ordinance No. 854-B/99',
        15: 'International student (bachelor)',
        16: '1st phase - special contingent (Madeira Island)',
        17: '2nd phase - general contingent',
        18: '3rd phase - general contingent',
        26: 'Ordinance No. 533-A/99, item b2) (Different Plan)',
        27: 'Ordinance No. 533-A/99, item b3 (Other Institution)',
        39: 'Over 23 years old',
        42: 'Transfer',
        43: 'Change of course',
        44: 'Technological specialization diploma holders',
        51: 'Change of institution/course',
        53: 'Short cycle diploma holders',
        57: 'Change of institution/course (International)'
    },
    'Application order': {0: 'First choice', 1: 'Not first choice (between 1-9)'},
    'Course': {
        33: 'Biofuel Production Technologies',
        171: 'Animation and Multimedia Design',
        8014: 'Social Service (evening attendance)',
        9003: 'Agronomy',
        9070: 'Communication Design',
        9085: 'Veterinary Nursing',
        9119: 'Informatics Engineering',
        9130: 'Equinculture',
        9147: 'Management',
        9238: 'Social Service',
        9254: 'Tourism',
        9500: 'Nursing',
        9556: 'Oral Hygiene',
        9670: 'Advertising and Marketing Management',
        9773: 'Journalism and Communication',
        9853: 'Basic Education',
        9991: 'Management (evening attendance)'
    },
    'Daytime/evening attendance': {1: 'Daytime', 0: 'Evening'},
    'Previous qualification': {
        1: 'Secondary education',
        2: 'Higher education - bachelor\'s degree',
        3: 'Higher education - degree',
        4: 'Higher education - master\'s',
        5: 'Higher education - doctorate',
        6: 'Frequency of higher education',
        9: '12th year of schooling - not completed',
        10: '11th year of schooling - not completed',
        12: 'Other - 11th year of schooling',
        14: '10th year of schooling',
        15: '10th year of schooling - not completed',
        19: 'Basic education 3rd cycle (9th/10th/11th year)',
        38: 'Basic education 2nd cycle (6th/7th/8th year)',
        39: 'Technological specialization course',
        40: 'Higher education - degree (1st cycle)',
        42: 'Professional higher technical course',
        43: 'Higher education - master (2nd cycle)'
    },
    "Mother's qualification": {
        1: 'Secondary education - 12th Year of Schooling or Equivalent',
        2: 'Higher education - bachelor\'s degree',
        3: 'Higher education - degree',
        4: 'Higher education - master\'s',
        5: 'Higher education - doctorate',
        6: 'Frequency of higher education',
        9: '12th year of schooling - not completed',
        10: '11th year of schooling - not completed',
        11: '7th year (old)',
        12: 'Other - 11th year of schooling',
        14: '10th year of schooling',
        18: 'General commerce course',
        19: 'Basic education 3rd cycle (9th/10th/11th year) or equivalent',
        22: 'Technical-professional course',
        26: '7th year of schooling',
        27: '2nd cycle of general high school course',
        29: '9th year of schooling - not completed',
        30: '8th year of schooling',
        34: 'Unknown',
        35: 'Can\'t read or write',
        36: 'Can read without having a 4th year of schooling',
        37: 'Basic education 1st cycle (4th/5th/6th year) or equivalent',
        38: 'Basic education 2nd cycle (6th/7th/8th year) or equivalent',
        39: 'Technological specialization course',
        40: 'Higher education - degree (1st cycle)',
        41: 'Specialized higher studies course',
        42: 'Professional higher technical course',
        43: 'Higher education - master (2nd cycle)',
        44: 'Higher education - doctorate (3rd cycle)'
    },
    "Father's qualification": {
        1: 'Secondary education - 12th Year of Schooling or Equivalent',
        2: 'Higher education - bachelor\'s degree',
        3: 'Higher education - degree',
        4: 'Higher education - master\'s',
        5: 'Higher education - doctorate',
        6: 'Frequency of higher education',
        9: '12th year of schooling - not completed',
        10: '11th year of schooling - not completed',
        11: '7th year (old)',
        12: 'Other - 11th year of schooling',
        13: '2nd year complementary high school course',
        14: '10th year of schooling',
        18: 'General commerce course',
        19: 'Basic education 3rd cycle (9th/10th/11th year) or equivalent',
        20: 'Complementary High School Course',
        22: 'Technical-professional course',
        25: 'Complementary High School Course - not concluded',
        26: '7th year of schooling',
        27: '2nd cycle of general high school course',
        29: '9th year of schooling - not completed',
        30: '8th year of schooling',
        31: 'General Course of Administration and Commerce',
        33: 'Supplementary Accounting and Administration',
        34: 'Unknown',
        35: 'Can\'t read or write',
        36: 'Can read without having a 4th year of schooling',
        37: 'Basic education 1st cycle (4th/5th/6th year) or equivalent',
        38: 'Basic education 2nd cycle (6th/7th/8th year) or equivalent',
        39: 'Technological specialization course',
        40: 'Higher education - degree (1st cycle)',
        41: 'Specialized higher studies course',
        42: 'Professional higher technical course',
        43: 'Higher education - master (2nd cycle)',
        44: 'Higher education - doctorate (3rd cycle)'
    },
    "Mother's occupation": {0: 'Student', 1: 'Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers', 2: 'Specialists in Intellectual and Scientific Activities', 3: 'Intermediate Level Technicians and Professions', 4: 'Administrative staff', 5: 'Personal Services, Security and Safety Workers and Sellers', 6: 'Farmers and Skilled Workers in Agriculture, Fisheries and Forestry', 7: 'Skilled Workers in Industry, Construction and Craftsmen', 8: 'Installation and Machine Operators and Assembly Workers', 9: 'Unskilled Workers', 10: 'Armed Forces Professions', 90: 'Other Situation', 99: '(blank) Not Classified', 122: 'Health professionals', 123: 'Teachers', 125: 'Specialists in information and communication technologies', 131: 'Intermediate level science and engineering technicians and professions', 132: 'Technicians and associate professionals of health', 134: 'Intermediate level technicians from legal, social, sports, cultural and similar services', 141: 'Office workers, receptionists and similar workers', 143: 'Data, accounting, statistical, financial services and registry-related workers', 144: 'Other administrative support staff', 151: 'Personal service workers', 152: 'Sellers', 153: 'Personal care workers and the like', 171: 'Skilled construction workers and the like, except electricians', 173: 'Skilled workers in printing, precision instrument manufacturing, jewelers, artisans and the like', 175: 'Workers in food processing, wood, clothing and other industries and crafts', 191: 'Cleaning workers', 192: 'Unskilled workers in agriculture, animal production, fisheries and forestry', 193: 'Unskilled workers in extractive industry, construction, manufacturing and transport', 194: 'Meal preparation assistants'},
    "Father's occupation": {0: 'Student', 1: 'Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers', 2: 'Specialists in Intellectual and Scientific Activities', 3: 'Intermediate Level Technicians and Professions', 4: 'Administrative staff', 5: 'Personal Services, Security and Safety Workers and Sellers', 6: 'Farmers and Skilled Workers in Agriculture, Fisheries and Forestry', 7: 'Skilled Workers in Industry, Construction and Craftsmen', 8: 'Installation and Machine Operators and Assembly Workers', 9: 'Unskilled Workers', 10: 'Armed Forces Professions', 90: 'Other Situation', 99: '(blank) Not Classified', 101: 'Armed Forces Officers', 102: 'Armed Forces Sergeants', 103: 'Other Armed Forces personnel', 112: 'Directors of administrative and commercial services', 114: 'Hotel, catering, trade and other services directors', 121: 'Specialists in the physical sciences, mathematics, engineering and related techniques', 122: 'Health professionals', 123: 'Teachers', 124: 'Specialists in finance, accounting, administrative organization, public and commercial relations', 131: 'Intermediate level science and engineering technicians and professions', 132: 'Technicians and associate professionals of health', 134: 'Intermediate level technicians from legal, social, sports, cultural and similar services', 135: 'Information and communication technology technicians', 141: 'Office workers, receptionists and similar workers', 143: 'Data, accounting, statistical, financial services and registry-related workers', 144: 'Other administrative support staff', 151: 'Personal service workers', 152: 'Sellers', 153: 'Personal care workers and the like', 154: 'Protection and security services personnel', 161: 'Market-oriented farmers and skilled agricultural and animal production workers', 163: 'Farmers, livestock keepers, fishermen, hunters and gatherers, subsistence', 171: 'Skilled construction workers and the like, except electricians', 172: 'Skilled workers in metallurgy, metalworking and related trades', 174: 'Skilled workers in electricity and electronics', 175: 'Workers in food processing, wood, clothing and other industries and crafts', 181: 'Fixed plant and machine operators', 182: 'Assembly workers', 183: 'Vehicle drivers and mobile equipment operators', 192: 'Unskilled workers in agriculture, animal production, fisheries and forestry', 193: 'Unskilled workers in extractive industry, construction, manufacturing and transport', 194: 'Meal preparation assistants', 195: 'Street vendors (except food) and street service providers'},
    'Displaced': {1: 'Yes', 0: 'No'},
    'Educational special needs': {1: 'Yes', 0: 'No'},
    'Debtor': {1: 'Yes', 0: 'No'},
    'Tuition fees up to date': {1: 'Yes', 0: 'No'},
    'Gender': {1: 'Male', 0: 'Female'},
    'Scholarship holder': {1: 'Yes', 0: 'No'},
    'International': {1: 'Yes', 0: 'No'}
}

# --- Define Feature Engineering Function ---
def create_all_features(df):
    """Creates all basic and advanced features."""
    processed_df = df.copy()
    epsilon = 1e-6
    processed_df['1st_sem_pass_rate'] = processed_df['Curricular units 1st sem (approved)'] / (processed_df['Curricular units 1st sem (enrolled)'] + epsilon)
    processed_df['2nd_sem_pass_rate'] = processed_df['Curricular units 2nd sem (approved)'] / (processed_df['Curricular units 2nd sem (enrolled)'] + epsilon)
    processed_df['overall_pass_rate'] = (processed_df['Curricular units 1st sem (approved)'] + processed_df['Curricular units 2nd sem (approved)']) / (processed_df['Curricular units 1st sem (enrolled)'] + processed_df['Curricular units 2nd sem (enrolled)'] + epsilon)
    processed_df['performance_decline'] = processed_df['Curricular units 1st sem (grade)'] - processed_df['Admission grade']
    processed_df['financial_strain_flag'] = ((processed_df['Scholarship holder'] == 0) & (processed_df['Tuition fees up to date'] == 0)).astype(int)
    processed_df['academic_overload'] = (processed_df['Curricular units 1st sem (enrolled)'] + processed_df['Curricular units 2nd sem (enrolled)']) - (processed_df['Curricular units 1st sem (approved)'] + processed_df['Curricular units 2nd sem (approved)'])
    return processed_df

# --- Train Model and Preprocessors on First Run ---
@st.cache_resource
def train_model_and_preprocessors():
    """
    Loads data, fits the preprocessors (scaler, encoder), and trains the final model.
    """
    data_filename = 'train.csv' # Using train.csv as discussed
    if not os.path.exists(data_filename):
        st.error(f"Error: '{data_filename}' not found. Please ensure it's in the same directory as app.py.")
        return None, None, None, None, None, None, None

    df = pd.read_csv(data_filename)
    df_featured = create_all_features(df)
    
    X = df_featured.drop('Target', axis=1)
    y = df_featured['Target']

    numerical_cols = X.select_dtypes(include=np.number).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    # Fit Scaler and Encoder
    scaler = StandardScaler().fit(X[numerical_cols])
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(X[categorical_cols])
    
    # Transform the entire dataset
    X_num_scaled = scaler.transform(X[numerical_cols])
    X_cat_encoded = encoder.transform(X[categorical_cols])
    X_processed = np.concatenate([X_num_scaled, X_cat_encoded], axis=1)
    
    # Train the final model using the built-in class_weight parameter
    # This replaces the need for SMOTE and the imblearn library
    model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(X_processed, y)
    
    # Get feature names after encoding - ensure proper alignment
    numerical_feature_names = list(numerical_cols)
    categorical_feature_names = list(encoder.get_feature_names_out(categorical_cols))
    feature_names = numerical_feature_names + categorical_feature_names
    
    # Verify alignment
    expected_length = X_processed.shape[1]
    actual_length = len(feature_names)
    
    if expected_length != actual_length:
        st.warning(f"Feature name length mismatch: expected {expected_length}, got {actual_length}")
        # Fallback: create generic feature names
        feature_names = [f'feature_{i}' for i in range(expected_length)]
    
    return model, scaler, encoder, numerical_cols, categorical_cols, feature_names, X_processed.shape[1]

# --- Load Assets ---
model, scaler, encoder, numerical_cols, categorical_cols, feature_names, n_features = train_model_and_preprocessors()

# --- Create the User Interface (UI) ---
st.set_page_config(page_title="Student Success Predictor", page_icon="🎓", layout="wide")
st.title('🎓 Student Success Predictor & Explainer')

if model is None:
    st.stop()

# Load a sample of the data for the UI inputs, excluding specified columns
df_sample = pd.read_csv('train.csv').drop(['Target'] + [col for col in EXCLUDED_COLUMNS if col in pd.read_csv('train.csv').columns], axis=1)

st.sidebar.header('Student Information Input')
input_data = {}

# Create input fields only for non-excluded columns
for col in df_sample.columns:
    if col not in EXCLUDED_COLUMNS:
        if col in INPUT_CHOICES:
            # Use predefined choices for categorical variables
            choices = INPUT_CHOICES[col]
            choice_labels = [f"{key}: {value}" for key, value in choices.items()]
            selected = st.sidebar.selectbox(f'Select {col}', options=choice_labels)
            # Extract the key (code) from the selection
            input_data[col] = int(selected.split(':')[0])
        elif col in numerical_cols:
            # Use number input for continuous variables
            if col == 'Age at enrollment':
                input_data[col] = st.sidebar.number_input(f'Enter {col}', 
                                                        min_value=16, max_value=70, 
                                                        value=int(df_sample[col].mean()))
            elif col == 'Admission grade':
                input_data[col] = st.sidebar.number_input(f'Enter {col}', 
                                                        min_value=0.0, max_value=200.0, 
                                                        value=float(df_sample[col].mean()))
            elif col == 'Previous qualification (grade)':
                input_data[col] = st.sidebar.number_input(f'Enter {col}', 
                                                        min_value=0.0, max_value=200.0, 
                                                        value=float(df_sample[col].mean()))
            elif 'Curricular units' in col and 'enrolled' in col:
                input_data[col] = st.sidebar.number_input(f'Enter {col}', 
                                                        min_value=0, max_value=30, 
                                                        value=int(df_sample[col].mean()))
            elif 'Curricular units' in col and ('approved' in col or 'grade' in col or 'evaluations' in col):
                input_data[col] = st.sidebar.number_input(f'Enter {col}', 
                                                        min_value=0.0, max_value=20.0, 
                                                        value=float(df_sample[col].mean()))
            elif col in ['Unemployment rate', 'Inflation rate']:
                input_data[col] = st.sidebar.number_input(f'Enter {col}', 
                                                        min_value=0.0, max_value=100.0, 
                                                        value=float(df_sample[col].mean()))
            elif col == 'GDP':
                input_data[col] = st.sidebar.number_input(f'Enter {col}', 
                                                        min_value=-10.0, max_value=10.0, 
                                                        value=float(df_sample[col].mean()))
            else:
                input_data[col] = st.sidebar.number_input(f'Enter {col}', 
                                                        value=float(df_sample[col].mean()))
        else:
            # Fallback for any other categorical columns not in INPUT_CHOICES
            unique_vals = df_sample[col].unique()
            input_data[col] = st.sidebar.selectbox(f'Select {col}', options=unique_vals)

# Set default values for excluded columns (using training data means/modes)
df_full = pd.read_csv('train.csv')
for col in EXCLUDED_COLUMNS:
    if col in df_full.columns:
        if col in INPUT_CHOICES:
            # Use the first key from predefined choices
            input_data[col] = list(INPUT_CHOICES[col].keys())[0]
        elif df_full[col].dtype in ['int64', 'float64']:
            input_data[col] = float(df_full[col].mean())
        else:
            input_data[col] = df_full[col].mode()[0] if len(df_full[col].mode()) > 0 else df_full[col].iloc[0]

# --- Prediction and Explanation Logic ---
if st.sidebar.button('Predict Student Outcome'):
    input_df = pd.DataFrame([input_data])
    
    # Manually apply all the same steps to the input data
    input_df_featured = create_all_features(input_df)
    input_num_scaled = scaler.transform(input_df_featured[numerical_cols])
    input_cat_encoded = encoder.transform(input_df_featured[categorical_cols])
    input_processed = np.concatenate([input_num_scaled, input_cat_encoded], axis=1)

    # Get prediction
    prediction = model.predict(input_processed)[0]
    prediction_proba = model.predict_proba(input_processed)

    st.header('Prediction Result')
    if prediction == 'Dropout':
        st.error(f'Predicted Outcome: **{prediction}**')
    elif prediction == 'Enrolled':
        st.warning(f'Predicted Outcome: **{prediction}**')
    else:
        st.success(f'Predicted Outcome: **{prediction}**')

    st.write("Prediction Probabilities:")
    st.write(pd.DataFrame(prediction_proba, columns=model.classes_, index=["Probability"]))

    # --- SHAP Explanation ---
    st.header('Why did the model make this prediction?')
    
    # First, try a simple feature importance approach as backup
    try:
        # Get model's built-in feature importance
        model_importance = model.feature_importances_
        
        # Ensure feature names alignment
        if len(feature_names) != len(model_importance):
            aligned_feature_names = [f'Feature_{i}' for i in range(len(model_importance))]
        else:
            aligned_feature_names = feature_names
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': aligned_feature_names,
            'importance': model_importance,
        }).sort_values('importance', ascending=False).head(10)
        
        st.write("📊 **Model Feature Importance (Top 10)**")
        st.write("*These features are most important to the model overall*")
        
        for _, row in importance_df.iterrows():
            st.write(f"**{row['feature']}**: {row['importance']:.4f}")
        
        # Try to create a plot
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.barh(range(len(importance_df)), importance_df['importance'], color='steelblue', alpha=0.7)
            ax.set_yticks(range(len(importance_df)))
            ax.set_yticklabels(importance_df['feature'], fontsize=8)
            ax.set_xlabel('Feature Importance')
            ax.set_title('Top 10 Most Important Features (Global)')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, importance_df['importance'])):
                ax.text(val + 0.001, i, f'{val:.3f}', ha='left', va='center', fontsize=7)
            
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)
            plt.close()
        except Exception as plot_error:
            st.write(f"Plot error: {plot_error}")
        
    except Exception as importance_error:
        st.error(f"Could not get feature importance: {importance_error}")
    
    # Now try SHAP with a completely different approach
    st.write("---")
    st.write("🔍 **SHAP Analysis (Instance-Specific Explanation)**")
    
    try:
        explainer = shap.TreeExplainer(model)
        
        # Try to get SHAP values with extensive debugging
        st.write("Attempting SHAP calculation...")
        shap_values = explainer.shap_values(input_processed)
        
        st.write("✅ SHAP values calculated successfully!")
        st.write(f"SHAP values type: {type(shap_values)}")
        
        # Handle the SHAP values based on their actual structure
        if isinstance(shap_values, list):
            st.write(f"SHAP values is a list with {len(shap_values)} elements")
            for i, sv in enumerate(shap_values):
                st.write(f"Element {i} shape: {sv.shape}")
            
            # Try to use the first element or find the right one
            if len(shap_values) == 1:
                # Binary classification case
                current_shap_vals = shap_values[0][0]  # First class, first instance
                st.write("Using binary classification approach")
            else:
                # Multi-class case - try to find the predicted class
                try:
                    class_names = list(model.classes_)
                    if prediction in class_names:
                        class_idx = class_names.index(prediction)
                        current_shap_vals = shap_values[class_idx][0]
                        st.write(f"Using class index {class_idx} for prediction '{prediction}'")
                    else:
                        current_shap_vals = shap_values[0][0]  # Fallback
                        st.write("Using fallback: first class")
                except:
                    current_shap_vals = shap_values[0][0]  # Fallback
                    st.write("Using fallback approach")
        else:
            # Single array case
            st.write(f"SHAP values shape: {shap_values.shape}")
            current_shap_vals = shap_values[0]  # First instance
        
        st.write(f"Selected SHAP values shape: {current_shap_vals.shape}")
        
        # Align feature names
        if len(feature_names) != len(current_shap_vals):
            st.warning(f"Feature name mismatch: {len(feature_names)} names vs {len(current_shap_vals)} SHAP values")
            aligned_feature_names = [f'Feature_{i}' for i in range(len(current_shap_vals))]
        else:
            aligned_feature_names = feature_names
            st.write("✅ Feature names aligned correctly")
        
        # Create SHAP importance DataFrame
        shap_importance = pd.DataFrame({
            'feature': aligned_feature_names,
            'shap_value': current_shap_vals,
            'abs_shap_value': np.abs(current_shap_vals)
        }).sort_values('abs_shap_value', ascending=False).head(10)
        
        st.subheader("Top 10 Features Affecting This Prediction")
        for _, row in shap_importance.iterrows():
            impact = "🔴 Pushes toward positive class" if row['shap_value'] > 0 else "🟢 Pushes toward negative class"
            st.write(f"**{row['feature']}**: {row['shap_value']:.4f} - {impact}")
        
        # Try SHAP plot
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            colors = ['red' if x > 0 else 'green' for x in shap_importance['shap_value']]
            bars = ax.barh(range(len(shap_importance)), shap_importance['shap_value'], color=colors, alpha=0.7)
            ax.set_yticks(range(len(shap_importance)))
            ax.set_yticklabels(shap_importance['feature'], fontsize=8)
            ax.set_xlabel('SHAP Value (Impact on This Prediction)')
            ax.set_title(f'SHAP Feature Impact for: {prediction}')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, shap_importance['shap_value'])):
                ax.text(val + (0.01 if val > 0 else -0.01), i, f'{val:.3f}', 
                       ha='left' if val > 0 else 'right', va='center', fontsize=7)
            
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)
            plt.close()
            
            st.write("✅ SHAP visualization completed!")
            
        except Exception as plot_error:
            st.error(f"SHAP plot error: {plot_error}")
        
    except Exception as e:
        st.error(f"SHAP calculation failed: {str(e)}")
        st.write("Using feature importance as explanation instead.")
    
else:
    st.info("Please enter student data in the sidebar and click 'Predict' to see the results.")
    
    # Display information about the interface
    st.write("### About this interface:")
    st.write(f"**Excluded from UI (automatically set):** {', '.join(EXCLUDED_COLUMNS)}")
    st.write("**Interactive Variables:** All other variables are available for input in the sidebar.")
    
    # Show some statistics about the categorical variables
    st.write("### Variable Information:")
    st.write("- **Categorical variables** have predefined choices based on the data dictionary")
    st.write("- **Numerical variables** have appropriate ranges and default values")
    st.write("- **Excluded variables** are automatically set to reasonable defaults")
    
    # Display example choices for key variables
    st.write("### Example Variable Choices:")
    
    example_vars = ['Gender', 'Scholarship holder', 'International', 'Educational special needs']
    for var in example_vars:
        if var in INPUT_CHOICES:
            choices_text = ", ".join([f"{k}: {v}" for k, v in list(INPUT_CHOICES[var].items())[:3]])
            st.write(f"- **{var}**: {choices_text}")
    
    st.write("**Note**: All categorical variables show their full range of options when selected in the sidebar.")