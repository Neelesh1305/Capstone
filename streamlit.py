import streamlit as st
import pandas as pd
import numpy as np
from itertools import product
from scipy import stats
from scipy import stats as ss
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import randint
from sklearn.metrics import classification_report, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from joblib import load
import plotly.express as px

st.title("Mental Health Prediction using Machine Learning")

# Section 1: Data Loading
#st.header("Data Loading")
# Section 1: Data Loading
file_path = "CDC/clean_data1.csv"
df = pd.read_csv(file_path, low_memory=False)
# Display DataFrame
st.write(df)

if df is not None:
    # Load the data
    #st.write("Data loaded successfully!")

    # Section 2: Data Preprocessing
    #st.header("Data Preprocessing")

    df1 = df.copy()

    # Map gender values
    gender_mapping = {1: 'Male', 0: 'Female'}
    df['Gender'] = df['Gender'].map(gender_mapping)

    # Display gender value counts
    st.subheader("Gender Value Counts")
    st.bar_chart(df['Gender'].value_counts())
    relevant_cat_features =  ['race', 'education_level','birth_place','Gender','asthma','asthma_currently','hay_fever',
                              'anemia','ever_overweight','arthritis','heart_failure','heart_disease','angina','heart_attack','stroke',
                              'thyroid_problem','thyroid_problem_currently','liver_condition','liver_condition_currently','cancer','asthma_relative','diabetes_relative','heart_attack_relative','work_type','trouble_sleeping_history','vigorous_recreation',
                              'moderate_recreation','moderate_work','lifetime_alcohol_consumption']




    X = df.drop(columns={'depression'})
    y = df['depression']

    # Label encoding for categorical features
    label_encoders = {}
    for feature in relevant_cat_features:
        label_encoders[feature] = LabelEncoder()
        X[feature] = label_encoders[feature].fit_transform(X[feature])

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit PCA with the optimal number of components
    pca = PCA(n_components=201)
    X_pca = pca.fit_transform(X_scaled)

    # Section 3: Model Training and Evaluation
    #st.header("Model Training and Evaluation")

    # Section 2: Data Preprocessing
    # st.header("Data Preprocessing")

    # Section 3: Model Training and Evaluation
    #st.header("Model Training and Evaluation")

    X_train_preprocessed, X_test_preprocessed, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    # Set up RandomizedSearchCV for K-Nearest Neighbors
    # Load the saved KNN model
    model_path = 'bagging_lr_model.pkl'
    bagging_lr = load(model_path)

    # Assuming X_train, X_test, y_train, and y_test are already defined
    # Calculate test accuracy
    # bagging_lr = BaggingClassifier(base_estimator=best_logistic_regression, n_estimators=10, random_state=42)
    bagging_lr.fit(X_train_preprocessed, y_train)
    #st.subheader("Test Accuracy")
    #st.write(test_accuracy)

    # Step 7: Make predictions on both training and testing data
    y_train_pred = bagging_lr.predict(X_train_preprocessed)
    y_test_pred = bagging_lr.predict(X_test_preprocessed)
    test_accuracy = bagging_lr.score(X_test_preprocessed, y_test)
    st.write("Test Accuracy:", test_accuracy)
    # Step 8: Print the classification reports for both training and testing data
    st.write("Training Classification Report:")
    st.write(classification_report(y_train, y_train_pred))

    st.write("Testing Classification Report:")
    st.write(classification_report(y_test, y_test_pred))





    st.header("User Inputs and Predictions")
    st.write("Please fill in the information below to the best of your knowledge:")
    def collect_user_data():
        user_data = {}
        user_data['Gender'] = st.selectbox("Gender", ['Male', 'Female'])
        user_data['Race'] = st.selectbox("Race", ['Other and Multiracial', 'White', 'Mexican', 'Black', 'Other Hispanic'])
        user_data['Age'] = st.number_input("Age", min_value=0, step=1)
        user_data['Weight'] = st.number_input("Weight (lbs)", min_value=0, step=1)
        user_data['Height'] = st.number_input("Height (cm)", min_value=0, step=1)
        user_data['Education_Level'] = st.selectbox("Education Level", ['College Graduate', '9th to 11th', 'Some College', 'High School', 'Below 9th'])
        user_data['Birthplace'] = st.selectbox("Birthplace", ['Mexico', 'USA'])
        user_data['Medical_Conditions'] = st.multiselect("Medical Conditions", ['heart failure', 'anemia', 'hay_fever', 'asthma', 'heart disease', 'angina', 'heart attack', 'stroke', 'thyroid problem', 'liver condition', 'arthritis', 'None'])
        user_data['Current_Medical_Conditions'] = st.multiselect("Current Medical Conditions", ['thyroid problem', 'asthma condition', 'liver condition', 'None'])
        user_data['Asthma_Relative'] = st.selectbox("Close Relative with Asthma", ['Yes', 'No'])
        user_data['Diabetes_Relative'] = st.selectbox("Close Relative with Diabetes", ['Yes', 'No'])
        user_data['Heart_Attack_Relative'] = st.selectbox("Close Relative with Heart Attack", ['Yes', 'No'])
        user_data['Work_Type'] = st.selectbox("Current Work Type", ['Working at a job or business', 'Not working at a job or business', 'Looking for work', 'With a job or business but not at work'])
        user_data['Current_Smoker'] = st.selectbox("Current Smoker", ['Yes', 'No'])
        user_data['Recreational_Activities'] = st.selectbox("Recreational Activities", ['vigorous', 'moderate', 'None'])
        user_data['Work_Stress'] = st.selectbox("Work Stress", ['moderate', 'vigorous', 'None'])
        user_data['Sedentary_Time'] = st.number_input("Sedentary Time per day (mins)", min_value=0, step=1)
        user_data['trouble_sleeping_history'] = st.selectbox("Trouble sleeping history ", ['Yes', 'No'])
        user_data['ever_overweight'] = st.selectbox("Ever overweight ", ['Yes', 'No'])
        user_data['sleep_hours'] = st.number_input("Sleep Hours", min_value=0, step=1)
        user_data['Alcohol_Consumption'] = st.selectbox("Alcohol Consumption", ['Yes', 'No'])
        user_data['Drinks_Per_Occasion'] = st.number_input("Drinks per Occasion", min_value=0, step=1)
        user_data['Drinks_Past_Year'] = st.number_input("Drinks_Past_Year", min_value=0, step=1)
        user_data['Current_cigarettes_per_day'] = st.number_input("Current cigarettes per day", min_value=0, step=1)
        user_data['Prescriptions_Count'] = st.number_input("Prescriptions Count", min_value=0, step=1)
        # Medications and number of days
        medications = st.multiselect("Medications", ['None', 'Rx_LISINOPRIL', 'Rx_METFORMIN', 'Rx_ALBUTEROL', 'Rx_LEVOTHYROXINE', 'Rx_SIMVASTATIN', 'Rx_METOPROLOL', 'Rx_ATORVASTATIN', 'Rx_AMLODIPINE', 'Rx_OMEPRAZOLE', 'Rx_HYDROCHLOROTHIAZIDE', 'Rx_FUROSEMIDE', 'Rx_ATENOLOL', 'Rx_LOSARTAN', 'Rx_MONTELUKAST', 'Rx_AMOXICILLIN', 'Rx_ACETAMINOPHEN; HYDROCODONE', 'Rx_GABAPENTIN', 'Rx_POTASSIUM CHLORIDE', 'Rx_GLIPIZIDE', 'Rx_IBUPROFEN', 'Rx_RANITIDINE', 'Rx_CLOPIDOGREL', 'Rx_PRAVASTATIN', 'Rx_WARFARIN', 'Rx_ESOMEPRAZOLE', 'Rx_FLUTICASONE NASAL', 'Rx_CARVEDILOL', 'Rx_ROSUVASTATIN', 'Rx_INSULIN GLARGINE', 'Rx_FLUTICASONE; SALMETEROL', 'Rx_CETIRIZINE', 'Rx_TAMSULOSIN', 'Rx_PANTOPRAZOLE', 'Rx_TRAMADOL', 'Rx_ALPRAZOLAM', 'Rx_HYDROCHLOROTHIAZIDE; LISINOPRIL', 'Rx_LOVASTATIN', 'Rx_PREDNISONE', 'Rx_NAPROXEN', 'Rx_ALLOPURINOL', 'Rx_VALSARTAN', 'Rx_ASPIRIN', 'Rx_ZOLPIDEM', 'Rx_CYCLOBENZAPRINE', 'Rx_DILTIAZEM', 'Rx_ALENDRONATE', 'Rx_CLONIDINE', 'Rx_LANSOPRAZOLE', 'Rx_METHYLPHENIDATE', 'Rx_MELOXICAM', 'Rx_HYDROCHLOROTHIAZIDE; TRIAMTERENE', 'Rx_GLIMEPIRIDE', 'Rx_CLONAZEPAM', 'Rx_PIOGLITAZONE', 'Rx_ENALAPRIL', 'Rx_FENOFIBRATE', 'Rx_FLUTICASONE', 'Rx_GLYBURIDE', 'Rx_SPIRONOLACTONE', 'Rx_LORAZEPAM', 'Rx_HYDROCHLOROTHIAZIDE; LOSARTAN', 'Rx_AMPHETAMINE; DEXTROAMPHETAMINE', 'Rx_NIFEDIPINE', 'Rx_HYDROCODONE', 'Rx_FINASTERIDE', 'Rx_FEXOFENADINE', 'Rx_EZETIMIBE', 'Rx_ETHINYL ESTRADIOL; NORGESTIMATE', 'Rx_DIGOXIN', 'Rx_INSULIN ASPART', 'Rx_ETHINYL ESTRADIOL; NORETHINDRONE', 'Rx_AMLODIPINE; BENAZEPRIL', 'Rx_OXYCODONE', 'Rx_AZITHROMYCIN', 'Rx_RAMIPRIL', 'Rx_CELECOXIB', 'Rx_BENAZEPRIL', 'Rx_TERAZOSIN', 'Rx_FAMOTIDINE', 'Rx_ACETAMINOPHEN; OXYCODONE', 'Rx_EZETIMIBE; SIMVASTATIN', 'Rx_MOMETASONE NASAL', 'Rx_SITAGLIPTIN', 'Rx_BUDESONIDE', 'Rx_HYDROXYZINE', 'Rx_NITROGLYCERIN', 'Rx_VERAPAMIL', 'Rx_PROPRANOLOL', 'Rx_TIOTROPIUM', 'Rx_ISOSORBIDE MONONITRATE', 'Rx_DONEPEZIL', 'Rx_DOXAZOSIN', 'Rx_OXYBUTYNIN', 'Rx_HYDROCHLOROTHIAZIDE; VALSARTAN', 'Rx_ESTRADIOL', 'Rx_DICLOFENAC', 'Rx_TOPIRAMATE', 'Rx_LATANOPROST OPHTHALMIC', 'Rx_HYDRALAZINE', 'Rx_OLMESARTAN', 'Rx_SULFAMETHOXAZOLE; TRIMETHOPRIM', 'Rx_CEPHALEXIN', 'Rx_GEMFIBROZIL', 'Rx_INSULIN LISPRO', 'Rx_PREGABALIN', 'Rx_AMOXICILLIN; CLAVULANATE', 'Rx_ALBUTEROL; IPRATROPIUM', 'Rx_CONJUGATED ESTROGENS', 'Rx_DOXYCYCLINE', 'Rx_BECLOMETHASONE', 'Rx_BUSPIRONE', 'Rx_QUINAPRIL', 'Rx_ACETAMINOPHEN; CODEINE', 'Rx_DIAZEPAM', 'Rx_ETHINYL ESTRADIOL; LEVONORGESTREL', 'Rx_INSULIN DETEMIR', 'Rx_BUDESONIDE; FORMOTEROL', 'Rx_MECLIZINE', 'Rx_POLYETHYLENE GLYCOL 3350', 'Rx_BACLOFEN', 'Rx_LISDEXAMFETAMINE', 'Rx_LEVALBUTEROL', 'Rx_METOCLOPRAMIDE', 'Rx_METHOTREXATE', 'Rx_TEMAZEPAM', 'Rx_PROMETHAZINE', 'Rx_LEVETIRACETAM', 'Rx_IRBESARTAN', 'Rx_DOCUSATE', 'Rx_HYDROCHLOROTHIAZIDE; OLMESARTAN', 'Rx_CIPROFLOXACIN', 'Rx_ONDANSETRON', 'Rx_PHENYTOIN', 'Rx_NIACIN', 'Rx_HYDROXYCHLOROQUINE', 'Rx_AMIODARONE', 'Rx_TIZANIDINE', 'Rx_BRIMONIDINE OPHTHALMIC', 'Rx_RISEDRONATE', 'Rx_IPRATROPIUM', 'Rx_TIMOLOL OPHTHALMIC', 'Rx_ROPINIROLE', 'Rx_CEFDINIR', 'Rx_GLYBURIDE; METFORMIN', 'Rx_ACYCLOVIR', 'Rx_INSULIN ISOPHANE; INSULIN REGULAR', 'Rx_COLCHICINE', 'Rx_TRIAMCINOLONE TOPICAL', 'Rx_RALOXIFENE', 'Rx_RABEPRAZOLE', 'Rx_SUMATRIPTAN', 'Rx_TRIAMTERENE', 'Rx_NITROFURANTOIN', 'Rx_TOLTERODINE', 'Rx_ISOSORBIDE', 'Rx_PREDNISOLONE', 'Rx_DROSPIRENONE; ETHINYL ESTRADIOL', 'Rx_ACETAMINOPHEN; PROPOXYPHENE', 'Rx_INSULIN REGULAR', 'Rx_MEMANTINE', 'Rx_PENICILLIN', 'Rx_DIPHENHYDRAMINE', 'Rx_MINOCYCLINE', 'Rx_METHOCARBAMOL'])
    
        # Dictionary to store medication days
        medication_days = {}
    
        # For each selected medication, collect the number of days
        for medication in medications:
            if medication != 'None':
                days = st.text_input(f"Number of days for {medication}", key=medication)
                medication_days[medication] = int(days) if days else None
    
        user_data['Medication_Days'] = medication_days
    
        return user_data

    user_data = collect_user_data()

    st.write("User Data Collected:")
    st.write(user_data)
    medications = ['Rx_LISINOPRIL','Rx_METFORMIN','Rx_ALBUTEROL', 'Rx_LEVOTHYROXINE',
    'Rx_SIMVASTATIN','Rx_METOPROLOL','Rx_ATORVASTATIN','Rx_AMLODIPINE','Rx_OMEPRAZOLE',
    'Rx_HYDROCHLOROTHIAZIDE','Rx_FUROSEMIDE','Rx_ATENOLOL','Rx_LOSARTAN',
    'Rx_MONTELUKAST','Rx_AMOXICILLIN','Rx_ACETAMINOPHEN; HYDROCODONE','Rx_GABAPENTIN',
    'Rx_POTASSIUM CHLORIDE','Rx_GLIPIZIDE','Rx_IBUPROFEN','Rx_RANITIDINE','Rx_CLOPIDOGREL',
    'Rx_PRAVASTATIN','Rx_WARFARIN','Rx_ESOMEPRAZOLE','Rx_FLUTICASONE NASAL','Rx_CARVEDILOL',
    'Rx_ROSUVASTATIN','Rx_INSULIN GLARGINE','Rx_FLUTICASONE; SALMETEROL','Rx_CETIRIZINE',
    'Rx_TAMSULOSIN','Rx_PANTOPRAZOLE','Rx_TRAMADOL','Rx_ALPRAZOLAM','Rx_HYDROCHLOROTHIAZIDE; LISINOPRIL',
    'Rx_LOVASTATIN','Rx_PREDNISONE','Rx_NAPROXEN','Rx_ALLOPURINOL','Rx_VALSARTAN',
    'Rx_ASPIRIN','Rx_ZOLPIDEM','Rx_CYCLOBENZAPRINE','Rx_DILTIAZEM','Rx_ALENDRONATE',
    'Rx_CLONIDINE','Rx_LANSOPRAZOLE','Rx_METHYLPHENIDATE','Rx_MELOXICAM','Rx_HYDROCHLOROTHIAZIDE; TRIAMTERENE',
    'Rx_GLIMEPIRIDE','Rx_CLONAZEPAM','Rx_PIOGLITAZONE','Rx_ENALAPRIL','Rx_FENOFIBRATE',
    'Rx_FLUTICASONE','Rx_GLYBURIDE','Rx_SPIRONOLACTONE','Rx_LORAZEPAM','Rx_HYDROCHLOROTHIAZIDE; LOSARTAN',
    'Rx_AMPHETAMINE; DEXTROAMPHETAMINE','Rx_NIFEDIPINE','Rx_HYDROCODONE','Rx_FINASTERIDE',
    'Rx_FEXOFENADINE','Rx_EZETIMIBE','Rx_ETHINYL ESTRADIOL; NORGESTIMATE','Rx_DIGOXIN',
    'Rx_INSULIN ASPART','Rx_ETHINYL ESTRADIOL; NORETHINDRONE','Rx_AMLODIPINE; BENAZEPRIL',
    'Rx_OXYCODONE','Rx_AZITHROMYCIN','Rx_RAMIPRIL','Rx_CELECOXIB','Rx_BENAZEPRIL','Rx_TERAZOSIN',
    'Rx_FAMOTIDINE','Rx_ACETAMINOPHEN; OXYCODONE','Rx_EZETIMIBE; SIMVASTATIN',
    'Rx_MOMETASONE NASAL','Rx_SITAGLIPTIN','Rx_BUDESONIDE','Rx_HYDROXYZINE','Rx_NITROGLYCERIN',
    'Rx_VERAPAMIL','Rx_PROPRANOLOL','Rx_TIOTROPIUM','Rx_ISOSORBIDE MONONITRATE',
    'Rx_DONEPEZIL','Rx_DOXAZOSIN','Rx_OXYBUTYNIN','Rx_HYDROCHLOROTHIAZIDE; VALSARTAN',
    'Rx_ESTRADIOL','Rx_DICLOFENAC','Rx_TOPIRAMATE','Rx_LATANOPROST OPHTHALMIC','Rx_HYDRALAZINE',
    'Rx_OLMESARTAN','Rx_SULFAMETHOXAZOLE; TRIMETHOPRIM','Rx_CEPHALEXIN','Rx_GEMFIBROZIL','Rx_INSULIN LISPRO',
    'Rx_PREGABALIN','Rx_AMOXICILLIN; CLAVULANATE','Rx_ALBUTEROL; IPRATROPIUM','Rx_CONJUGATED ESTROGENS',
    'Rx_DOXYCYCLINE','Rx_BECLOMETHASONE','Rx_BUSPIRONE','Rx_QUINAPRIL','Rx_ACETAMINOPHEN; CODEINE',
    'Rx_DIAZEPAM','Rx_ETHINYL ESTRADIOL; LEVONORGESTREL','Rx_INSULIN DETEMIR','Rx_BUDESONIDE; FORMOTEROL',
    'Rx_MECLIZINE','Rx_POLYETHYLENE GLYCOL 3350','Rx_BACLOFEN','Rx_LISDEXAMFETAMINE',
    'Rx_LEVALBUTEROL','Rx_METOCLOPRAMIDE','Rx_METHOTREXATE','Rx_TEMAZEPAM','Rx_PROMETHAZINE',
    'Rx_LEVETIRACETAM','Rx_IRBESARTAN','Rx_DOCUSATE','Rx_HYDROCHLOROTHIAZIDE; OLMESARTAN',
    'Rx_CIPROFLOXACIN','Rx_ONDANSETRON','Rx_PHENYTOIN','Rx_NIACIN','Rx_HYDROXYCHLOROQUINE',
    'Rx_AMIODARONE','Rx_TIZANIDINE','Rx_BRIMONIDINE OPHTHALMIC','Rx_RISEDRONATE',
    'Rx_IPRATROPIUM','Rx_TIMOLOL OPHTHALMIC','Rx_ROPINIROLE','Rx_CEFDINIR','Rx_GLYBURIDE; METFORMIN',
    'Rx_ACYCLOVIR','Rx_INSULIN ISOPHANE; INSULIN REGULAR','Rx_COLCHICINE','Rx_TRIAMCINOLONE TOPICAL','Rx_RALOXIFENE','Rx_RABEPRAZOLE',
    'Rx_SUMATRIPTAN','Rx_TRIAMTERENE','Rx_NITROFURANTOIN','Rx_TOLTERODINE','Rx_ISOSORBIDE',
    'Rx_PREDNISOLONE','Rx_DROSPIRENONE; ETHINYL ESTRADIOL','Rx_ACETAMINOPHEN; PROPOXYPHENE',
    'Rx_INSULIN REGULAR','Rx_MEMANTINE','Rx_PENICILLIN','Rx_DIPHENHYDRAMINE',
    'Rx_MINOCYCLINE','Rx_METHOCARBAMOL']
    medication_columns_mapping = {}
    for medication in medications:
        medication_columns_mapping[medication] = "Rx_days_" + medication.split('_')[1]
    # Function to convert user data to X row
    def user_data_to_X_row(user_data):
        # Initialize X row with zeros
        X_row = {
        'race': user_data['Race'],
        'education_level': user_data['Education_Level'],
        'birth_place': user_data['Birthplace'],
        'Gender': user_data['Gender'],
        'asthma': 1 if 'asthma' in user_data['Medical_Conditions'] else 0,
        'arthritis': 1 if 'arthritis' in user_data['Medical_Conditions'] else 0,
        'heart_failure': 1 if 'heart_failure' in user_data['Medical_Conditions'] else 0,
        'heart_disease': 1 if 'heart_disease' in user_data['Medical_Conditions'] else 0,
        'angina': 1 if 'angina' in user_data['Medical_Conditions'] else 0,
        'hay_fever': 1 if 'hay_fever' in user_data['Medical_Conditions'] else 0,
        'anemia': 1 if 'anemia' in user_data['Medical_Conditions'] else 0,        
        'heart_attack': 1 if 'heart_attack' in user_data['Medical_Conditions'] else 0,
        'stroke': 1 if 'stroke' in user_data['Medical_Conditions'] else 0,
        'thyroid_problem': 1 if 'thyroid problem' in user_data['Medical_Conditions'] else 0,
        'thyroid_problem_currently': 1 if 'thyroid problem' in user_data['Current_Medical_Conditions'] else 0,
        'liver_condition': 1 if 'liver_condition' in user_data['Medical_Conditions'] else 0,
        'liver_condition_currently': 1 if 'liver condition' in user_data['Current_Medical_Conditions'] else 0,
        'asthma_currently': 1 if 'asthma condition' in user_data['Current_Medical_Conditions'] else 0,
        'asthma_relative': 1 if user_data['Asthma_Relative'] == "Yes" else 0,
        'diabetes_relative': 1 if user_data['Diabetes_Relative'] == "Yes" else 0,
        'heart_attack_relative': 1 if user_data['Heart_Attack_Relative'] == "Yes" else 0,
        'work_type': user_data['Work_Type'],
        'moderate_recreation': 1 if user_data['Recreational_Activities'] == "moderate" else 0,
        'vigorous_recreation': 1 if user_data['Recreational_Activities'] == "vigorous" else 0,
        'moderate_work': 1 if user_data['Work_Stress'] == "moderate" else 0,
        'vigorous_work': 1 if user_data['Work_Stress'] == "moderate" else 0,
        'ever_overweight': 1 if user_data['ever_overweight'] == "Yes" else 0,
        'drinks_per_occasion': user_data['Drinks_Per_Occasion'],
        'lifetime_alcohol_consumption': 1 if user_data['Alcohol_Consumption'] == "Yes" else 0,
        'current_cigarettes_per_day': user_data['Current_cigarettes_per_day'],
        'Age_in_years': user_data['Age'],
        'weight': user_data['Weight'],
        'height': user_data['Height'],
        'sedentary_time': user_data['Sedentary_Time'],
        'drinks_past_year': user_data['Drinks_Past_Year'],
        'trouble_sleeping_history': 1 if user_data['trouble_sleeping_history'] == "Yes" else 0,
        'sleep_hours': user_data['sleep_hours'],
        'prescriptions_count': user_data['Prescriptions_Count']
        }
        # Set medication days
        for medication, days in user_data['Medication_Days'].items():
            if medication in medication_columns_mapping:
                X_row[medication_columns_mapping[medication]] = days
    
        # Set all other medication columns to zero
        for medication_column in medication_columns_mapping.values():
            if medication_column not in X_row:
                X_row[medication_column] = 0

    
        return X_row

# Convert user data to X row
    X_row = user_data_to_X_row(user_data)

# Create a DataFrame with the X row
    X_df = pd.DataFrame([X_row])
    X_df.fillna(0)
    relevant_cat_features = ['race','education_level','birth_place','Gender','work_type']
    label_encoders = {}
    for feature in relevant_cat_features:
        label_encoders[feature] = LabelEncoder()

# Apply LabelEncoder to the categorical features
        #X_preprocessed = X_row.copy()  # Make a copy of X to avoid modifying the original data
    for feature in relevant_cat_features:
        X_df[feature] = label_encoders[feature].fit_transform(X_df[feature])
    #st.subheader("X_sample")
    #st.write(X_df)


    X_row_standardized = scaler.fit_transform(X_df)

    X_row_pca = pca.transform(X_row_standardized)
    #X_pca = np.vstack([X_pca, X_row_pca])
    y_test_proba = bagging_lr.predict_proba(X_row_pca)

    df_proba = bagging_lr.predict_proba(X_pca)

    #X_row_proba = knn_model.predict_proba(X_row_pca)
    # sample_proba = X_row_proba
    st.write(y_test_proba)
    #st.write(df_proba.shape)
    # Print the percentage of Depressed class and Not depressed class
    # depressed_percentage = df_proba[1218][1] * 100:
    # not_depressed_percentage = df_proba[1218][0] * 100:
    
    st.subheader("According to the received inputs from Questionnaire, the prediction is that")
    st.subheader(f"You are {y_test_proba[0][1] * 100:.2f}% Depressed")
    st.subheader(f"You are {y_test_proba[0][0] * 100:.2f}% Not depressed")
    
    st.write("The above prediction is only partially accurate, due to less f1 scores of the model")
    df_proba = bagging_lr.predict_proba(X_pca)
    # sample_proba = df_proba[1218]
    distances = np.linalg.norm(df_proba - y_test_proba[0], axis=1)
    closest_indices = np.argsort(distances)[:20]
    st.write("Below are the stats of 20 of the individuals with mental state most similar to yours")
    for idx in closest_indices:
        st.write("Row:", idx)
        st.write("Probabilities (Not depressed, Depressed):", df_proba[idx])
    selected_df = df.iloc[closest_indices, :]
    class_counts = {}
    categorical_01 = ['depression', 'asthma', 'asthma_currently', 'asthma_emergency',
                  'hay_fever', 'anemia', 'ever_overweight', 'arthritis',
                  'heart_failure', 'heart_disease', 'angina', 'heart_attack', 'stroke',
                  'thyroid_problem', 'thyroid_problem_currently', 'liver_condition',
                  'liver_condition_currently', 'cancer', 'asthma_relative',
                  'diabetes_relative', 'heart_attack_relative', 'trouble_sleeping_history',
                  'vigorous_recreation', 'moderate_recreation', 'vigorous_work', 'moderate_work', 'lifetime_alcohol_consumption']
    
    for column in selected_df.columns:
        if column in categorical_01:  # Only consider columns with categorical [0, 1] data
            class_counts[column] = selected_df[column].sum()


    st.pyplot(plt.figure(figsize=(16, 6)))  # This line ensures that barchart is displayed in Streamlit
    st.subheader(f"Medical conditions")
    # Create the bar chart using st.barchart()
    st.bar_chart(class_counts)

    # Optionally, you can add labels and adjust layout as per your requirements
    plt.xlabel('Columns')
    plt.ylabel('Number of Class 1')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Define the list of categorical columns
    categorical_columns = ['race', 'education_level', 'birth_place', 'work_type']

    # Iterate over each categorical column
    # Iterate over each categorical column
    for column in categorical_columns:
        if column == 'race' or column == 'work_type':
            # Create pie chart for 'race' and 'work_type'
            df_column_counts = df[column].value_counts()
            fig, ax = plt.subplots()
            ax.pie(df_column_counts, labels=df_column_counts.index, autopct='%1.1f%%', startangle=140)
            ax.set_title(f'Pie chart for {column}')
            st.pyplot(fig)
        else:
            # Create barchart for other columns
            plt.figure(figsize=(10, 6))
            sns.countplot(data=df.iloc[closest_indices], x=column, hue='Gender', palette='Set2')  # Specify palette here
            plt.title(f'Count of {column} by Gender')
            plt.xlabel(column)
            plt.ylabel('Count')
            plt.legend(title='Gender')
            plt.xticks(rotation=45) 
            st.pyplot(plt)

    

    # Iterate over each categorical column

    

