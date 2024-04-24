import streamlit as st
import pandas as pd
import numpy as np
from itertools import product
from scipy import stats
from scipy import stats as ss
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import randint
from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier

st.title("Data Analysis and Machine Learning with Streamlit")

# Section 1: Data Loading
st.header("Data Loading")
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    # Load the data
    df = pd.read_csv(uploaded_file, low_memory=False)
    st.write("Data loaded successfully!")

    # Section 2: Data Preprocessing
    st.header("Data Preprocessing")

    df1 = df.copy()

    # Map gender values
    gender_mapping = {1: 'Male', 0: 'Female'}
    df['Gender'] = df['Gender'].map(gender_mapping)

    # Display gender value counts
    st.subheader("Gender Value Counts")
    st.bar_chart(df['Gender'].value_counts())

    # Filter out unnecessary columns and map certain values
    columns_to_drop = ['Age range at first menstrual period', 'Age range at last menstrual period',
                       'How many times have been pregnant?', 'Age at first live birth', 'Age at last live birth',
                       'heart_disease_onset', 'Hours worked last week', 'work schedule past 3 months',
                       'Usually work 35 or more hours per week', 'arthritis_type', 'pregnant', 'days_quit_smoking',
                       'start_smoking_age', 'household_smokers', 'income_poverty_index']

    df.drop(columns=columns_to_drop, inplace=True)

    categorical_columns = ['depression', 'race', 'education_level', 'birth_place', 'Gender', 'arthritis',
                           'heart_failure', 'heart_disease', 'angina', 'heart_attack', 'stroke', 'thyroid_problem',
                           'thyroid_problem_currently', 'liver_condition', 'liver_condition_currently',
                           'asthma_relative', 'diabetes_relative', 'heart_attack_relative', 'work_type',
                           'vigorous_recreation', 'moderate_recreation', 'vigorous_work', 'moderate_work',
                           'drinks_per_occasion', 'lifetime_alcohol_consumption', 'current_smoker']

    continuous_columns = ['Age_in_years', 'heart_failure_onset', 'weight', 'height', 'sedentary_time',
                          'drinks_past_year', 'prescriptions_count']

    columns_to_convert = ['vigorous_work', 'moderate_recreation', 'vigorous_recreation', 'moderate_work',
                          'lifetime_alcohol_consumption']

    # Apply mapping for yes/no columns
    mapping = {'Yes': 1, 'No': 0}
    for column in columns_to_convert:
        df[column] = df[column].map(mapping)

    # Remove rows with 'Missing' or 'Dont know' in categorical columns
    for column in categorical_columns:
        df = df[(df[column] != 'Missing') & (df[column] != 'Dont know')]

    # Define categorical and continuous features
    categorical_features = categorical_columns
    continuous_features = continuous_columns

    # Define additional features for further processing
    df['depression'] = df['depression'].replace({'Not Depressed': 0, 'Depressed': 1})
    prs = df.iloc[:, 108:].columns
    prescription = [col for col in prs if 'days' not in col]
    prescription_duration = [col for col in prs if 'days' in col]

    continuous_features.extend(prescription_duration)

    # Calculate relevant categorical features using chi-squared test
    pres_var_prod = list(product(['depression'], categorical_features))
    result = []
    for var in pres_var_prod:
        if var[0] != var[1]:
            contingency_table = pd.crosstab(df[var[0]], df[var[1]])
            chi2, p, _, _ = ss.chi2_contingency(contingency_table)
            result.append((var[0], var[1], p))

    # Determine highly relevant categorical features
    threshold = 0.05
    highly_relevant_features = [feature for feature in result if feature[2] < threshold]
    relevant_cat_features = [feature[1] for feature in highly_relevant_features]

    # Calculate F-scores for continuous features
    F_scores = {}
    for cont_var in continuous_features:
        F, p = stats.f_oneway(
            df[df['depression'] == 0][cont_var],
            df[df['depression'] == 1][cont_var]
        )
        F_scores[cont_var] = F

    # Display F-scores
    F_scores_df = pd.DataFrame.from_dict(F_scores, orient='index', columns=['F_score'])
    relevant_cont_features = F_scores_df.index.tolist()

    st.subheader("Relevant Categorical Features")
    st.write(relevant_cat_features)

    st.subheader("F-Scores of Continuous Features")
    st.write(F_scores_df)

    # Create preprocessed features
    relevant_features = relevant_cat_features + relevant_cont_features
    # relevant_features.remove('depression')

    X = df[relevant_features]
    y = df['depression']

    # Label encoding for categorical features
    label_encoders = {}
    for feature in relevant_cat_features:
        label_encoders[feature] = LabelEncoder()
        X[feature] = label_encoders[feature].fit_transform(X[feature])

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA for feature reduction
    n_features = X_scaled.shape[1]
    explained_variance_ratios = []
    n_components_range = range(1, n_features + 1)

    for n_components in n_components_range:
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)
        explained_variance_ratios.append(np.sum(pca.explained_variance_ratio_))

    optimal_n_components = np.argmax(explained_variance_ratios) + 1

    st.subheader("Optimal Number of PCA Components")
    st.write(optimal_n_components)

    # Fit PCA with the optimal number of components
    pca = PCA(n_components=optimal_n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Section 3: Model Training and Evaluation
    st.header("Model Training and Evaluation")

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    # Set up RandomizedSearchCV for K-Nearest Neighbors
    knn = KNeighborsClassifier()
    param_grid = {
        'n_neighbors': randint(1, 50),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
    }

    random_search = RandomizedSearchCV(knn, param_distributions=param_grid, n_iter=100, scoring='accuracy', cv=5,
                                       random_state=42)
    random_search.fit(X_train, y_train)
    knn_model = random_search.best_estimator_

    test_accuracy = knn_model.score(X_test, y_test)
    st.subheader("Test Accuracy")
    st.write(test_accuracy)

    # Determine the best threshold for binary classification
    y_train_proba = knn_model.predict_proba(X_train)[:, 1]
    y_test_proba = knn_model.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0, 1, 100)

    best_threshold = None
    best_f1_score = 0

    for threshold in thresholds:
        y_train_pred = (y_train_proba >= threshold).astype(int)
        y_test_pred = (y_test_proba >= threshold).astype(int)

        train_f1_score = f1_score(y_train, y_train_pred)
        test_f1_score = f1_score(y_test, y_test_pred)

        if test_f1_score > best_f1_score:
            best_threshold = threshold
            best_f1_score = test_f1_score

    # Use the best threshold for final predictions
    y_train_pred = (y_train_proba >= best_threshold).astype(int)
    y_test_pred = (y_test_proba >= best_threshold).astype(int)

    # Display classification reports
    st.subheader("Classification Report (Training)")
    st.text(classification_report(y_train, y_train_pred))

    st.subheader("Classification Report (Testing)")
    st.text(classification_report(y_test, y_test_pred))

    # Bagging with KNN
    bagging_knn = BaggingClassifier(estimator=knn_model, n_estimators=10, random_state=42)
    bagging_knn.fit(X_train, y_train)

    # Evaluate Bagging
    y_train_pred = bagging_knn.predict(X_train)
    y_test_pred = bagging_knn.predict(X_test)

    st.subheader("KNN Bagging Classification Report (Training)")
    st.text(classification_report(y_train, y_train_pred))

    st.subheader("KNN Bagging Classification Report (Testing)")
    st.text(classification_report(y_test, y_test_pred))
