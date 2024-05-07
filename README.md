## Mental health Prediction using Questionnaire Data
Mental Health plays a prominent a role in our present day daily routine. There are many factors which influence the mental health of an individual. In the project here we accessed the data from CDC(Center for Disease Control), to extract the features of medical data, physiological data, data regarding individual’s lifestyle, and the prescription medication used by the individual. 
The aim is to find the mental state of an individual, from a questionnaire data, which is formed by taking the required features from the extracted data. The data contains many features including medical conditions of the individuals, demographic data, lifestyle factors and the prescription data. 
We have also build an application prototype using streamlit to take inputs for the questionnaire and process the data through the ML models to output the predicted mental state of the individual. It also presents the data of the individuals with similar mental states.

The data is collected from the below source https://wwwn.cdc.gov/nchs/nhanes/default.aspx. The data collected is between 2017 and 2020.

The collected data is in an .XPT format which is extracted using a module 'project_functions', which is taken from the source https://github.com/HeyThatsViv/Predicting-Depression/tree/33af52c20d6dd10d850c82fb10f66a50605e52be/project_functions.
## Data Cleaning/features
Lot of cleaning process, went into the data pre-processing. There are many features in the data that we have obtained including individual's physiological conditions, medical conditions, prescription medications and lifestyle factors. As our aim to to create a questionnaire data, which is easily answerable by any individual, hence colums like total_cholesterol, blood_transfusion, RBC_Count, platelet_count are removed from the dataset.
The features that we used for the questionnaie data include 

'race', 'education_level', 'birth_place',
'Gender', 'asthma', 'asthma_currently', 'hay_fever', 'anemia', 'ever_overweight',
'arthritis', 'heart_failure', 'heart_disease', 'angina', 'heart_attack',
'stroke', 'thyroid_problem', 'thyroid_problem_currently', 'liver_condition',
'liver_condition_currently', 'cancer','asthma_relative', 'diabetes_relative',
'heart_attack_relative', 'work_type', 'trouble_sleeping_history',
'vigorous_recreation', 'moderate_recreation', 'vigorous_work',
'moderate_work', 'lifetime_alcohol_consumption', 'Age_in_years', 'height', 'weight', 'sleep_hours',
'sedentary_time', 'drinks_per_occasion', 'drinks_past_year', 'current_cigarettes_per_day', 'prescriptions_count'
It also includes 164 prescription medications which are among the features given as inputs for modeling.
