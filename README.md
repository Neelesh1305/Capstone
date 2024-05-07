## Mental health Prediction using Questionnaire Data

Mental Health plays a prominent a role in our present day daily routine. There are many factors which influence the mental health of an individual. In the project here we accessed the data from CDC(Center for Disease Control), to extract the features of medical data, physiological data, data regarding individual’s lifestyle, and the prescription medication used by the individual. 
The aim is to find the mental state of an individual, from a questionnaire data, which is formed by taking the required features from the extracted data. The data contains many features including medical conditions of the individuals, demographic data, lifestyle factors and the prescription data. 
We have also build an application prototype using streamlit to take inputs for the questionnaire and process the data through the ML models to output the predicted mental state of the individual. It also presents the data of the individuals with similar mental states.
![image](https://github.com/Neelesh1305/Capstone/assets/113800036/c85ea46c-173a-4d75-a2fd-dfb03dbee490)


## Data Collection/Cleaning
The data is collected from the below source https://wwwn.cdc.gov/nchs/nhanes/default.aspx. The data collected is between 2017 and 2020.


Lot of cleaning process, went into the data pre-processing. There are many features in the data that we have obtained including individual's physiological conditions, medical conditions, prescription medications and lifestyle factors. As our aim to to create a questionnaire data, which is easily answerable by any individual, hence colums like total_cholesterol, blood_transfusion, RBC_Count, platelet_count are removed from the dataset.
The features that we used for the questionnaie data include 

Features = ['race', 'education_level', 'birth_place',
'Gender', 'asthma', 'asthma_currently', 'hay_fever', 'anemia', 'ever_overweight',
'arthritis', 'heart_failure', 'heart_disease', 'angina', 'heart_attack',
'stroke', 'thyroid_problem', 'thyroid_problem_currently', 'liver_condition',
'liver_condition_currently', 'cancer','asthma_relative', 'diabetes_relative',
'heart_attack_relative', 'work_type', 'trouble_sleeping_history',
'vigorous_recreation', 'moderate_recreation', 'vigorous_work',
'moderate_work', 'lifetime_alcohol_consumption', 'Age_in_years', 'height', 'weight', 'sleep_hours',
'sedentary_time', 'drinks_per_occasion', 'drinks_past_year', 'current_cigarettes_per_day', 'prescriptions_count']
It also includes 164 prescription medications which are among the features given as inputs for modeling, which we have also manually clustered into 14 clusters by doing our research, which are mentioned below 
1. psychotic_medications
2. bloodpressure_medications
3. respiratory_medications
4. endocrine_medications
5. git_medications
6. antibiotic_medications
7. opiods_medications
8. nervous_system_medications
9. nonopiod_antiinflammatory_medications
10. anti_clotting_medications
11. immunosuppressants_medications
12. muscle_relaxant_medications
13. cardiovasular_medications
14. metabolic_disorders_medications

The collected data is in an .XPT format which is extracted using a module 'project_functions', which is taken from the source https://github.com/HeyThatsViv/Predicting-Depression/tree/33af52c20d6dd10d850c82fb10f66a50605e52be/project_functions.
 ## Model Development:
For model development, we considered the mixture of continuous (for example: Age_in_years, weight, height, BMI) and categorical variables(for example, race, education_level, pregnant, birth_place) in our dataset. 
We selected models that are suited for handling categorical variables, such as Decision Trees, Naive Bayes classifiers, and Support Vector classifiers (SVC). 
Additionally, we utilized models like Logistic Regression, Feedforward Neural Networks (FCNNs), and Gradient Boosting, which are effective for both categorical and continuous variables. 
The models that showed significant performaces are
1. Logistic Regression
2. Decision Trees
3. SVC
4. NBClassifier
5. FCNN model

Logistic Regression model after bagging is the model that we have finally used and the metrics of the model include a testing accuracy of 0.89 and the Classification report is as follows:

<img width="483" alt="Screenshot 2024-05-07 at 6 39 47 PM" src="https://github.com/Neelesh1305/Capstone/assets/113800036/f02d5ced-4054-48b2-a285-799d2bd37377">

## Research Outcomes:

The analysis conducted using various models, indicates that it is indeed possible to predict the likelihood of depression with more features and probably more data along with existing medical conditions, lifestyle factors, and prescription medications.
 Though the data is valuable, it couldn’t excel in predictive modeling in mental health prediction.
While specific data points are not detailed here, generally, models like the KNN classifier performed better than other models at predicting both classes. Using more features which are relevant might result in better predictions.

## Conclusions Drawn: 
Despite the inherent challenges posed by the imbalanced class distribution in the dataset,we tried our best to build models to handle them, our model is robust enough to handle skewed data distributions, which is common in medical datasets. The model's ability to accurately predict depressed/Not depressed based on medical conditions, lifestyle factors, and prescription medications can be instrumental in healthcare settings. It allows for early identification of individuals at risk, facilitating timely interventions and potentially preventing the escalation of depressive symptoms.
In conclusion, our project has focused on predicting depression based on an analysis of medical conditions, lifestyle factors, and prescription medications. Through the deployment phase, we have implemented a predictive model and provided users with an interface for accessing the model's predictions.
Throughout the project, we have demonstrated the potential impact of data science and machine learning in improving mental health outcomes. Despite our challenges in achieving satisfactory results, we believe that our model has the potential to positively impact individuals' lives and rise the awareness regarding Mental Health.
Moving forward, it is crucial to continue research in mental health prediction. This includes refining our model, expanding our dataset, and applying our model to different populations. By doing so, we can improve the accuracy and effectiveness of our predictions, ultimately leading to better mental health outcomes for all.
