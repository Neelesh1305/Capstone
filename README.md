## Mental health Prediction using Questionnaire Data

Mental Health plays a prominent a role in our present day daily routine. There are many factors which influence the mental health of an individual. In the project here we accessed the data from CDC(Center for Disease Control), to extract the features of medical data, physiological data, data regarding individual’s lifestyle, and the prescription medication used by the individual. 
The aim is to find the mental state of an individual, from a questionnaire data, which is formed by taking the required features from the extracted data. The data contains many features including medical conditions of the individuals, demographic data, lifestyle factors and the prescription data. 
We have also build an application prototype using streamlit to take inputs for the questionnaire and process the data through the ML models to output the predicted mental state of the individual. It also presents the data of the individuals with similar mental states.
![image](https://github.com/Neelesh1305/Capstone/assets/113800036/c85ea46c-173a-4d75-a2fd-dfb03dbee490)


## Data Collection/Cleaning
The data is collected from the below source https://wwwn.cdc.gov/nchs/nhanes/default.aspx. The data collected is between 2017 and 2020.


Lot of cleaning process, went into the data pre-processing. There are many features in the data that we have obtained including individual's physiological conditions, medical conditions, prescription medications and lifestyle factors. As our aim to to create a questionnaire data, which is easily answerable by any individual, hence colums like total_cholesterol, blood_transfusion, RBC_Count, platelet_count are removed from the dataset.
The features that we used for the questionnaie data include 

Features: 
1. race - the race of the individual
2. education_level - the educational level of the individual (High school/college/graduate)
3. birth_place - the birth country of the individual USA/Mexico
4. Gender - the individual's Gender (Male/Female)
5. asthma - categorical column indicating if the individual has/had asthma
6. asthma_currently - categorical column indicating if the individual has asthma currently
7. hay_fever - categorical column indicating if the individual has/had hay fever
8. anemia - categorical column indicating if the individual has/had anemia
9. ever_overweight - categorical column indicating if the individual has ever been over weight
10. arthritis - categorical column indicating if the individual has arthritis
11. heart_failure - categorical column indicating if the individual has/had heart failure
12. heart_disease - categorical column indicating if the individual has/had heart disease
13. angina - categorical column indicating if the individual has/had angina
14. heart_attack - categorical column indicating if the individual had a heart attack
15. stroke - categorical column indicating if the individual has/had stroke
16. thyroid_problem - categorical column indicating if the individual has/had thyroid problem
17. thyroid_problem_currently - categorical column indicating if the individual has thyroid problem
18. liver_condition - categorical column indicating if the individual has/had liver condition
19. liver_condition_currently - categorical column indicating if the individual has liver condition currently
20. cancer - categorical column indicating if the individual has/had heart cancer
21. asthma_relative - categorical column indicating if the individual has/had an asthma relative
22. diabetes_relative - categorical column indicating if the individual has/had a diabetes relative
23. heart_attack_relative - categorical column indicating if the individual has/had a heart attack  relative
24. work_type - categorical column indicating the type of work the individual does
25. trouble_sleeping_history - categorical column indicating if the individual had a troubled sleep history
26. vigorous_recreation - categorical column indicating if the recreational activities of the individual in a day is vigorous
27. moderate_recreation - categorical column indicating if the recreational activities of the individual in a day is moderate
28. vigorous_work - categorical column indicating if the work nature of the individual in a day is vigorous
29. moderate_work - categorical column indicating if the work nature of the individual in a day is vigorous
30. lifetime_alcohol_consumption - categorical column indicating if the individual had ever consumed alcohol
31. Age_in_years - the age of the individual(in years)
32. height - the height of the individual (in cms)
33. weight - the weight of the individual (in pounds)
34. sleep_hours - the number of hours an individual usually sleeps in a day
35. sedentary_time - the amount of sedentary time an individual spends in a day (in minutes)
36. drinks_per_occasion - the number of drinks the individual usually takes per occasion
37. drinks_past_year -  the number of drinks the individual had in the past year
38. current_cigarettes_per_day - the number of cigarettes the individual smokes per day
39. prescriptions_count - the number of prescription medications the indivdual is currently taking.

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

## Exploratory Data Analysis:

The following plots represent the distributions few continuous features

![Unknown-35](https://github.com/Neelesh1305/Capstone/assets/113800036/52830567-ec87-49d7-8d2d-2f6be5e4dd73)
![Unknown-34](https://github.com/Neelesh1305/Capstone/assets/113800036/eb4ce616-f3a6-4a75-b65e-34a61b438dea)
![Unknown-33](https://github.com/Neelesh1305/Capstone/assets/113800036/39c7e2d2-704c-4e49-bd57-9cb6dbe4f3fd)
![Unknown-32](https://github.com/Neelesh1305/Capstone/assets/113800036/471f808b-16de-4468-a023-e0f6d4683e51)


The below specified plots give some informatio regarding their variation with respect to the target variable.
These stats help us understand the probability distribution of depressed/Not depressed individuals across various classes of race, education level and ages.

![WhatsApp Image 2024-05-07 at 18 40 45](https://github.com/Neelesh1305/Capstone/assets/113800036/b3b916a2-bacd-4624-8ea9-eada0831a7b7)
![WhatsApp Image 2024-05-07 at 18 40 16](https://github.com/Neelesh1305/Capstone/assets/113800036/68f470e1-2e68-4142-b970-3d6f38e82cd4)
![WhatsApp Image 2024-05-07 at 18 41 01](https://github.com/Neelesh1305/Capstone/assets/113800036/006db524-ca87-4746-b10a-0033bd1e8521)



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
