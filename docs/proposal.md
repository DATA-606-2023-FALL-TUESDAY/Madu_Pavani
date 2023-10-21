 # Capstone Proposal
 
## 1. Title and Author

- **Heartattack Predictions and Data Analysis**
- Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang
- Pavani Madu
- [GitHub](https://github.com/Madu_Pavani)
- [LinkedIn](https://www.linkedin.com/in/pavani-madu9/)
- **Link to your PowerPoint presentation file** - In Progress
- **Link to your YouTube video** - In Progress

    
## 2. Background


Heart attack prediction datasets are valuable resources for researchers, data scientists, and healthcare professionals working on predictive modeling and risk assessment in cardiovascular medicine. These datasets typically contain a variety of patient-related features, medical test results, and outcomes (such as whether a patient had a heart attack or not). The goal is to develop predictive models that can identify individuals at high risk of experiencing a heart attack. These models can potentially aid in early intervention and personalized healthcare.

As of today, cardiovascular diseases remain a significant global health concern, with a substantial impact on public health and healthcare systems worldwide. The World Health Organization (WHO) has estimated that approximately 17.9 million deaths occur each year due to cardiovascular diseases, including heart diseases and stroke. This represents a concerning increase from the previously mentioned estimate of 12 million deaths per year, highlighting the ongoing challenge posed by these conditions. This project aims to predict the heart disease in patient based on various factors like age, gender,cholesterol levels, blood pressure etc.,


**Research questions**
1. What are the key features or risk factors that significantly contribute to predicting heart disease outcomes?
2. Which features (e.g., age, gender, cholesterol levels, blood pressure, ECG results) have the most significant impact on heart disease prediction, and can we prioritize them?
3. Are there gender-based differences in heart disease risk and outcomes, and how can these be addressed?
4. How does age impact the incidence and prognosis of heart disease, and are there age-specific risk factors?


## 3. Data 

**Describe the datasets you are using to answer your research questions.**

- Data sources: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
- Data size: 38 KB
- Data shape: # of rows - 1026 and # columns - 14

**What features are important, what column means what**

1. age (Age of the patient in years)
2. sex (Male/Female) (1/0)
3. cp chest pain type ([typical angina, atypical angina, non-anginal, asymptomatic])
4. trestbps resting blood pressure (resting blood pressure (in mm Hg on admission to the hospital))
5. chol (serum cholesterol in mg/dl)
6. fbs (if fasting blood sugar > 120 mg/dl) (1/0)
7. restecg (resting electrocardiographic results)-- Values: [normal, stt abnormality, lv hypertrophy]
8. thalach: maximum heart rate achieved
9. exang: exercise-induced angina (True/ False) (1/0)
10. oldpeak: ST depression induced by exercise relative to rest
11. slope: the slope of the peak exercise ST segment (downsloping/flat/upsloping)
12. ca: number of major vessels (0-3) colored by fluoroscopy
13. thal: 0 = normal; 1 = fixed defect; 2 = reversable defect
14. disease: yes or no (1/0)
- The "disease" field refers to the presence of heart attack in the patient. It is integer valued 0 = no disease and 1 = disease.
- Age, Sex, cp(chest pain type), trestbps, chol may be some variables/predictors in the ML models. More research is required on the features selction going forward

## 4. Exploratory Data Analysis (EDA)

In an effort to fully understand the dataset and prepare for model training, Exploratory Data Analysis (EDA) was conducted. **Add in quote about the importance of EDA**. EDA serves as a crucial bridge between data collection and the predictive modeling process. Through this analysis, the project is able to gain insights into data variability and relationships, enabling informed decisions on methods and model choices. This project takes a multifaceted approach to EDA, including the use of Plotly Visualizations, Pandas, and an EDA Application built using Streamlit/Flask.

**Preliminary Exploration with Pandas**

Leveraging the pandas library, the project starts the exploratory data analysis by observing an overview of the dataset, summarizing key statistics across all columns, checking for missing values, and determining the unique values of the target variables.
- `heart.shape` - obtaining the shape of the dataset
- `heart.head()` - previewing the intial rows of the dataset
- `heart.dtypes` - determining the data types for each column in the dataset
- `heart.isnull().sum()` - checking for missing values by column in the dataset

In an effort to better understand the dataset and provide an effective way for future data exploration

Ability for the user to upload a dataset.
Displays basic information about the dataset.
Show information about missing values in the dataset.
Provides statistical summaries of the dataset.
Allows selection of a column and view its histogram.
Shows distributions of numerical columns.
Count plots of categorical columns.
Box plots for numerical columns.
Shows information about outliers in the dataset.
Allows users to see how the target variable varies with categorical columns.


## 5. Model Training 

- What models you will be using for predictive analytics?
  - Random Forest
  - XGBoost Classifier
  - Naïve Bayes (may decide not to use based on early results)
  - Support Vector Machines (SVM)
  - K-means
  - K-Nearest Neighbors (KNN)
- How will you train the models?
  - The train vs test split of 80/20 will be used for the models
  - Python packages used for this project include:
     - scikit-learn
     - xgboost
  - The development environments used include:
     - Personal laptop
     - Google CoLab
- How will you measure and compare the performance of the models?
  - Accuracy
  - Precision
  - Recall
  - f1-score

## 6. Application of the Trained Models

Develop a web app for people to interact with your trained models. Potential tools for web app development:

Streamlit will be used to create an application for exploratory data analysis and to create an application that will allow users to interact with the models. 

The model interaction application will feature the following functionality:
  - Users will have a set of option to select from (sex, weight, race, age, and product)
  - Once user selects their options above, they will be given an output of the body parts most likely injured and diagnosis.
    
The exploratory data analysis application will feature the following functionality:
  - Users will be able to upload a dataset and see it's dimensions
  - View N/A Values
  - Gather descriptive analytics (mean, median, etc.)
  - Visualize a histogram for a target column
  - Show the distribution of numerical columns
  - Show count plots of categorical columns
  - Visualize box plots for numerical columns
  - View outliers in the dataset
  - Visualize how a target variable varies with categorical columns 

## Distribution of Patient Gender

<img width="600" alt="image" src="https://github.com/Madu-Pavani/UMBC-DATA606-FALL2023-TUESDAY/blob/main/Img1.PNG">
There is a clear difference in the gender distribution between heart disease cases and non-cases. Males are more likely to have heart disease than females. This suggests that gender may be a significant risk factor for heart disease.

## Distribution of Heart Disease Presence
<img width="600" alt="image" src="https://github.com/Madu-Pavani/UMBC-DATA606-FALL2023-TUESDAY/blob/main/Img%202.PNG">
From above plot, there are 526 patients with heart disease cases and 499 patients with no heart disease

## Scatter Plot
<img width="600" alt="image" src="https://github.com/Madu-Pavani/UMBC-DATA606-FALL2023-TUESDAY/blob/main/Img%203.PNG">
The scatter plot suggests that age and maximum heart rate may have an association with heart disease. It appears that as individuals age, those with heart disease tend to have lower maximum heart rates compared to those without heart disease.

## Box Plot
<img width="600" alt="image" src="https://github.com/Madu-Pavani/UMBC-DATA606-FALL2023-TUESDAY/blob/main/Img%204.PNG">
Age is an important factor in heart disease. As age increases, the likelihood of heart disease also increases. There is a clear age difference between heart disease cases and non-cases. Heart disease is more prevalent in older individuals.


## 7. Conclusion

- Summarize your work and its potetial application
- Point out the limitations of your work
- Lessons learned 
- Talk about future research direction

## 8. References 

List articles, blogs, and websites that you have referenced or used in your project.