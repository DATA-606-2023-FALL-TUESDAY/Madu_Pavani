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