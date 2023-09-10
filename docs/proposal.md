 # Capstone Proposal
 
## 1. Proposal Title: Heartattack Predictions and Data Analysis

- **Author Name** - Pavani Madu

- Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang
- [GitHub](https://github.com/Madu_Pavani)
- [LinkedIn](https://www.linkedin.com/in/pavani-madu9/)
- **Link to your PowerPoint presentation file** - In Progress
- **Link to your YouTube video** - In Progress

    
## 2. Background


Heart attack prediction datasets are valuable resources for researchers, data scientists, and healthcare professionals working on predictive modeling and risk assessment in cardiovascular medicine. These datasets typically contain a variety of patient-related features, medical test results, and outcomes (such as whether a patient had a heart attack or not). The goal is to develop predictive models that can identify individuals at high risk of experiencing a heart attack. These models can potentially aid in early intervention and personalized healthcare.

As of today, cardiovascular diseases remain a significant global health concern, with a substantial impact on public health and healthcare systems worldwide. The World Health Organization (WHO) has estimated that approximately 17.9 million deaths occur each year due to cardiovascular diseases, including heart diseases and stroke. This represents a concerning increase from the previously mentioned estimate of 12 million deaths per year, highlighting the ongoing challenge posed by these conditions.

Current research in this area aims to:

**Risk Factor Identification:** Scientists are continuously studying and refining the identification of risk factors for cardiovascular diseases. These factors include age, gender, family history, high blood pressure, elevated cholesterol levels, smoking, diabetes, obesity, and physical inactivity. Emerging research may also explore additional genetic, environmental, and lifestyle factors that contribute to cardiovascular risk.

**Predictive Modeling:** Advanced machine learning models, including logistic regression, are being employed to develop predictive models that can assess an individual's risk of developing cardiovascular diseases. These models consider a wide range of patient-specific variables and can provide personalized risk assessments.

**Prevention and Intervention:** The emphasis on early prognosis extends to personalized prevention and intervention strategies. Healthcare providers are increasingly using risk prediction models to tailor interventions and lifestyle recommendations for high-risk patients. This proactive approach can include diet and exercise plans, medication management, and regular monitoring.

**Public Health Initiatives:** In addition to individual patient care, public health organizations and policymakers are working on initiatives to promote heart-healthy behaviors at the population level. These efforts aim to reduce the overall burden of cardiovascular diseases through initiatives like public awareness campaigns, promoting healthier lifestyles, and improving access to healthcare services.

**Data Privacy and Ethics:** In today's context, the responsible use of patient data and adherence to privacy regulations (e.g., GDPR, HIPAA) are of paramount importance in cardiovascular disease research. Researchers and healthcare institutions must ensure that patient data is handled securely and ethically.

**Research questions**
1. Can we develop accurate predictive models for heart disease risk using machine learning techniques like logistic regression, decision trees, or neural networks?
2. What are the key features or risk factors that significantly contribute to predicting heart disease outcomes?
3. Which features (e.g., age, gender, cholesterol levels, blood pressure, ECG results) have the most significant impact on heart disease prediction, and can we prioritize them?
4. Can feature selection techniques help improve the efficiency and interpretability of predictive models?
5. Are there gender-based differences in heart disease risk and outcomes, and how can these be addressed?
6. How does age impact the incidence and prognosis of heart disease, and are there age-specific risk factors?
7. What lifestyle modifications (e.g., diet, exercise, smoking cessation) are most effective in reducing heart disease risk?
8. What are the survival rates, readmission rates, and quality of life outcomes for patients with heart disease?
9. How do comorbidities (e.g., diabetes, hypertension) affect heart disease outcomes?

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
13. thal: [normal; fixed defect; reversible defect]
14. target: yes or no (1/0)