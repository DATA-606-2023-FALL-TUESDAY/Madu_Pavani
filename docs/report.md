 # Capstone Proposal
 
## 1. Title and Author

- **Heartattack Predictions and Data Analysis**
- Prepared for UMBC Data Science Master Degree Capstone by Dr Chaojie (Jay) Wang
- Pavani Madu
- [GitHub](https://github.com/Madu_Pavani)
- [LinkedIn](https://www.linkedin.com/in/pavani-madu9/)
- [Capstone.ppt](https://github.com/DATA-606-2023-FALL-TUESDAY/Madu_Pavani/blob/main/docs/Heart%20Disease%20Prediction.pptx)
- [Youtube](https://www.linkedin.com/in/pavani-madu9/)

    
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

# Preprocessing

No potential missing values found in the dataset
The data is pretty clean and no further pre-processing required

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


## 5. Model Training 

What models you will be using for predictive analytics?
Models are trained and analyzed
- K- nearest neighbors
- Random Forest Classifier
- Logistic regression

The confusion matrix is drawn for analysis

How will you train the models?
The train vs test split of 80/20 will be used for the models.

Python packages used for this project include:
scikit-learn2

The development environments used include:
Personal laptop
Jupyter Notebook
Visual Studio Code

How will you measure and compare the performance of the models?
Accuracy
Precision
Recall
f1-score

Model Improvement Methods:
Hyperparameter Tuning - Part of the scikit-learn library is a hyperparameter tuning technique called RandomizedSearchCV. Being that the dataset is large, this was the optimal hyperparameter tuning technique, as it randomly selects combinations of parameters to try.
Feature Importance - This improvement technique assigns a score to the machine learning models input features based on how useful they are at predicting a target variable. Once these values are assigned, it can help us improve and interpret the models.


## K - nearest neighbors:
- K-nearest neighbors (KNN) is a type of supervised learning algorithm.
- It makes predictions based on the similarity of input data to labeled examples in a training dataset.
- The algorithm records all of the labels and training examples during training.
- KNN locates the k nearest neighbors in the training data and gives the majority label among them to the input to create a prediction for a new input.
KNN has been performed with different values for k and the respective accuracy values are plotted as below

![image](https://github.com/Madu-Pavani/UMBC-DATA606-FALL2023-TUESDAY/blob/main/KNN%20Model.PNG)

- We can see how accuracy increased until k is 8 and decreased after k is 9. Similar case was oserved with different proportions of testsets with different random states. 

- By taking k as 5 validations are performed and the results are given in results section.

## Random Forest Classifier
A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is controlled with the max_samples parameter if bootstrap=True (default), otherwise the whole dataset is used to build each tree.

## Logistic Regression
- Logistic regression is a supervised learning algorithm used for binary or multi-class classification tasks.
- It models the relationship between the input features and the probability of the output label.
- The approach calculates the input feature coefficients that maximize the likelihood of the labels that were observed in the training set.
The likelihood of each class is calculated during prediction, and the output label is given to the class with the highest probability.

## Random Search CV

This is a method for hyperparameter tuning that performs a randomized search over a specified parameter grid. It randomly samples a set of hyperparameter combinations from predefined distributions. 

In this case, cv is set to 5-fold cross-validation, meaning the dataset is split into 5 parts, and the model is trained and evaluated 5 times, each time using a different part as the test set.

## Grid Search CV

This is another method for hyperparameter tuning that performs an exhaustive search over a specified parameter grid. Unlike RandomizedSearchCV, which randomly samples a set of hyperparameter combinations, GridSearchCV considers all possible combinations in the specified parameter grid.


## AUC ROC Curve
![image](https://github.com/Madu-Pavani/UMBC-DATA606-FALL2023-TUESDAY/blob/main/AUC%20ROC%20Curve.PNG)

## Confusion Matrix
![image](https://github.com/Madu-Pavani/UMBC-DATA606-FALL2023-TUESDAY/blob/main/Confusion%20Matrix.PNG)

## 6. Application of the Trained Models

Streamlit is used to create an application for exploratory data analysis and to create an application that will allow users to interact with the models. 
When user interacts with the app and input different parameters of a patient, the prediction results are displayed.

The model interaction application will feature the following functionality:
  - Users will have a set of option to select from (sex, gender, age, and chest pain type)
  - Once user selects their options above, they will be given an output of the heart disease presence or not.
  
The goal was to create a user-friendly application that will be both a practical tool and bridge the gap between complex models and end-users, such as healthcare professionals, researchers, or injured citzens. The highest performing model was Random Forest Classifier for a broader data range. But for the project a minimal data was used and Logistic Regression performs best, which was used as the machine learning prediction model. The application offers a range of functionalities, providing users with various options to input and obtain tailored results. These functionalities include:

User Input Options - Users can input specific details related to patients, such as age, sex, gender and the type chest pain involved. This level of detail allows for a more personalized analysis.
Predictive Analytics - Based on the input data, the application predicts the presence of heart disease utilizing the trained Logistic Regression model.
Accessibility and Ease of Use - Designed with a focus on user experience, the application facilitates easy navigation and interpretation of results, making it accessible to a broad audience.

##  Model Results
![image](https://github.com/Madu-Pavani/UMBC-DATA606-FALL2023-TUESDAY/blob/main/Results.PNG)

## 7. Conclusion

1. The logistic regression model achieved AUC score of 94% indicating a good level on predictive performance
2. Random forest algorithm achieved 100% accuracy which seems to be overfitting the model
3. Different factors like age, gender, chest pain type, maximum heart rate achieved have a significant impact on the heart disease
4. As the RFC model is overfitting in this case, I have chosen logistic regression to be the best model to predict heart disease presence in the patients.


## 8. References 

- Dataset: https://www.kaggle.com/code/talhabarkaatahmad/heartattack-predictions-and-data-analysis/input

- Sklearn.linear_model.logisticregression. scikit. (n.d.). Retrieved May 11, 2022, from 
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

- Sklearn.ensemble.randomforestclassifier. scikit. (n.d.). Retrieved May 11, 2022, from https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

- Supervised learning. scikit. (n.d.). Retrieved May 11, 2022, from 
https://scikit-learn.org/stable/supervised_learning.html