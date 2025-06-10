# Stroke Risk Prediction – Project Documentation

DATASCI223

Rayan Saeed


## Problem Overview: 

Stroke is one of the leading causes of death and disability worldwide. Predicting stroke risk based on patient attributes can help identify high-risk individuals for early intervention.
This project builds a machine learning pipeline to classify patients as high or low stroke risk based on demographic and clinical features.

## Dataset Description:

Source: Kaggle Stroke Prediction Dataset 

Observations (rows): 5110

Variables (columns): 12 (after cleaning)

Key Features:

- gender, 
- age, 
- hypertension, 
- heart_disease, 
- ever_married,
- work_type, 
- Residence_type, 
- avg_glucose_level, 
- bmi, 
- smoking_status

Target variable:

- stroke (target: 1 = stroke, 0 = no stroke)

## Tools and Methods Used:

Language: Python

Libraries: 

- for data manipulation: pandas, numpy
- for data visualization: matplotlib, seaborn
- for preprocessing, modeling, evaluation: scikit-learn

Models:

- Random Forest
- Logistic Regression
- Gradient Boosting
- Support Vector Machine (SVM)

## Critical Decisions: 

1. Dropped rows with missing bmi instead of imputing (simplified approach)
2. Removed 'Other' from gender due to small sample size
3. Used one-hot encoding for categorical variables
4. Focused on 4 core models due to time constraints; skipped ensembling / deep learning


Models evaluated using accuracy, precision, recall, specificity, sensitivity, and ROC-AUC — due to class imbalance

## Issues Overcome: 

- Class imbalance: the dataset was heavily skewed toward "no stroke." Evaluated using sensitivity/specificity to account for this.
- Scaling needed for SVM: added standardization using StandardScaler
- Ensured fair comparison by training and evaluating all models on the same split

## Running the Code:

I saved my CSV file as stroke.csv in the same directory then ran each code block sequentially (data loading → cleaning → modeling → evaluation).

## Example Output:

After running the notebook, I had several elements as the output including confusion matrices, ROC curves, AUC values for all models, accuracy, precision, specificity, and sensitivity for the models as well. 

## Citations: 

Dataset: Kaggle Stroke Dataset (Soriani, F. (2021). Stroke Prediction Dataset [Data set]. Kaggle. https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
