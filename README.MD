# Student Performance Prediction using Machine Learning

---

## a. Problem Statement

Student academic performance is influenced by multiple academic, social, and personal factors. Early identification of students who are at risk of failing can help educators take timely intervention measures.

The objective of this project is to predict whether a student will pass or fail based on selected features such as study time, previous failures, absences, family support, and lifestyle-related factors using multiple machine learning classification models. The project also compares the performance of different models using standard evaluation metrics.

---

## b. Dataset Description

The dataset used in this project is the Student Performance Dataset, which contains academic and personal attributes of students.

Each record in the dataset represents an individual student. The target variable **pass** is created based on the final grade **G3**:

- pass = 1 if G3 ≥ 10  
- pass = 0 otherwise  

Only test data is uploaded and used in the Streamlit application to comply with Streamlit free-tier limitations.

### Selected Features
- studytime  
- failures  
- absences  
- schoolsup  
- famsup  
- paid  
- activities  
- higher  
- internet  
- romantic  
- health  
- goout  
- Dalc  
- Walc  
- age  

### Target Variable
- pass (Binary classification: Pass / Fail)

---

## c. Models Used and Evaluation Metrics

Six machine learning models were trained and evaluated using the following metrics:

- Accuracy  
- Area Under the ROC Curve (AUC)  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.8077 | 0.5905 | 0.8512 | 0.9364 | 0.8918 | 0.0517 |
| Decision Tree | 0.7077 | 0.4591 | 0.8333 | 0.8182 | 0.8257 | -0.0787 |
| kNN | 0.8231 | 0.5364 | 0.8537 | 0.9545 | 0.9013 | 0.0872 |
| Naive Bayes | 0.7077 | 0.5986 | 0.8529 | 0.7909 | 0.8208 | 0.0359 |
| Random Forest (Ensemble) | 0.8154 | 0.6359 | 0.8707 | 0.9182 | 0.8938 | 0.1957 |
| XGBoost (Ensemble) | 0.7769 | 0.6018 | 0.8584 | 0.8818 | 0.8700 | 0.0876 |

---

## d. Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|--------------|-------------------------------------|
| Logistic Regression | Achieved strong recall and F1-score, indicating good performance in identifying students who pass. However, the low MCC suggests limited overall class discrimination. |
| Decision Tree | Showed comparatively weaker performance with the lowest AUC and a negative MCC, indicating poor generalization. |
| kNN | Produced the highest accuracy and F1-score among individual models, performing well in predicting passing students but with moderate class separation. |
| Naive Bayes | Delivered balanced results with reasonable precision and AUC, but overall accuracy was lower than other models. |
| Random Forest (Ensemble) | Demonstrated the best overall performance with high accuracy, AUC, F1-score, and the highest MCC, indicating strong and reliable predictions. |
| XGBoost (Ensemble) | Performed competitively with good precision and recall, but slightly underperformed compared to Random Forest on this dataset. |

---

## Conclusion

Based on the evaluation results, ensemble-based models performed better than individual classifiers. The Random Forest model achieved the most balanced and robust performance across all evaluation metrics, making it the most suitable model for this dataset.
