A web-based Machine Learning application that predicts a person's risk of sleep disorders based on daily lifestyle and health data, while providing lifestyle improvement recommendations based on profile similarity. Built with Streamlit and integrated with an SQLite database for history tracking.

Overview

* Sleep Quality & Stress Level Predictor is a healthcare Machine Learning application that analyzes user lifestyle and physical condition data to predict the risk of sleep disorders - No Disorder, Insomnia, or Sleep Apnea.
* Beyond prediction, the application generates lifestyle improvement recommendations by finding other individuals with similar health profiles who do not experience sleep disorders. This ensures the advice is realistic and data-driven rather than just generic rules.

Background

* Sleep quality plays a major role in daily health and productivity, yet lifestyle factors such as physical activity, stress levels, and irregular routine patterns are often overlooked.
* This project was developed to identify sleep disorder risks early using Machine Learning, while translating those prediction results into actionable lifestyle recommendations that the user can immediately implement, rather than just providing a diagnostic label.

How to Use

1. Users input daily lifestyle and health data (age, sleep duration, physical activity, stress level, heart rate, blood pressure, etc.).
2. The Logistic Regression model predicts the sleep disorder category along with its confidence score.
3. The system performs a profile similarity search using K-Nearest Neighbors (KNN) to find individuals with similar profiles who do not have sleep disorders, then generates tailored lifestyle recommendations.
4. Every prediction result is automatically saved into a local SQLite database.

Features

* Sleep disorder risk prediction with confidence scores.
* Profile similarity-based lifestyle recommendations (KNN).
* Prediction history automatically saved to an SQLite database.
* Public access availability via Ngrok tunneling.

Tech Stack

**Machine Learning**
* Python
* Scikit-learn
* imbalanced-learn (SMOTE)
* Joblib

**Web Application**
* Streamlit

**Database & Deployment**
* SQLite
* Pyngrok

**Development Tools**
* Google Colab / Visual Studio Code
* Git & GitHub

About the Model

| Item | Description |
| :--- | :--- |
| **Dataset** | Sleep Health and Lifestyle Dataset (Kaggle) - 374 rows, 13 columns. |
| **Preprocessing** | Missing value imputation, blood pressure column separation (systolic/diastolic), BMI category standardization, categorical feature label encoding. |
| **Data Balancing** | SMOTE (applied only to training data). |
| **Target** | Sleep Disorder: No Disorder / Insomnia / Sleep Apnea. |
| **Final Model** | Logistic Regression - selected because its computational complexity is lighter for real-time predictions and it produces stable, easily interpretable probability outputs. |

**Algorithm Performance Comparison (Test Data):**

| Model | Accuracy | F1-Score (Macro) |
| :--- | :--- | :--- |
| Logistic Regression | 93.33% | 0.9062 |
| KNN (k=5) | 93.33% | 0.9062 |
| Decision Tree | 92.00% | 0.8890 |
| Random Forest | 92.00% | 0.8799 |

System Architecture

User Input
      |
      v
Preprocessing (Scaler + Label Encoder)
      |
      v
Logistic Regression Model
      |
 +----+--------------+
 |                   |
 v                   v
Prediction       Similarity Search
+ Confidence     (KNN, healthy profile)
 |                   |
 v                   v
SQLite History   Lifestyle Recommendations

Project Structure

Sleep-Disorder-Predictor/
|-- app.py                     # Main Streamlit application
|-- run_with_ngrok.py          # Script to run app + public expose via Ngrok
|-- sleep_disorder_model.pkl   # Bundled model + scaler + encoder from training
|-- sleep_health_clean.csv     # Cleaned dataset
+-- .gitignore

Future Improvements

* Add more lifestyle parameters (diet, screen time, sleep environment conditions).
* Deploy to permanent cloud hosting as an alternative to Ngrok tunneling.
* Add a visual dashboard for user history trends.
* Expand the recommendation system with more detailed lifestyle segmentation.
* Enhance responsive design for mobile devices.

Data Source

* Sleep Health and Lifestyle Dataset - Kaggle

Team

* Chesya Kinanti
* Mutia Nur Khalifa
* Industrial Informatics Engineering
* Politeknik Manufaktur Negeri Bandung
