# Student Success ML Model - Predictive Analytics for Educational Outcomes

## 📋 Project Overview

This project develops machine learning models to predict student academic success in Walsoft Computer Institute's Business Intelligence program. The goal is to optimize admissions, curriculum support, and resource allocation through data-driven insights.

## 🎯 Business Problem

Walsoft Institute needs to improve:

1. **Admissions decisions** - Identify students most likely to succeed
2. **Curriculum support** - Detect at-risk students early
3. **Resource allocation** - Optimize teaching resources for maximum impact

## 📊 Dataset

* **Source**: Walsoft Computer Institute internal records
* **Size**: [Your dataset size]
* **Features**: Age, Gender, Country, Education Background, Entry Exam Scores, Study Hours
* **Target**: Total Percentage (combined Python + Database scores)

## 🛠️ Technical Stack

* **Python 3.8+**
* **Data Analysis**: pandas, numpy
* **Visualization**: matplotlib, seaborn
* **Machine Learning**: scikit-learn
* **Models**: Linear Regression, Random Forest, K-Nearest Neighbors, Decision Trees, SVR
* **GUI**: Gradio
* **App Deployment**: Hugging Face Spaces

## 🚀 Project Structure

student_success_ml/
├── data/
│ ├── bi.csv # Original dataset
│ └── cleaned_bi.csv # Processed data
├── notebooks/
│ ├── 01_data_cleaning_eda.ipynb
│ ├── 02_feature_engineering.ipynb
│ └── 03_model_training_evaluation.ipynb
├── app/
│ └── gradio_app.py # Interactive GUI
└── README.md

## 📈 Methodology

### 1. Data Preprocessing

* Handled missing values in Python scores
* Standardized categorical variables (Gender, Country, Education levels)
* Encoded categorical features using Label Encoding
* Scaled numerical features using StandardScaler

### 2. Exploratory Data Analysis (EDA)

* Correlation analysis between entry exams and final performance
* Demographic analysis of student success factors
* Study habits impact on academic outcomes
* Identification of at-risk student segments

### 3. Modeling Approach

**Algorithms Used:**

* Linear Regression (baseline)
* Random Forest Regressor
* K-Nearest Neighbors Regressor
* Decision Tree Regressor
* Support Vector Regressor (SVR)

**Training Strategy:**

* 80-20 train-test split with shuffling
* Fit transformers on training data only
* Transform test data using fitted transformers

### 4. Evaluation Metrics

* **MAE (Mean Absolute Error)**: Average prediction error in percentage points
* **RMSE (Root Mean Squared Error)**: Penalizes larger errors more heavily
* **R² Score**: Proportion of variance explained by the model

## 📊 Results

### Model Performance Comparison

| Model             | MAE  | RMSE  | R²   |
| ----------------- | ---- | ----- | ---- |
| Linear Regression | 7.07 | 10.03 | 0.50 |
| Random Forest     | 5.90 | 8.56  | 0.64 |
| K-Neighbors       | 7.60 | 9.59  | 0.55 |
| Decision Tree     | 6.13 | 8.84  | 0.62 |
| SVR               | 7.32 | 9.41  | 0.56 |

### 🏆 Best Performing Model: Random Forest Regressor

* **R²**: 0.64 (explains 64% of score variance)
* **Average Error (MAE)**: 5.90%
* **Robustness**: Handles non-linear relationships well

---

## 🎯 Key Insights & Recommendations

### 1. Admissions Optimization

* **Entry exams are strong predictors** (correlation: 0.65 with final scores)
* **Recommendation**: Maintain entry exams as primary filter with optimized threshold

### 2. At-Risk Student Identification

* **Key risk factors**: Specific educational backgrounds, low study hours
* **Recommendation**: Implement early warning system with targeted interventions

### 3. Resource Allocation

* **Optimal study hours**: 60-80 hours for best results
* **Recommendation**: Structured study programs and progress monitoring

---

## 🚀 Gradio GUI Application

* Built an interactive **Gradio GUI** for the model
* Users can input data manually or upload CSV files
* Results are displayed instantly without coding knowledge
* App deployed on Hugging Face: [Open App](https://huggingface.co/spaces/Sara65/ML-Stdent_Success)

---

### Dependencies (`requirements.txt`)

```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
jupyter>=1.0.0
gradio>=3.50
```

---

## 📝 License

This project is for educational purposes as part of data science consulting for Walsoft Computer Institute.

## 👥 Author

[Sara Ibrahim] - Data Science Consultant
saraomran433@gmail.com
