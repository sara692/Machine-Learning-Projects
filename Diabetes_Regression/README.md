# 🩸 Diabetes Progression Prediction(Notebook)
> Predicts how far a patient's diabetes will progress one year after baseline from 10 clinical measurements, and benchmarks eight regression algorithms on the same data to see which one generalizes best.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E) ![pandas](https://img.shields.io/badge/pandas-data-150458) ![Jupyter](https://img.shields.io/badge/Jupyter-notebook-F37626)

## Summary

- **Problem:** estimate a continuous disease-progression score so clinicians could flag patients likely to worsen, and understand which measurements drive the risk.
- **Data:** scikit-learn Diabetes dataset, 442 patients × 10 baseline features (age, sex, BMI, blood pressure, six blood-serum measurements). No missing values, no duplicates.
- **Approach:** EDA → 80/20 split → `StandardScaler` (fit on train only) → `GridSearchCV` (5-fold, R²) for 7 models + plain Linear Regression → test-set evaluation with residual diagnostics → 5-fold CV sanity check.
- **Result:** **Gradient Boosting** scores best on the held-out test set (**R² = 0.482, RMSE = 52.4, MAE = 42.6**), but the margin over the linear models is small (see Key insights).

## Results

Held-out test set (89 patients), sorted by R²:

| Model | MAE | RMSE | R² | Best hyperparameters |
|---|---|---|---|---|
| **Gradient Boosting** | 42.55 | 52.40 | **0.482** | learning_rate=0.03, max_depth=2, n_estimators=200 |
| Lasso Regression | 42.80 | 53.15 | 0.467 | alpha=1 |
| Support Vector Regression (RBF) | 42.05 | 53.41 | 0.462 | C=100, epsilon=0.1, gamma=0.01 |
| Ridge Regression | 43.03 | 53.44 | 0.461 | alpha=50 |
| Random Forest | 43.30 | 53.50 | 0.460 | max_depth=5, n_estimators=200 |
| Linear Regression | 42.79 | 53.85 | 0.453 | none |
| K-Nearest Neighbors | 44.17 | 54.07 | 0.448 | k=17, weights=distance |
| Decision Tree | 49.37 | 61.12 | 0.295 | max_depth=2 |

5-fold cross-validation on the training set (mean ± std of R²):

| Model | CV R² |
|---|---|
| Ridge | 0.456 ± 0.119 |
| Lasso | 0.454 ± 0.125 |
| Linear Regression | 0.449 ± 0.144 |
| SVR | 0.448 ± 0.121 |
| Gradient Boosting | 0.411 ± 0.133 |
| Random Forest | 0.408 ± 0.148 |
| KNN | 0.407 ± 0.097 |
| Decision Tree | 0.362 ± 0.110 |

<!-- Add 1-2 screenshots here, for example: ![Model comparison](docs/model_comparison.png) and ![Correlation heatmap](docs/correlation_heatmap.png) -->

## Key insights

- **Three features drive progression.** BMI (r = 0.59), `s5` (a triglyceride-related serum measure, r = 0.57) and blood pressure (r = 0.44) correlate most strongly with the target, which fits clinical intuition. Sex has almost no relationship (r = 0.04).
- **Simple models are as good as complex ones.** The top seven models are within 0.034 R² of each other on the test set, and the CV standard deviation (about 0.12) is larger than that gap. In cross-validation the regularized linear models (Ridge, Lasso) actually rank first, so Gradient Boosting's test-set win is not strong evidence that it is the better model.
- **Interpretability matters here.** Because the linear models perform on par, they are the sensible choice in a clinical setting where coefficients need to be explained.
- **Heavily constrained trees underfit.** The tuned Decision Tree picked `max_depth=2` and scored lowest (R² 0.295).
- **Serum features are correlated** (s1–s6), which is why Lasso and Ridge help with multicollinearity and why coefficients should be read with care.

## Project structure

```text
Diabetes_Regression/
├── diabetes_regression.ipynb   # EDA, preprocessing, 8 models, comparison, theory notes
└── README.md
```

The notebook follows this order: imports → load data → EDA → preprocessing → model training and evaluation → metrics explainer → model comparison + cross-validation → interview cheat-sheet → conclusion.

## Quick start

```bash
git clone https://github.com/sara692/Machine-Learning-Projects.git
cd Machine-Learning-Projects/Diabetes_Regression
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook diabetes_regression.ipynb
```

The dataset loads directly from `sklearn.datasets.load_diabetes()`, so no download is needed. Requires scikit-learn 1.4 or newer (`root_mean_squared_error`). Running all cells takes about a minute.

## What each model section includes

For every algorithm the notebook explains how it works, how it minimizes error (closed-form, coordinate descent, SMO, greedy splitting, or gradient descent in function space), its pros and cons, then trains and evaluates it with a predicted-vs-actual plot and a residual plot. A final cheat-sheet table compares all eight at a glance.

## Tech stack

Python, pandas, NumPy, scikit-learn (`GridSearchCV`, `cross_val_score`), matplotlib, seaborn, Jupyter.

## Limitations and next steps

- [ ] Small dataset (442 patients, 89 in the test set), so the model ranking is noisy; repeat with repeated or nested cross-validation.
- [ ] Put the scaler inside a scikit-learn `Pipeline` so scaling is refit within each CV fold and the CV estimate is leak-free.
- [ ] Try polynomial and interaction features for the linear models.
- [ ] Add prediction intervals to quantify uncertainty.
- [ ] Validate on external patient data before any real-world use. This is an educational benchmark, not a clinical tool.

## 📄 License

This project is released under the MIT License — feel free to use it for
learning or as a portfolio reference.

## 👨‍💻 Author

**Sara Ibrahim**  
*Data Scientist | Machine Learning / AI Engineer*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/sara-ibrahim-omran)  
[![Email](https://img.shields.io/badge/Email-Contact%20Me-red?style=flat&logo=gmail)](mailto:Saraomran433@gmail.com)

### 🌟 Let's Connect!

I'm passionate about solving real-world problems with data, machine learning, and AI. Feel free to reach out for:

- 💼 Job opportunities and collaborations
- 📚 Technical discussions and knowledge sharing
- 🚀 Project ideas and innovative solutions

**⭐ If you find this project helpful, please give the repo a star!**
