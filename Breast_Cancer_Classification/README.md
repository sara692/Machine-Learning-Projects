# 🩺 Breast Cancer Classification — Comparative ML Study

A clean, end-to-end machine learning notebook that predicts whether a breast
tumor is **malignant** or **benign** using the Wisconsin Breast Cancer diagnostic
dataset, while comparing **7 classification algorithms** on the same data.

This project is built both as a **portfolio piece** and a **study reference for
ML/classification interview prep** — every model section explains what the
algorithm does, how it minimizes its error (gradient descent vs. other
optimization strategies), and lists commonly-asked interview facts about it.

## 📌 Highlights

- Full **EDA**: class balance, distributions, outlier checks, correlation heatmap,
  pairplots
- Clean **preprocessing** pipeline: stratified train/test split + `StandardScaler`
- **7 models trained & tuned** with `GridSearchCV` / cross-validation:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM, RBF kernel)
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - Gaussian Naive Bayes
- **Confusion matrices** and **classification reports** for every model
- Side-by-side **model comparison** (accuracy, precision, recall, F1-score) with
  visual bar chart + 5-fold CV sanity check
- Feature importance plot (Random Forest)
- A theory write-up per model: how it works, its loss function / optimizer
  (gradient descent, quadratic programming/SMO, greedy splitting, closed-form
  MLE, etc.)
- A **Confusion Matrix & Classification Report** explainer section
- An **interview cheat-sheet table** summarizing all 7 models at a glance

## 📂 Dataset

[Wisconsin Breast Cancer Diagnostic dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset),
loaded directly from `sklearn.datasets.load_breast_cancer()` — no external
download required.

- 569 samples, 30 numeric features (radius, texture, perimeter, area,
  smoothness, concavity, etc., computed from digitized images of breast masses)
- Binary target: `0 = malignant`, `1 = benign`

## 🗂️ Project structure

```
.
├── breast_cancer_classification.ipynb   # Main notebook (EDA, modeling, evaluation)
└── README.md
```

## 🚀 Getting started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/breast-cancer-classification.git
cd breast-cancer-classification
```

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Run the notebook
```bash
jupyter notebook breast_cancer_classification.ipynb
```

## 🧠 Models & how each minimizes error

| Model | Error minimization strategy |
|---|---|
| Logistic Regression | Gradient-based optimization (lbfgs / gradient descent) on log-loss |
| KNN | No training-time optimization — lazy, distance-based voting |
| SVM | Quadratic programming (Sequential Minimal Optimization) on hinge loss |
| Decision Tree | Greedy recursive splitting (CART) on Gini impurity / entropy |
| Random Forest | Bagging — averages many greedily-built trees to reduce variance |
| Gradient Boosting | Gradient descent in function space, sequentially fitting trees to residuals |
| Gaussian Naive Bayes | Closed-form Maximum Likelihood Estimation — no iterative optimization |

## 📊 Evaluation metrics used

- **Confusion Matrix** (TP / TN / FP / FN)
- **Precision, Recall, F1-score** (per class + macro/weighted averages)
- **Accuracy**
- **5-fold Cross-Validation** for robustness

## 🎯 Interview prep

The final section of the notebook is a **cheat-sheet table** and a curated list
of the most commonly asked classification-model interview questions (bagging vs.
boosting, why/when to scale features, how gradient descent works, why accuracy
alone can be misleading, bias-variance trade-off, and more).

## 🛠️ Tech stack

- Python 3
- pandas, NumPy
- matplotlib, seaborn
- scikit-learn

## 📄 License

This project is released under the MIT License — feel free to use it for
learning or as a portfolio reference.

## 🙌 Acknowledgements

Dataset originally from the UCI Machine Learning Repository, provided in
scikit-learn's `datasets` module.
