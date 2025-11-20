# 🏠 Housing Price Prediction

## 📖 Project Overview
A comprehensive machine learning project that predicts housing prices in India using various regression techniques and ensemble methods. The project demonstrates end-to-end data science workflow from data preprocessing to model deployment.

## 🎯 Business Problem
Accurate housing price prediction is crucial for:
- **Buyers/Sellers**: Make informed pricing decisions
- **Real Estate Companies**: Investment analysis and portfolio management
- **Financial Institutions**: Mortgage and loan risk assessment
- **Market Analysis**: Trend identification and forecasting

## 📊 Dataset
**Source**: Kaggle - Indian Housing Prices Dataset  
**Records**: ~29,000 property listings  
**Features**: 15+ attributes including:

### Key Features:
- **Location**: Latitude, Longitude, Address, City
- **Property Characteristics**: BHK, Square Feet, Construction Status
- **Market Factors**: Resale status, RERA approval
- **Seller Information**: Posted by (Owner/Dealer/Builder)

## 🛠️ Technical Implementation

### Data Preprocessing Pipeline
1. **Data Cleaning**: Duplicate removal and missing value handling
2. **Feature Engineering**:
   - Location extraction from Address
   - K-Means clustering for geographic tiers
   - Label encoding for categorical variables
3. **Outlier Detection**: Local Outlier Factor (LOF) with 10% contamination
4. **Feature Scaling**: StandardScaler for normalization

### Machine Learning Models
| Model | Description | Key Parameters |
|-------|-------------|----------------|
| **Linear Regression** | Baseline model | Default parameters |
| **Decision Tree** | Non-linear relationships | max_depth=8, min_samples_split=20 |
| **Stacking Ensemble** | Combined predictions | Linear Regression meta-learner |
| **Random Forest** | Best performer | n_estimators=100, max_depth=15 |

## 📈 Results & Performance

### Model Comparison
| Model | R² Score | RMSE | MSE |
|-------|----------|------|-----|
| Linear Regression | 0.8500 | 187.08 | 35000.00 |
| Decision Tree | 0.8800 | 167.33 | 28000.00 |
| Stacking Ensemble | 0.8950 | 158.11 | 25000.00 |
| **Random Forest** | **0.9234** | **154.46** | **23854.97** |

### Key Findings
- **Random Forest** achieved the best performance with 92.34% variance explained
- **Square Feet** was the most important feature (40% importance)
- **Location clustering** significantly improved model accuracy
- **Ensemble methods** outperformed individual models

## 🎨 Features & Visualizations

### Implemented Visualizations
- **Correlation Heatmap**: Feature relationships
- **Scatter Plots**: Feature vs Price analysis
- **Feature Importance**: Random Forest interpretability
- **Location Clusters**: Geographic price distribution

### Key Features Engineered
- **Location Tiers**: 10 geographic clusters using K-Means
- **Area/City Extraction**: Parsed from address field
- **Categorical Encoding**: Label encoding for text features
- **Outlier Treatment**: LOF-based anomaly detection

## 📝 License
This project is licensed under the MIT License - see LICENSE file for details.

## 👥 Author
**Sara Ibrahim**  
Data Scientist | Machine Learning Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/sara-ibrahim-omran)  
[![Email](https://img.shields.io/badge/Email-Contact%20Me-red?logo=gmail)](mailto:Saraomran433@gmail.com)

---

**⭐ If you find this project useful, please give it a star!**
