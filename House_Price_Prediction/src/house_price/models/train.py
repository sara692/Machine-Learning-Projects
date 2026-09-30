from house_price.config import Settings
from house_price.config import settings as default_settings
from pathlib import Path
import pandas as pd 
import numpy as np

import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor,StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

class TrainModel:
    def __init__(self, settings: Settings):
        self.settings = settings or default_settings
        self.posted_by_encoder = LabelEncoder()
        self.city_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.best_model_name: str | None = None
        self.best_model = None
        self.best_params: dict = {}
        self.evaluation_results: dict[str, dict] = {}

    # ------------------------------------------------------------------ #
    # Data loading 
    # ------------------------------------------------------------------ #

    def load_data(self):
        """
        Load the dataset from the specified path.
        """
        try:
            data = pd.read_csv(self.settings.data_path)
            return data
        except FileNotFoundError:
            print(f"Data file not found at {self.settings.data_path}")
            return None

    # ------------------------------------------------------------------ #
    # Cleaning / feature engineering
    # ------------------------------------------------------------------ #

    def clean_data_and_feature_engineering(self, data: pd.DataFrame):
        """
        Clean and preprocess the dataset.
        """
        df=data.copy()
        # Drop rows with duplicated values
        df=df.drop_duplicates()
        # Extract Area and City from ADDRESS
        df["Area"]=df['ADDRESS'].str.split(",").str[0]
        df["city"]=df['ADDRESS'].str.split(",").str[1]
        # Create Location Tiers using KMeans Clustering
        # Create 10 clusters directly for 10 tiers
        kmeans = KMeans(n_clusters=10, random_state=42)
        cluster_labels = kmeans.fit_predict(df[['LATITUDE', 'LONGITUDE']])
        self.kmeans = kmeans  # Store the KMeans model for later use


        # Direct mapping to tier labels with "tier" prefix
        tier_mapping = {0: 'tier1', 1: 'tier2', 2: 'tier3', 3: 'tier4', 4: 'tier5', 
                        5: 'tier6', 6: 'tier7', 7: 'tier8', 8: 'tier9', 9: 'tier10'}
        df['loc_tier'] = cluster_labels
        df['loc_tier'] = df['loc_tier'].map(tier_mapping).astype('object')
        # Create dummy variables for location tiers
        df = pd.get_dummies(df, columns=['loc_tier'], dtype=int)
        # Label Encoding for categorical variables
        df["POSTED_BY"] = self.posted_by_encoder.fit_transform(df["POSTED_BY"])
        df["city"] = self.city_encoder.fit_transform(df["city"])

        return df

    # ------------------------------------------------------------------ #
    # OUTLIER DETECTION AND REMOVAL
    # ------------------------------------------------------------------ #
    def outlier_detection_and_removal(self, df: pd.DataFrame):

        df = df.copy()
        Numerical_data=df.select_dtypes(include=['int64','float64', "int32"])
        X=Numerical_data.drop(['UNDER_CONSTRUCTION','LONGITUDE', 'LATITUDE','city','TARGET(PRICE_IN_LACS)'],axis=1)
        y=df['TARGET(PRICE_IN_LACS)'] 

        from sklearn.neighbors import LocalOutlierFactor

        lof = LocalOutlierFactor(contamination=0.1, n_neighbors=20)
        outlier_labels = lof.fit_predict(X)
        
        # Keep only inliers (label == 1)
        inlier_mask = outlier_labels == 1
        
        X_clean = X[inlier_mask]
        y_clean = y[inlier_mask]
        return X_clean, y_clean

    # ------------------------------------------------------------------ #
    # Data Splitting
    # ------------------------------------------------------------------ #
    def split_data(self, X: pd.DataFrame, y: pd.Series):
        """
        Split the dataset into training and testing sets.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.settings.test_size, random_state=self.settings.random_seed
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test
    # ------------------------------------------------------------------ #
    # Hyperparameter tuning + model selection (MLflow-tracked)
    # ------------------------------------------------------------------ #
    def tune_and_evaluate_models(self, X_train, X_test, y_train, y_test) -> str:
        """
        Tune hyperparameters for different models and evaluate their performance.
        """
        models = {
            "RandomForestRegressor": RandomForestRegressor(random_state=self.settings.random_seed),
            "DecisionTreeRegressor": DecisionTreeRegressor(random_state=self.settings.random_seed),
            "LinearRegression": LinearRegression()
        }

        param_grids = {
            "RandomForestRegressor": {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            },
            "DecisionTreeRegressor": {
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            },
            "LinearRegression": {}
        }

        best_model_name = None
        best_model = None
        best_score = float('-inf')
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment("house_price_prediction")

        with mlflow.start_run() as parent_run:

            for model_name, model in models.items():
                print(f"Tuning {model_name}...")
                param_grid = param_grids[model_name]
                grid_search = GridSearchCV(model, param_grid, cv=self.settings.cv_folds, scoring='r2')
                grid_search.fit(X_train, y_train)

                best_params = grid_search.best_params_
                best_estimator = grid_search.best_estimator_

                # Evaluate on test set
                y_pred = best_estimator.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                self.evaluation_results[model_name] = {
                    "best_estimator": best_estimator,
                    "best_params": best_params,
                    "mse": mse,
                    "r2": r2,
                    "cv_score": grid_search.best_score_
                    
                }

                with mlflow.start_run(run_name=model_name,nested=True) as child_run:
                    self.evaluation_results[model_name]["run_id"] = child_run.info.run_id
                    mlflow.log_params(best_params)
                    mlflow.log_metric("mse", mse)
                    mlflow.log_metric("r2", r2)
                    mlflow.log_metric("cv_score", grid_search.best_score_)
                    mlflow.sklearn.log_model(
                    best_estimator,
                    name=model_name,
                    skops_trusted_types=["sklearn.tree._tree.Tree"]
                )
                
            best_model_name = max(self.evaluation_results, key=lambda x: self.evaluation_results[x]["r2"])
            run_id = self.evaluation_results[best_model_name]["run_id"]
            mlflow.register_model(f"runs:/{run_id}/{best_model_name}", self.settings.registered_model_name)
            return best_model_name
    # ------------------------------------------------------------------ #
    # Artifacts Saving
    # ------------------------------------------------------------------ #
    
    def save_artifacts(self, best_model_name: str):
        """
        Save the best model, scaler, and encoder to disk.
        """
        best_model = self.evaluation_results[best_model_name]["best_estimator"]
        joblib.dump(best_model, self.settings.model_path)
        joblib.dump(self.scaler, self.settings.scaler_path)
        joblib.dump(self.posted_by_encoder, self.settings.posted_by_encoder_path)
        joblib.dump(self.city_encoder, self.settings.city_encoder_path)
        joblib.dump(self.kmeans, self.settings.kmeans_path)
        return self.settings.model_path
    
    def run(self) -> Path:
        """Run the full pipeline end-to-end and return the saved model path."""
        df = self.load_data()
        df = self.clean_data_and_feature_engineering(df)
        X,y = self.outlier_detection_and_removal(df)
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        best_model_name = self.tune_and_evaluate_models(X_train, X_test, y_train, y_test)
        return self.save_artifacts(best_model_name)

def train_and_save(settings: Settings | None = None) -> Path:
    """Functional convenience wrapper around ModelTrainer, used by the console script."""
    return TrainModel(settings).run()


if __name__ == "__main__":
    train_and_save()
