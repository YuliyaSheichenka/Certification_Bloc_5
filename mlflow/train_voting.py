import os
import mlflow
from mlflow import log_metric, log_param, log_artifacts
from mlflow.models.signature import infer_signature
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score

from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor, RandomForestRegressor

# MLFLOW Experiment setup
experiment_name="voting_regressor"
mlflow.set_experiment(experiment_name)
experiment = mlflow.get_experiment_by_name(experiment_name)

# Set tracking URI to Heroku application
mlflow.set_tracking_uri(os.getenv("APP_URI"))

# Time execution
start_time = time.time()

# Call mlflow autolog
mlflow.sklearn.autolog(log_models=False)

# Load dataset
df = pd.read_csv("get_around_pricing_project_cleaned.csv")

# Separate target variable Y from features X
X = df.iloc[:,0:-1]
y = df.iloc[:, -1]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# Preprocessings
categorical_features = X_train.select_dtypes("object").columns # Select all the columns containing strings
categorical_transformer = OneHotEncoder(drop='first', handle_unknown='error')

numerical_feature_mask = ~X_train.columns.isin(X_train.select_dtypes("object").columns) # Select all the columns containing anything else than strings
numerical_features = X_train.columns[numerical_feature_mask]
numerical_transformer = StandardScaler()

feature_preprocessor = ColumnTransformer(
        transformers=[
            ("categorical_transformer", categorical_transformer, categorical_features),
            ("numerical_transformer", numerical_transformer, numerical_features)
        ]
    )

xgboost = XGBRegressor(max_depth=8, min_child_weight=13, n_estimators=55)
svr = SVR(C=130, epsilon=5, kernel='rbf')
random_forest = RandomForestRegressor(max_depth=14, min_samples_leaf=2, min_samples_split=4, n_estimators=70)

estimators = [("xgboost", xgboost), ("svr", svr), ("random_forest", random_forest)]

# Defining model
model = Pipeline(steps=[
    ('features_preprocessing', feature_preprocessor),
    ("Regressor", VotingRegressor(estimators))
    ])

with mlflow.start_run(experiment_id = experiment.experiment_id):
    model.fit(X_train, y_train)
    #predictions = model.predict(X_train)

    # Predictions on train and test set
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # Model metrics
    r2_train = r2_score(y_train, train_predictions)
    r2_test = r2_score(y_test, test_predictions)

    # Print results 
    print("XGBoost Regressor model")
    print("R2 on train: {}".format(r2_train))
    print("R2 on test: {}".format(r2_test))

    # Log Metric 
    mlflow.log_metric("R2 on train", r2_train)
    mlflow.log_metric("R2 on test", r2_test)

    mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="getaround",
            registered_model_name="getaround_voting",
            signature=infer_signature(X_train, train_predictions)
        )

    print(mlflow.get_artifact_uri())
    print("...Done!")
    print(f"---Total training time: {time.time()-start_time}")

