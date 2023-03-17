import os
import mlflow
from mlflow import log_metric, log_param, log_artifacts
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import  OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.model_selection import GridSearchCV


# Load dataset
df = pd.read_csv("get_around_pricing_project_cleaned.csv")

# Separate target variable Y from features X
print("Separating labels from features...")
features_list = df.columns[:-1]
target_variable = df.columns[-1]

X = df.loc[:,features_list]
Y = df.loc[:,target_variable]

print("...Done.")
print()

# Automatically detect names of numeric/categorical columns
numeric_features = []
categorical_features = []
for i,t in X.dtypes.items():
    if ('float' in str(t)) or ('int' in str(t)) :
        numeric_features.append(i)
    else :
        categorical_features.append(i)

print('Found numeric features ', numeric_features)
print('Found categorical features ', categorical_features)

# Train/test splitting
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Creating pipeline for numeric features
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

# Create pipeline for categorical features
categorical_transformer = Pipeline(
    steps=[
    ('encoder', OneHotEncoder(drop='first'))
    ])

# Use ColumnTransformer to make a preprocessor object that describes all the treatments to be done
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])
    
# Preprocessings on train set
X_train = preprocessor.fit_transform(X_train)

# Preprocessings on test set
X_test = preprocessor.transform(X_test)


# Set your variables for your environment
EXPERIMENT_NAME="adaboost_dt_regressor"

# Set tracking URI to your Heroku application
APP_URI =  os.getenv("APP_URI")
mlflow.set_tracking_uri(APP_URI)

# Set experiment's info 
mlflow.set_experiment(EXPERIMENT_NAME)

# Get our experiment info
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

# Call mlflow autolog
mlflow.sklearn.autolog()

with mlflow.start_run(experiment_id = experiment.experiment_id):

    # Instanciate and fit the model
    decision_tree = DecisionTreeRegressor()
    regressor = AdaBoostRegressor(decision_tree)
    
    params = {
    'base_estimator__max_depth': [10, 12, 14, 16, 17, 18, 19], # iteration 3
    'base_estimator__min_samples_leaf': [1, 2, 3], # iteration 3
    'base_estimator__min_samples_split': [2, 3, 4], # iteration 3
    'n_estimators': [100, 110, 115, 120, 125, 130] # iteration 3
    }

    best_regressor = GridSearchCV(regressor, param_grid = params, cv = 5, scoring="r2")

    best_regressor.fit(X_train, Y_train)

    # Predictions on train and test set
    Y_train_pred = best_regressor.predict(X_train)

    Y_test_pred = best_regressor.predict(X_test)


    # Model metrics
    r2_train = r2_score(Y_train, Y_train_pred)
    r2_test = r2_score(Y_test, Y_test_pred)

    # Print results 
    print("Adaboost Regressor model")
    print("R2 on train: {}".format(r2_train))
    print("R2 on test: {}".format(r2_test))

    # Log Metric 
    mlflow.log_metric("R2 on train", r2_train)
    mlflow.log_metric("R2 on test", r2_test)

    # log the best model's parameters
    for param_name, param_value in best_regressor.best_params_.items():
        mlflow.log_param(param_name, param_value)

    # Loging model 
    mlflow.sklearn.log_model(best_regressor, "model")

    print(mlflow.get_artifact_uri())

