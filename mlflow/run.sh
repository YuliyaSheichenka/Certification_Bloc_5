docker run -it \
-p 4000:4000 \
-v "$(pwd):/home/app" \
-e APP_URI=$APP_URI \
-e MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI \
-e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
-e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
-e BACKEND_STORE_URI=$BACKEND_STORE_URI \
-e ARTIFACT_ROOT=$ARTIFACT_ROOT \
getaround-mlflow-15032023 python train_adaboost_regressor_dt.py