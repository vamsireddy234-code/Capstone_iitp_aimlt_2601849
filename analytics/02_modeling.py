import os

import pandas as pd

import joblib

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import plot_tree

from imblearn.over_sampling import SMOTE

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (accuracy_score,precision_score, recall_score, f1_score,confusion_matrix, roc_curve, roc_auc_score)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

folder = os.path.dirname(os.path.abspath(__file__))

clean_data_path = os.path.join(folder, "titanic_cleaned.csv")

df = pd.read_csv(clean_data_path)

# for column in df.columns:
#     print(f"{column} --- {df[column].unique()}")


X = df.drop(columns=["survived"])

y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,  stratify=y)

## As from the the EDA it was clear that the passenger who survived is much less than the one who not survived so in order to equate the slipt this is important


# X_train["sex"] = X_train["sex"].map({"male" : '1', "female" : '0'})

# X_test["sex"] = X_test["sex"].map({"male" : '1', "female" : '0'})

# X_train = pd.get_dummies(X_train, columns=["embark_town"] ,drop_first=False)

# X_test = pd.get_dummies(X_test, columns=["embark_town"] ,drop_first=False)

# stdsc = StandardScaler()

# X_train_scaler = stdsc.fit_transform(X_train)

# X_test_scaler = stdsc.transform(X_test)


categorical_features = ["sex", "embark_town"]

numeric_features = ["pclass","age","sibsp","parch","fare"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            StandardScaler(),
            numeric_features
        )
    ]
)



## No missing values as this is taken from the cleaned data set from the part A

models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

log_models = {}

for name, model in models.items():


    log_model = Pipeline([

        ("preprocessor",preprocessor),
        ("classifier" , model)

    ]


    )

    log_models[name] = log_model

    log_model.fit(X_train,y_train)



plot_tree(
    log_models["Decision Tree"].named_steps["classifier"],
    filled=True
)

plt.title("Decision Tree")
plt.show()

y_pred_log = log_models["Logistic Regression"].predict(X_test)
y_pred_Dec = log_models["Decision Tree"].predict(X_test)
y_pred_Ran = log_models["Random Forest"].predict(X_test)

print(y_pred_log)
print(y_pred_Dec)
print(y_pred_Ran)

print("Logistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))

print("Decision Tree Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_Dec))

print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_Ran))

model_preds = {

   "Logistic Regression": y_pred_log,
    "Decision Tree": y_pred_Dec,
    "Random Forest": y_pred_Ran
}

results = []

for name, y_pred in model_preds.items():
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    

    results.append([name,accuracy,precision,recall,f1])

results_df = pd.DataFrame(results, columns= ["Model","accuracy","precision","recall","f1"] )

print(results_df)

y_pred_log_proba = log_models["Logistic Regression"].predict_proba(X_test)[:,1]
y_pred_Dec_proba = log_models["Decision Tree"].predict_proba(X_test)[:,1]
y_pred_Ran_proba = log_models["Random Forest"].predict_proba(X_test)[:,1]


fpr_log, tpr_log, thresholds_log = roc_curve(y_test, y_pred_log_proba)
auc_log = roc_auc_score(y_test, y_pred_log_proba)

fpr_Dec, tpr_Dec, thresholds_Dec = roc_curve(y_test, y_pred_Dec_proba)
auc_Dec = roc_auc_score(y_test, y_pred_Dec_proba)

fpr_Ran, tpr_Ran, thresholds_Ran = roc_curve(y_test, y_pred_Ran_proba)
auc_Ran = roc_auc_score(y_test, y_pred_Ran_proba)


print("Logistic Regression AUC:", auc_log)
print("Decision Tree AUC:", auc_Dec)
print("Random Forest AUC:", auc_Ran)

plt.plot(fpr_log,tpr_log,label=f"Logistic Regression (AUC = {auc_log})")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Titanic Survival")

plt.show()

plt.plot(fpr_Dec,tpr_Dec,label=f"Desicion Tree (AUC = {auc_Dec})")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Titanic Survival")

plt.show()


plt.plot(fpr_Ran,tpr_Ran,label=f"Random Forest (AUC = {auc_Ran})")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Titanic Survival")

plt.show()


baseline_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(random_state=42))
])

baseline_model.fit(X_train, y_train)

y_pred_baseline = baseline_model.predict(X_test)



balanced_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        random_state=42
    ))
])

balanced_model.fit(X_train, y_train)

y_pred_balanced = balanced_model.predict(X_test)


smote_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            StandardScaler(),
            numeric_features
        )
    ]
)


X_train_processed = smote_preprocessor.fit_transform(X_train)


X_test_processed = smote_preprocessor.transform(X_test)

smote = SMOTE(random_state=42)

X_train_smote , y_train_smote = smote.fit_resample(X_train_processed, y_train)

smote_model = LogisticRegression(random_state=42)

smote_model.fit(X_train_smote,y_train_smote)

y_pred_smote = smote_model.predict(X_test_processed)



imbalance_results = []

imbalance_predictions = {
    "Baseline": y_pred_baseline,
    "Class Weight Balanced": y_pred_balanced,
    "SMOTE": y_pred_smote
}


for name, y_pred in imbalance_predictions.items():
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    imbalance_results.append([name,precision,recall,f1])

imbalance_results_df = pd.DataFrame(imbalance_results,columns=["Method", "Precision", "Recall", "F1"])
print(imbalance_results_df)

# Baseline: The model makes fairly accurate predictions, but it misses some passengers who actually survived.
# Class Weight Balanced: The model identifies more passengers who actually survived, but it also makes slightly more wrong predictions.
# SMOTE: The model gives the best overall balance by identifying more survivors while keeping the wrong predictions reasonably low.

ran_pipeline = Pipeline([

    ("preprocessor" , preprocessor),
    ("classifier", RandomForestClassifier(oob_score=True, random_state=42))
])


parameter_grid = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [None, 5, 10, 15], 
    "classifier__max_features": ["sqrt", "log2"]
}

grid_search = GridSearchCV(estimator=ran_pipeline,param_grid=parameter_grid,cv=5,scoring=None,n_jobs=-1)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

best_ran_pipeline = grid_search.best_estimator_

oob_score = best_ran_pipeline.named_steps["classifier"].oob_score_

print("OOB Score:", oob_score)

X_reg = df.drop(columns=["fare"])
y_reg = df["fare"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg,y_reg, test_size=0.2, random_state=42)

reg_categorical_features = ["sex", "embark_town"]

reg_numeric_features = ["survived","pclass","age","sibsp","parch"]

reg_preprocessor = ColumnTransformer(
    transformers=[
        ("categorical",
            OneHotEncoder(handle_unknown="ignore"),
            reg_categorical_features),
        ("numeric",
            StandardScaler(),
            reg_numeric_features)
    ]
)

reg_pipeline = Pipeline([("preprocessor", reg_preprocessor),("regressor", LinearRegression())])

reg_pipeline.fit(X_train_reg, y_train_reg)

y_reg_pred = reg_pipeline.predict(X_test_reg)

residuals = y_test_reg - y_reg_pred

#print(residuals)

plt.scatter(y_reg_pred, residuals)
plt.xlabel("Predicted Fare")
plt.ylabel("Residuals")
plt.title("Residual Plot - Fare Prediction")

plt.show()

##The residuals are not evenly spread around zero and their spread increases with predicted fare, indicating heteroscedasticity.


mae = mean_absolute_error(y_test_reg,y_reg_pred)

mse = mean_squared_error(y_test_reg,y_reg_pred) 

rmse = mse**0.5

r2 = r2_score(y_test_reg,y_reg_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)


n = len(y_train_reg)

p = len(reg_preprocessor.get_feature_names_out())

adjusted_R2 = 1-((1-r2)*(n-1)/(n -p - 1))

print("Adjusted R²:", adjusted_R2)


final_comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Linear Regression"
    ],

    # Classification metrics
    "Accuracy": [0.783439, 0.751592, 0.777070, None],
    "Precision": [0.767857, 0.719298, 0.784314, None],
    "Recall": [0.671875, 0.640625, 0.625000, None],
    "F1": [0.716667, 0.677686, 0.695652, None],
    "AUC": [auc_log, auc_Dec, auc_Ran, None],

    # Regression metrics
    "MAE": [None, None, None, mae],
    "RMSE": [None, None, None, rmse],
    "R²": [None, None, None, r2],
    "Adjusted R²": [None, None, None, adjusted_R2]
})

print(final_comparison)

print(r"""Based on the results, I would choose Logistic Regression for the Titanic survival prediction. 
It has the highest accuracy of 78.34%, the highest recall of 67.19%, and an F1 score of 71.67% among the three models. 
Its AUC is also good, which means the model can separate survivors and non-survivors reasonably well. 
Therefore, Logistic Regression is a simple and reliable choice for this dataset.""")

baseline_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(random_state=42))
])

baseline_model.fit(X_train, y_train)   

model_joblib_logistic = os.path.join(folder, "titanic_model.joblib")

joblib.dump(baseline_model, model_joblib_logistic)

loaded_model = joblib.load(model_joblib_logistic)

y_pred_loaded = loaded_model.predict(X_test)

print("Predictions are same:",
      (y_pred_loaded == y_pred_baseline).all())


