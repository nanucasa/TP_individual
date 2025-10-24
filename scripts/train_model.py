
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import json, joblib
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import mlflow, mlflow.sklearn

PROJ   = Path('/content/TP_individual')
DATA   = PROJ / 'data' / 'online_shoppers.csv'
MODELS = PROJ / 'models'
MLRUNS = PROJ / 'mlruns'
MODELS.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(f"file://{MLRUNS.as_posix()}")
mlflow.set_experiment("Proyecto_Integrador_LMD")

df = pd.read_csv(DATA)
y = df['Revenue'].astype(int)
X = df.drop(columns=['Revenue'])

num_cols = X.select_dtypes(include=['int64','int32','float64','float32']).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

pre = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def fit_eval(model, name):
    with mlflow.start_run(run_name=name, nested=True):
        pipe = Pipeline([('pre', pre), ('clf', model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:,1] if hasattr(pipe, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba) if y_proba is not None else float('nan')
        mlflow.log_metrics({'accuracy': acc, 'f1': f1, 'roc_auc': roc})

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(); plt.imshow(cm); plt.title(f'CM - {name}'); plt.colorbar()
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i,j], ha='center', va='center')
        plt.tight_layout()
        cm_path = MODELS / f"cm_{name}.png"
        plt.savefig(cm_path, bbox_inches='tight', dpi=120); plt.close()
        mlflow.log_artifact(str(cm_path), artifact_path='figures')

        rep = classification_report(y_test, y_pred)
        rep_path = MODELS / f"report_{name}.txt"
        rep_path.write_text(rep, encoding='utf-8')
        mlflow.log_artifact(str(rep_path), artifact_path='reports')

        mlflow.sklearn.log_model(pipe, artifact_path='model')
        joblib.dump(pipe, MODELS / f"{name}.joblib")

        return mlflow.active_run().info.run_id, acc, f1, roc

with mlflow.start_run(run_name="Padre"):
    rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced', n_jobs=-1)
    rf_run, rf_acc, rf_f1, rf_roc = fit_eval(rf, "RandomForest")

    lr = LogisticRegression(max_iter=1000)
    lr_run, lr_acc, lr_f1, lr_roc = fit_eval(lr, "LogisticRegression")

    resumen = {
        'RandomForest': {'run_id': rf_run, 'roc_auc': rf_roc},
        'LogisticRegression': {'run_id': lr_run, 'roc_auc': lr_roc}
    }
    print(json.dumps(resumen, indent=2))
