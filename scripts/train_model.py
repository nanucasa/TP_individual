# dejo imports aquí porque es un archivo independiente ejecutado por Python
import os
import mlflow, mlflow.sklearn
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42
DATA_PATH = "data/online_shoppers.csv"
TARGET = "Revenue"
REGISTERED_MODEL_NAME = "Cliente_Compra_Model"

def load_data():
    df = pd.read_csv(DATA_PATH)
    assert TARGET in df.columns, f"Falta columna '{TARGET}'."
    return df

def build_preprocessor(df):
    X = df.drop(columns=[TARGET])
    cat_cols = X.select_dtypes(include=["object","bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])

def plot_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(cm, interpolation='nearest')
    ax.set_title(title); ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    for (i,j), v in np.ndenumerate(cm): ax.text(j,i,int(v),ha='center',va='center')
    fig.tight_layout(); return fig

def eval_and_log(y_true, y_proba, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    try: auc = roc_auc_score(y_true, y_proba)
    except Exception: auc = float("nan")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("roc_auc", auc)
    return {"accuracy":acc, "f1":f1, "roc_auc":auc}

def run_child(name, model, pre, Xtr, Xte, ytr, yte):
    with mlflow.start_run(run_name=name, nested=True) as child:
        pipe = Pipeline([("pre", pre), ("clf", model)])
        pipe.fit(Xtr, ytr)
        y_pred  = pipe.predict(Xte)
        y_proba = pipe.predict_proba(Xte)[:,1] if hasattr(pipe,"predict_proba") else y_pred.astype(float)
        metrics = eval_and_log(yte, y_proba, y_pred)
        fig = plot_cm(yte, y_pred, f"CM - {name}"); pth=f"cm_{name}.png"; fig.savefig(pth); plt.close(fig)
        mlflow.log_artifact(pth)
        mlflow.sklearn.log_model(pipe, artifact_path="model")
        return child.info.run_id, metrics

def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    df = load_data()
    X, y = df.drop(columns=[TARGET]), df[TARGET].astype(int)
    pre = build_preprocessor(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    with mlflow.start_run(run_name="parent_run"):
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("test_size", 0.2)

        rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
        lr = LogisticRegression(max_iter=1000, solver="liblinear")

        runs = []
        for name, mdl in [("RandomForest", rf), ("LogisticRegression", lr)]:
            rid, mets = run_child(name, mdl, pre, Xtr, Xte, ytr, yte)
            runs.append((name, rid, mets))

        runs.sort(key=lambda x: x[2]["roc_auc"], reverse=True)
        best_name, best_run_id, best_metrics = runs[0]
        mlflow.log_param("best_model", best_name)
        print("Mejor:", best_name, "run_id:", best_run_id, "metrics:", best_metrics)

        mv = mlflow.register_model(model_uri=f"runs:/{best_run_id}/model",
                                   name=REGISTERED_MODEL_NAME)
        print("Registrado en Model Registry:", REGISTERED_MODEL_NAME, "→ versión:", mv.version)

if __name__ == "__main__":
    main()
