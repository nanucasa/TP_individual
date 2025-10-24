
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

PROJ = Path('/content/TP_individual')
MLRUNS = PROJ / 'mlruns'
EXPERIMENT = "Proyecto_Integrador_LMD"

# Tracking local por archivos
mlflow.set_tracking_uri(f"file://{MLRUNS.as_posix()}")
client = MlflowClient()

exp = client.get_experiment_by_name(EXPERIMENT)
assert exp is not None, "No encontré el experimento."

# Traigo muchos runs y filtro en Python (sin filter_string)
runs = client.search_runs([exp.experiment_id], max_results=1000)
cands = []
for r in runs:
    name = r.data.tags.get("mlflow.runName", "")
    if name in ("RandomForest", "LogisticRegression"):
        roc = r.data.metrics.get("roc_auc")
        if roc is not None:
            cands.append((roc, name, r))

assert cands, "No hay runs hijos con 'roc_auc'. Ejecutá antes el BLOQUE 5."
cands.sort(key=lambda x: x[0], reverse=True)
best_roc, best_name, best_run = cands[0]

print("Top por roc_auc:")
for roc, name, r in cands[:5]:
    print(f"  {name}  {r.info.run_id}  roc_auc={roc:.4f}")

# Permito pegar un run_id; Enter = mejor
try:
    run_id = input("Pegá el run_id ganador (Enter = mejor): ").strip()
except EOFError:
    run_id = ""
if not run_id:
    run_id = best_run.info.run_id

model_uri = f"runs:/{run_id}/model"
model_name = "Cliente_Compra_Model"

print(f"Registrando model_uri={model_uri} como '{model_name}'...")
try:
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    print(f"OK: registrado {model_name} v{mv.version}")
except Exception as e:
    print("No se pudo registrar en el Model Registry con el backend actual.")
    print("Detalle:", repr(e))
    print("Sugerencia: usar un tracking server con backend DB (p.ej., DagsHub) y reintentar.")
    print(f"De todas formas, podés referenciar el modelo con: {model_uri}")
