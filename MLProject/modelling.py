import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Pastikan experiment di-set di awal (Global)
experiment_name = "Fraud_No_Tuning"
mlflow.set_experiment(experiment_name)

# 2. Autolog aktif (Akan mencatat parameter model secara otomatis)
mlflow.autolog()

def run_model():
    # Load dataset
    path = "train_fraud.csv" 
    df = pd.read_csv(path)

    # Preprocessing sederhana: Pastikan integer jadi float untuk hindari warning schema
    for col in df.select_dtypes(include=['int64', 'int32']).columns:
        df[col] = df[col].astype('float64')

    X = df.drop(columns=["Fraud_Label"])
    y = df["Fraud_Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Gunakan nested=True untuk menghindari error "Active run ID does not match" 
    # jika dijalankan via MLflow Projects/GitHub Actions
    with mlflow.start_run(nested=True) as run:
        run_id = run.info.run_id
        
        # Simpan Run ID ke file txt
        with open("run_id.txt", "w") as f:
            f.write(run_id)

        print(f"Run ID disimpan: {run_id}")

        # Inisialisasi Model
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42
        )

        # Training
        model.fit(X_train, y_train)

        # Prediksi & Evaluasi
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        # Log manual untuk metrik tambahan
        mlflow.log_metric("accuracy", acc)

        # Log model dengan Input Example untuk menghindari Warning Schema
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train.iloc[:5]
        )

        print(f"Training selesai. Accuracy: {acc}")

if __name__ == "__main__":
    run_model()