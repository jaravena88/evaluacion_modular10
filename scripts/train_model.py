# scripts/train_model.py
# -*- coding: utf-8 -*-

"""
Entrenamiento del modelo Breast Cancer.
- Preprocesa el dataset.
- Entrena un modelo de Logistic Regression (modelo simple e interpretable).
- Guarda artefactos: model.pkl, metadata.json, feature_names.json.

Justificación: Logistic Regression es adecuado para dataset pequeño, rápido de entrenar
y fácil de interpretar (importante en problemas de salud).
"""

import os
import json
from datetime import datetime
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


def train_and_save_model():
    """
    Entrena un modelo de Regresión Logística para clasificar diagnóstico (Benigno/Maligno).

    Parámetros:
        - Ninguno (las rutas están inferidas desde la estructura del repo).

    Efectos:
        - Carga data desde data/data.csv.
        - Preprocesa: elimina columnas innecesarias y codifica 'diagnosis'.
        - Entrena modelo.
        - Guarda modelo en models/model.pkl.
        - Guarda metadatos en models/metadata.json.
        - Guarda lista de columnas en models/feature_names.json.
        - Imprime métricas en consola.
    """
    # Ruta de este script y raíz del proyecto
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '..'))

    # Ruta del dataset
    data_path = os.path.join(project_root, 'data', 'data.csv')

    # Cargar dataset
    if not os.path.exists(data_path):
        print(f"❌ Error: No se encontró el dataset en: {data_path}")
        return

    df = pd.read_csv(data_path)
    print("✅ Dataset cargado exitosamente.")

    # Validaciones y limpieza
    cols_to_drop = [c for c in ['id', 'Unnamed: 32'] if c in df.columns]
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # Verificación de columnas mínimas necesarias
    if 'diagnosis' not in df.columns:
        print("❌ Error: La columna 'diagnosis' no está presente en el CSV.")
        return

    # Etiquetado (M=1, B=0)
    label_encoder = LabelEncoder()
    df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])

    # Separar variables
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Modelo (Regresión Logística)
    model = LogisticRegression(random_state=42, max_iter=10000)
    model.fit(X_train, y_train)

    # Métricas
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=['Benigno', 'Maligno']
    )

    print("\n🚀 Entrenamiento completado.\n")
    print(f"✅ Accuracy: {accuracy:.4f}\n")
    print("📋 Classification Report:\n", report)

    # Guardado de artefactos
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, 'model.pkl')
    joblib.dump(model, model_path)
    print(f"💾 Modelo guardado en: {os.path.relpath(model_path)}\n")

    metadata = {
        'model_type': 'Logistic Regression',
        'training_date': datetime.now().isoformat(),
        'accuracy': float(accuracy),
        'metrics': classification_report(y_test, y_pred, output_dict=True)
    }
    metadata_path = os.path.join(models_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print(f"💾 Metadatos guardados en: {os.path.relpath(metadata_path)}\n")

    # Guardar columnas/orden esperado
    feature_names = list(X.columns)
    feat_path = os.path.join(models_dir, 'feature_names.json')
    with open(feat_path, 'w', encoding='utf-8') as f:
        json.dump(feature_names, f, indent=2, ensure_ascii=False)
    print(f"💾 Columnas del modelo guardadas en: {os.path.relpath(feat_path)}\n")


if __name__ == '__main__':
    train_and_save_model()