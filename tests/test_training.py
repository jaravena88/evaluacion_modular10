# tests/test_training.py
# -*- coding: utf-8 -*-

"""
Prueba de entrenamiento: verifica que se generen los artefactos.
"""

import os
import json
import joblib
import subprocess
import sys


def test_training_script_runs_and_creates_artifacts():
    # Ejecutar script de entrenamiento en un proceso separado
    result = subprocess.run(
        [sys.executable, "scripts/train_model.py"],
        capture_output=True, text=True
    )
    assert result.returncode == 0, f"Entrenamiento falló: {result.stderr}"

    # Verificar artefactos
    assert os.path.exists("models/model.pkl"), "No se creó models/model.pkl"
    assert os.path.exists("models/metadata.json"), "No se creó models/metadata.json"
    assert os.path.exists("models/feature_names.json"), "No se creó models/feature_names.json"

    # Cargar y validar
    model = joblib.load("models/model.pkl")
    assert hasattr(model, "predict"), "El modelo no tiene método predict"

    with open("models/feature_names.json", "r", encoding="utf-8") as f:
        feats = json.load(f)
    assert isinstance(feats, list) and len(feats) > 0, "feature_names.json inválido"