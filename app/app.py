# app/app.py
# -*- coding: utf-8 -*-

"""
API REST de inferencia para el modelo de Breast Cancer.

Rutas:
- GET  /            -> status del servicio
- GET  /health      -> healthcheck simple
- POST /predict     -> recibe {"instances": [ {...}, {...} ] } y responde predicciones

Ejecución local:
    export FLASK_APP=app.app:app
    flask run --host=0.0.0.0 --port=5000

Ejecución con Gunicorn (producción local):
    gunicorn -w 2 -b 0.0.0.0:5000 app.app:app
"""

import json
import os
from typing import List

import joblib
import numpy as np
from flask import Flask, jsonify, request

from .logger import get_logger
from .utils import validate_instances

# Crear app y logger
app = Flask(__name__)
logger = get_logger(__name__)

# Rutas de artefactos
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'model.pkl')
FEATURES_PATH = os.path.join(MODELS_DIR, 'feature_names.json')
METADATA_PATH = os.path.join(MODELS_DIR, 'metadata.json')

# Cargar artefactos al iniciar
if not os.path.exists(MODEL_PATH):
    logger.warning("El modelo no existe todavía. Ejecuta 'python scripts/train_model.py' primero.")
    MODEL = None
    FEATURE_NAMES: List[str] = []
else:
    MODEL = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, 'r', encoding='utf-8') as f:
        FEATURE_NAMES = json.load(f)
    logger.info(f"Modelo y columnas cargados. n_features={len(FEATURE_NAMES)}")


@app.get("/")
def root():
    """
    Ruta raíz. Devuelve información básica del servicio.

    Parámetros:
        - Ninguno.

    Efectos:
        - Ninguno (solo respuesta HTTP).

    Retorna:
        JSON con estado y, si existe, metadatos del modelo.
    """
    meta = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return jsonify({
        "status": "ok",
        "message": "Breast Cancer API up & running",
        "model_loaded": MODEL is not None,
        "metadata": meta
    }), 200


@app.get("/health")
def health():
    """
    Healthcheck simple.

    Parámetros: Ninguno.
    Efectos: Ninguno.
    Retorna: 200 si la app corre; 500 si falta el modelo.
    """
    if MODEL is None:
        return jsonify({"status": "error", "detail": "Modelo no cargado"}), 500
    return jsonify({"status": "healthy"}), 200


@app.post("/predict")
def predict():
    """
    Endpoint de predicción.

    Cuerpo esperado (JSON):
    {
        "instances": [
            {"mean_radius": 10.95, "mean_texture": 21.35, ...},
            ...
        ]
    }

    Efectos:
        - Valida entradas y ordena columnas según 'feature_names.json'.
        - Calcula predicciones (0: Benigno, 1: Maligno).

    Retorna:
        - 200 con {"predictions": [...], "proba": [...]} si OK.
        - 400/500 con detalle de error si falla.
    """
    if MODEL is None:
        return jsonify({"error": "Modelo no cargado en el servidor"}), 500

    payload = request.get_json(silent=True)
    if not payload or 'instances' not in payload:
        return jsonify({"error": "JSON inválido. Debe incluir 'instances'."}), 400

    instances = payload['instances']
    is_ok, msg = validate_instances(instances, FEATURE_NAMES)
    if not is_ok:
        logger.warning(f"Validación fallida: {msg}")
        return jsonify({"error": msg}), 400

    try:
        # Ensamblar matriz X con el orden exacto de columnas
        X = [[row[c] for c in FEATURE_NAMES] for row in instances]
        X = np.asarray(X, dtype=float)

        preds = MODEL.predict(X).tolist()
        # Probabilidad (si el estimador la soporta)
        proba = MODEL.predict_proba(X).tolist() if hasattr(MODEL, "predict_proba") else None

        logger.info(f"Predicciones generadas para {len(instances)} instancia(s).")
        return jsonify({"predictions": preds, "proba": proba}), 200

    except Exception as e:
        logger.exception("Error durante la inferencia")
        return jsonify({"error": f"Error en predicción: {str(e)}"}), 500


# Punto de entrada para ejecutar con `python -m app.app`
if __name__ == "__main__":
    # ¡Útil en desarrollo local! Para producción usa Gunicorn o Docker.
    app.run(host="0.0.0.0", port=5000, debug=False)