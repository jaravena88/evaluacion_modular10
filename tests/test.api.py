# tests/test_api.py
# -*- coding: utf-8 -*-

"""
Pruebas básicas del API con el test client de Flask.
"""

import json
import os
import pytest

# Para importar 'app' del paquete app.app
from app.app import app, FEATURE_NAMES, MODEL


@pytest.fixture(scope="module")
def client():
    # Cliente de pruebas de Flask
    with app.test_client() as c:
        yield c


def test_root_ok(client):
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "status" in data and data["status"] == "ok"


def test_health(client):
    # Si no hay modelo, debería fallar con 500; si hay modelo -> 200
    resp = client.get("/health")
    if MODEL is None:
        assert resp.status_code == 500
    else:
        assert resp.status_code == 200


def test_predict_happy_path(client):
    if MODEL is None or len(FEATURE_NAMES) == 0:
        pytest.skip("Modelo/feature_names no disponibles para test_predict_happy_path")

    # Crear una instancia dummy con columnas válidas (valores arbitrarios)
    instance = {c: 0.0 for c in FEATURE_NAMES}
    payload = {"instances": [instance]}

    resp = client.post("/predict",
                       data=json.dumps(payload),
                       content_type="application/json")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "predictions" in data
    # proba puede ser None si el modelo no la soporta; en LogReg sí la soporta.
    assert "proba" in data


def test_predict_bad_payload(client):
    # Enviar payload inválido
    resp = client.post("/predict",
                       data=json.dumps({"foo": "bar"}),
                       content_type="application/json")
    assert resp.status_code == 400