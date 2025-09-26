# app/utils.py
# -*- coding: utf-8 -*-

"""
Funciones auxiliares para validación de payloads y transformación de datos.
"""

from typing import List, Dict, Any, Tuple


def validate_instances(instances: Any, feature_names: List[str]) -> Tuple[bool, str]:
    """
    Valida que 'instances' sea una lista de diccionarios con las claves correctas.

    Parámetros:
        instances (Any): Objeto recibido en el JSON bajo la clave 'instances'.
        feature_names (List[str]): Lista de columnas esperadas por el modelo.

    Efectos:
        - No tiene efectos colaterales (solo validación).

    Retorna:
        (bool, str): (es_valido, mensaje_error_si_corresponde)
    """
    if not isinstance(instances, list) or len(instances) == 0:
        return False, "'instances' debe ser una lista no vacía."

    for i, row in enumerate(instances):
        if not isinstance(row, dict):
            return False, f"Elemento {i} de 'instances' no es un objeto JSON (dict)."
        missing = [c for c in feature_names if c not in row]
        extras = [c for c in row.keys() if c not in feature_names]
        if missing:
            return False, f"Faltan columnas en la instancia {i}: {missing}"
        if extras:
            return False, f"Columnas no reconocidas en la instancia {i}: {extras}"

        # Validar que los valores sean numéricos
        for c in feature_names:
            v = row[c]
            if not isinstance(v, (int, float)) and not isinstance(v, bool):
                return False, f"Valor no numérico para '{c}' en instancia {i}: {v}"
    return True, ""