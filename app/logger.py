# app/logger.py
# -*- coding: utf-8 -*-

"""
Configura un logger consistente para toda la app.
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Crea y devuelve un logger con formato uniforme.

    Parámetros:
        name (str): Nombre lógico del logger (p.ej. __name__ del módulo).

    Efectos:
        - Configura el handler de stream a stdout.
        - Establece nivel INFO por defecto.

    Retorna:
        logging.Logger: Instancia configurada.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.propagate = False
    return logger