"""Interfaces préparatoires pour les expériences du projet (S1)."""

from __future__ import annotations

from typing import Any


def preparer_experience(**parametres: Any) -> dict[str, Any]:
    """Retourner la configuration d'une expérience future sans l'exécuter."""
    return {"statut": "preparee", "parametres": parametres}
