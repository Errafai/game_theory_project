"""Signatures des fonctions de théorie des jeux prévues pour la S2.

La S1 contient uniquement le cadrage et les interfaces de travail.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


def equilibres_ebp(
    types: Sequence[float],
    prior: Sequence[float],
    messages: Sequence[str],
    actions: Sequence[str],
    u_emetteur: Callable[..., float],
    u_recepteur: Callable[..., float],
    grille: float = 0.1,
) -> list[dict[str, Any]]:
    """Énumérer les équilibres bayésiens parfaits (implémentation prévue en S2)."""
    raise NotImplementedError("Le solveur EBP sera implémenté en S2.")


def enchere(
    n: int,
    alpha: float,
    N: int = 100_000,
    seed: int = 0,
    prix: str = "premier",
) -> dict[str, Any]:
    """Préparer une simulation d'enchère (implémentation prévue en S2)."""
    raise NotImplementedError("La simulation d'enchères sera implémentée en S2.")
