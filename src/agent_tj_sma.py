"""Template d'agent local LangChain/Ollama pour le projet.

L'intégration opérationnelle sera développée après la validation du prototype S2.
"""

from __future__ import annotations

import os
from typing import Any

DEFAULT_MODEL = "llama3.2:1b"


def nom_modele() -> str:
    """Lire le modèle Ollama local configuré pour le projet."""
    return os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)


def create_agent(model_name: str | None = None, **_: Any) -> Any:
    """Créer le squelette d'agent; l'assemblage LangChain est prévu en S2."""
    raise NotImplementedError(
        f"Le template est configuré pour Ollama ({model_name or nom_modele()}); "
        "l'agent sera assemblé en S2."
    )
