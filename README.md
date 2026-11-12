# Projet 3 — Information imparfaite, EBP et enchères

**Module :** Théorie des jeux et systèmes multi-agents  
**Groupe 3 :** ELATIFI Anas, ELMAHTAL Mohamed Yassine, ERRAFAI Achraf, ERRAOUI Wiame  
**Périmètre actuel :** Semaine 1 (S1) — cadrage, lecture théorique et préparation de l'environnement.

## Installation

Python 3.10+ est recommandé. Sur un ordinateur peu puissant :

```bash
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Installer [Ollama](https://ollama.com), puis télécharger le modèle léger :

```bash
ollama pull llama3.2:1b
export OLLAMA_MODEL=llama3.2:1b   # Windows PowerShell : $env:OLLAMA_MODEL="llama3.2:1b"
```

Le template d'agent lit `OLLAMA_MODEL` et utilise `llama3.2:1b` par défaut.

## Tests

```bash
pytest -q
```

Les tests et modules présents en S1 sont des squelettes préparatoires. L'implémentation du solveur et des expériences est planifiée pour S2.

## Structure S1

```text
docs/
├── avancement/rapport_s1.md
├── fiche_lecture.tex
├── gantt_previsionnel.mmd
├── gantt_previsionnel.png
└── rapport/
notebooks/
├── 01_theorie_prototype.ipynb
├── 02_outils_agent.ipynb
└── 03_experiences.ipynb
src/
├── agent_tj_sma.py
├── experiences.py
└── jeux.py
tests/
├── test_hors_ligne.py
└── test_jeux.py
```
