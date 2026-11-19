"""Projet 3 : solveur EBP pur et simulation d'enchères scellées."""
from __future__ import annotations
from itertools import product
from collections.abc import Callable, Sequence
from typing import Any
import numpy as np


def equilibres_ebp(types: Sequence[Any], prior: Sequence[float], messages: Sequence[str],
                   actions: Sequence[str], u_emetteur: Callable[..., float],
                   u_recepteur: Callable[..., float], grille: float = 0.1) -> list[dict[str, Any]]:
    """Énumère les EBP en stratégies pures, croyances hors chemin discrétisées.

    Gains : u_emetteur(type, message, action), u_recepteur(type, message, action).
    Les croyances hors chemin sont des points d'une grille du simplexe.
    Pour de petits jeux finis uniquement (énumération exponentielle).
    """
    types, messages, actions = list(types), list(messages), list(actions)
    p = np.asarray(prior, dtype=float)
    if not types or not messages or not actions or len(p) != len(types):
        raise ValueError("Types, messages, actions et prior doivent être cohérents et non vides")
    if not np.all(np.isfinite(p)) or np.any(p < 0) or not np.isclose(p.sum(), 1):
        raise ValueError("prior doit être une distribution de probabilités")
    k = round(1 / grille) if grille > 0 else 0
    if not 1 <= k <= 100 or not np.isclose(k * grille, 1):
        raise ValueError("grille doit diviser 1 (ex. 0.1, 0.05)")
    if len(types) * len(messages) > 12 or len(actions) ** len(messages) * len(messages) ** len(types) > 100000:
        raise ValueError("Jeu trop grand pour l'énumération exhaustive")

    def compositions(total, n):
        if n == 1:
            yield (total,)
        else:
            for i in range(total + 1):
                for tail in compositions(total - i, n - 1):
                    yield (i,) + tail

    beliefs_grid = [np.array(c, dtype=float) / k for c in compositions(k, len(types))]
    equilibria = []
    for sender in product(range(len(messages)), repeat=len(types)):
        masses = [sum(p[t] for t in range(len(types)) if sender[t] == m) for m in range(len(messages))]
        on_path = [None if masses[m] < 1e-12 else np.array(
            [p[t] * (sender[t] == m) / masses[m] for t in range(len(types))])
            for m in range(len(messages))]
        off = [m for m, mass in enumerate(masses) if mass < 1e-12]
        for off_beliefs in product(beliefs_grid, repeat=len(off)):
            beliefs = list(on_path)
            for m, b in zip(off, off_beliefs):
                beliefs[m] = b
            for receiver in product(range(len(actions)), repeat=len(messages)):
                # Rationalité séquentielle du récepteur, y compris hors chemin.
                if any(sum(beliefs[m][t] * u_recepteur(types[t], messages[m], actions[receiver[m]])
                           for t in range(len(types))) + 1e-9 <
                       max(sum(beliefs[m][t] * u_recepteur(types[t], messages[m], actions[a])
                               for t in range(len(types))) for a in range(len(actions)))
                       for m in range(len(messages))):
                    continue
                # Incitations de chaque type de l'émetteur (même ceux de probabilité nulle).
                if any(u_emetteur(types[t], messages[sender[t]], actions[receiver[sender[t]]]) + 1e-9 <
                       max(u_emetteur(types[t], messages[m], actions[receiver[m]])
                           for m in range(len(messages))) for t in range(len(types))):
                    continue
                equilibria.append({
                    "emetteur": {str(types[t]): messages[sender[t]] for t in range(len(types))},
                    "recepteur": {messages[m]: actions[receiver[m]] for m in range(len(messages))},
                    "croyances": {messages[m]: {str(types[t]): round(float(beliefs[m][t]), 10)
                                                  for t in range(len(types))} for m in range(len(messages))},
                    "nature": "separateur" if len(set(sender)) == len(types) else
                              "poolant" if len(set(sender)) == 1 else "partiel",
                })
    return equilibria


def enchere(n: int, alpha: float, N: int = 100_000, seed: int = 0,
            prix: str = "premier") -> dict[str, Any]:
    """Simule des enchères indépendantes U[0,1], mises alpha*v pour le joueur 0.

    Les autres joueurs suivent l'équilibre symétrique du format d'enchère.
    En cas d'égalité, le gagnant est tiré uniformément parmi les ex aequo.
    """
    if not isinstance(n, int) or n < 2 or n > 100:
        raise ValueError("n doit être un entier entre 2 et 100")
    if not isinstance(N, int) or N < 1 or N > 1_000_000:
        raise ValueError("N doit être entre 1 et 1 000 000")
    if not np.isfinite(alpha) or alpha < 0 or alpha > 1:
        raise ValueError("alpha doit être entre 0 et 1")
    if prix not in {"premier", "second"}:
        raise ValueError("prix doit être 'premier' ou 'second'")
    rng = np.random.default_rng(seed)
    # Calcul par lots pour maîtriser la mémoire.
    total_profit = total_revenue = total_revenue_sq = 0.0
    processed = 0
    while processed < N:
        size = min(10_000, N - processed)
        values = rng.random((size, n))
        baseline = (n - 1) / n if prix == "premier" else 1.0
        bids = values * baseline
        bids[:, 0] = alpha * values[:, 0]
        high = bids.max(axis=1)
        ties = bids == high[:, None]
        # Tirage uniforme parmi les joueurs ex aequo.
        scores = rng.random((size, n))
        scores[~ties] = -1
        winner = scores.argmax(axis=1)
        payment = high if prix == "premier" else np.partition(bids, -2, axis=1)[:, -2]
        utility = np.where(winner == 0, values[:, 0] - payment, 0.0)
        total_profit += utility.sum()
        total_revenue += payment.sum()
        total_revenue_sq += np.square(payment).sum()
        processed += size
    mean = total_revenue / N
    var = max(0.0, total_revenue_sq / N - mean ** 2)
    se = np.sqrt(var / N)
    return {"n": n, "alpha": alpha, "N": N, "seed": seed, "prix": prix,
            "gain_moyen_joueur_0": float(total_profit / N),
            "recette_moyenne": float(mean),
            "ic95_recette": [float(mean - 1.96 * se), float(mean + 1.96 * se)],
            "recette_theorique_equilibre": (n - 1) / (n + 1),
            "alpha_equilibre": (n - 1) / n if prix == "premier" else 1.0}


def analyser_deviations(n: int, alphas=None, N: int = 100_000, seed: int = 42,
                        prix: str = "premier") -> dict[str, Any]:
    """Compare les gains d'un joueur qui dévie à ceux de la stratégie d'équilibre.

    Les adversaires gardent leur stratégie d'équilibre ; les simulations
    partagent la même graine pour faciliter la comparaison (bruit Monte-Carlo).
    Une différence simulée positive ne prouve pas une déviation profitable.
    """
    if alphas is None:
        alphas = np.linspace(0, 1, 21)
    alpha_eq = (n - 1) / n if prix == "premier" else 1.0
    baseline = enchere(n, alpha_eq, N=N, seed=seed, prix=prix)
    rows = []
    for alpha in alphas:
        res = enchere(n, float(alpha), N=N, seed=seed, prix=prix)
        rows.append({"alpha": float(alpha),
                     "gain_moyen": res["gain_moyen_joueur_0"],
                     "ecart_vs_equilibre": res["gain_moyen_joueur_0"] - baseline["gain_moyen_joueur_0"]})
    return {"n": n, "prix": prix, "alpha_equilibre": alpha_eq,
            "gain_equilibre": baseline["gain_moyen_joueur_0"], "deviations": rows}


# Modélisation pédagogique classique de Beer or Quiche (Cho-Kreps).
# IMPORTANT : les gains exacts ne figurent pas dans les supports fournis.
# L'émetteur gagne 2 si le récepteur ne combat pas, plus 1 s'il choisit
# son petit-déjeuner préféré (Fort : Bière ; Faible : Quiche).
# Le récepteur gagne 1 s'il combat un Faible ou épargne un Fort, 0 sinon.

def gain_emetteur_biere_quiche(type_joueur: str, message: str, action: str) -> float:
    if type_joueur not in {"Fort", "Faible"} or message not in {"Biere", "Quiche"} or action not in {"Combattre", "Ne_pas_combattre"}:
        raise ValueError("Type, message ou action inconnu")
    prefere = "Biere" if type_joueur == "Fort" else "Quiche"
    return float(2 * (action == "Ne_pas_combattre") + (message == prefere))


def gain_recepteur_biere_quiche(type_joueur: str, message: str, action: str) -> float:
    if type_joueur not in {"Fort", "Faible"} or message not in {"Biere", "Quiche"} or action not in {"Combattre", "Ne_pas_combattre"}:
        raise ValueError("Type, message ou action inconnu")
    return float((type_joueur == "Faible" and action == "Combattre") or
                 (type_joueur == "Fort" and action == "Ne_pas_combattre"))


def equilibria_biere_quiche(p_fort: float = 0.9, grille: float = 0.1) -> list[dict[str, Any]]:
    if not 0 <= p_fort <= 1:
        raise ValueError("p_fort doit appartenir à [0,1]")
    return equilibres_ebp(["Fort", "Faible"], [p_fort, 1-p_fort],
                          ["Biere", "Quiche"], ["Combattre", "Ne_pas_combattre"],
                          gain_emetteur_biere_quiche, gain_recepteur_biere_quiche, grille)


def decision_recepteur(p_faible: float) -> str:
    if not 0 <= p_faible <= 1:
        raise ValueError("La croyance doit appartenir à [0,1]")
    if abs(p_faible - 0.5) < 1e-12:
        return "Indifferent"
    return "Combattre" if p_faible > 0.5 else "Ne_pas_combattre"
