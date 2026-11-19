import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pytest
from jeux import enchere, equilibres_ebp


def sender(t, m, a):
    # Jeu de signalisation jouet avec préférences communes pour le bon diagnostic.
    return 1.0 if (t == 'H' and a == 'accepter') or (t == 'L' and a == 'refuser') else 0.0


def receiver(t, m, a):
    return sender(t, m, a)


def test_ebp_separateur():
    eq = equilibres_ebp(['H','L'], [.6,.4], ['x','y'], ['accepter','refuser'], sender, receiver)
    assert any(e['nature'] == 'separateur' for e in eq)
    for e in eq:
        for belief in e['croyances'].values():
            assert sum(belief.values()) == pytest.approx(1)


def test_prior_invalide():
    with pytest.raises(ValueError):
        equilibres_ebp(['H','L'], [.7,.7], ['x','y'], ['a','b'], sender, receiver)


@pytest.mark.parametrize('n', [2,3,5])
def test_encheres_recette(n):
    for prix in ['premier', 'second']:
        alpha = (n-1)/n if prix == 'premier' else 1
        result = enchere(n, alpha, N=100_000, seed=42, prix=prix)
        assert abs(result['recette_moyenne'] - (n-1)/(n+1)) < 0.015
        assert result['ic95_recette'][0] <= result['recette_moyenne'] <= result['ic95_recette'][1]


def test_enchere_reproductible():
    assert enchere(3, 2/3, N=1000, seed=4) == enchere(3, 2/3, N=1000, seed=4)


def test_parametres_enchere():
    with pytest.raises(ValueError):
        enchere(1, 0.5)


@pytest.mark.parametrize('prior', [[.9,.1], [.3,.7]])
def test_bayes_sur_chemin_et_hors_chemin(prior):
    # Exemple jouet: toutes les croyances sur chemin sont déterminées par Bayes.
    eq = equilibres_ebp(['H','L'], prior, ['x','y'], ['accepter','refuser'], sender, receiver)
    assert eq
    for e in eq:
        for m, croyance in e['croyances'].items():
            assert sum(croyance.values()) == pytest.approx(1)
            types_qui_envoient = [t for t, choix in e['emetteur'].items() if choix == m]
            masse = sum(prior[['H','L'].index(t)] for t in types_qui_envoient)
            if masse > 0:
                for typ in ['H','L']:
                    attendu = prior[['H','L'].index(typ)] / masse if typ in types_qui_envoient else 0
                    assert croyance[typ] == pytest.approx(attendu)


def test_ebp_hors_chemin_grille():
    eq = equilibres_ebp(['H','L'], [.6,.4], ['x','y'], ['accepter','refuser'], sender, receiver, grille=.2)
    assert eq
    for e in eq:
        for croyance in e['croyances'].values():
            assert 0 <= croyance['H'] <= 1
            assert 0 <= croyance['L'] <= 1


@pytest.mark.parametrize('n', [2,3,5])
def test_encheres_equilibre_et_second_prix(n):
    from jeux import analyser_deviations
    for prix in ['premier','second']:
        alpha = (n-1)/n if prix == 'premier' else 1.0
        result = analyser_deviations(n, alphas=[alpha], N=5000, seed=12, prix=prix)
        assert result['deviations'][0]['ecart_vs_equilibre'] == pytest.approx(0)
        assert result['alpha_equilibre'] == pytest.approx(alpha)


def test_encheres_intervalles_theorie():
    # Les IC sont des estimations aléatoires : une réalisation peut ne pas
    # couvrir la valeur vraie. On teste une tolérance statistique raisonnable.
    for n in [2,3,5]:
        for prix in ['premier','second']:
            alpha = (n-1)/n if prix == 'premier' else 1
            r = enchere(n, alpha, N=100_000, seed=123, prix=prix)
            lo, hi = r['ic95_recette']
            theorique = (n-1)/(n+1)
            assert lo - .01 <= theorique <= hi + .01


from jeux import equilibria_biere_quiche, decision_recepteur, gain_emetteur_biere_quiche

def _pool(eq, message, croyance_hors, p_fort=0.9):
    autre = "Quiche" if message == "Biere" else "Biere"
    return [e for e in eq if e["nature"] == "poolant"
            and set(e["emetteur"].values()) == {message}
            and abs(e["croyances"][autre]["Faible"] - croyance_hors) < 1e-8]

def test_biere_quiche_pool_biere_croyance_06_et_non_02():
    eq = equilibria_biere_quiche(0.9)
    assert _pool(eq, "Biere", 0.6)
    assert not _pool(eq, "Biere", 0.2)
    assert all(e["recepteur"]["Quiche"] == "Combattre" for e in _pool(eq, "Biere", 0.6))

def test_biere_quiche_pool_quiche_existe():
    eq = equilibria_biere_quiche(0.9)
    assert _pool(eq, "Quiche", 0.6)

def test_biere_quiche_bayes_sur_chemin():
    for p in (0.9, 0.3):
        for e in equilibria_biere_quiche(p):
            for message in ("Biere", "Quiche"):
                if set(e["emetteur"].values()) == {message}:
                    assert e["croyances"][message]["Fort"] == pytest.approx(p)
                    assert e["croyances"][message]["Faible"] == pytest.approx(1-p)

def test_seuil_combat():
    assert decision_recepteur(.5) == "Indifferent"
    assert decision_recepteur(.51) == "Combattre"
    assert decision_recepteur(.2) == "Ne_pas_combattre"

def test_critere_intuitif_pool_quiche():
    # Faible ne peut jamais gagner plus que 3 (son gain en pooling Quiche).
    assert gain_emetteur_biere_quiche("Faible", "Biere", "Ne_pas_combattre") == 2
    # Fort passe de 2 à 3 si Bière est interprétée comme venant du Fort.
    assert gain_emetteur_biere_quiche("Fort", "Quiche", "Ne_pas_combattre") == 2
    assert gain_emetteur_biere_quiche("Fort", "Biere", "Ne_pas_combattre") == 3
