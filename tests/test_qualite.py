"""appli/qualite.py : le protocole GSM8K de lm-eval, la lecture des jeux, la note et l'incertitude."""

import json
import math

import pytest

from appli import jeux, qualite
from appli.qualite import ErreurQualite, Question
from tests.outils_jeux import CONTENUS, JEU_ESSAI, FauxReseau, ligne

BONNE_REPONSE = "Il en reste 9 chez lui. Donc 9 x 2 = 18. #### 18"


# ----- Les deux filtres de lm-eval -----

@pytest.mark.parametrize("texte, strict, flexible", [
    (" Il reste 9. 9 x 2 = 18. #### 18", "18", "18"),                      # le cas normal
    (" #### 70,000", "70,000", "70,000"),                                   # la virgule est gardée à l'extraction
    (" donc #### -5", "-5", "-5"),                                          # un nombre négatif
    (" #### 18 et ensuite 20", "18", "20"),                                 # strict : le premier ####, flexible : le dernier nombre
    (" le total est 50 puis 12 puis 7", qualite.INVALIDE, "7"),             # pas de ####, un seul chiffre en dernier
    (" la réponse est $50.", qualite.INVALIDE, "$50."),                     # le « $ » et le point sont retirés à la comparaison
    (" aucune réponse chiffrée ici", qualite.INVALIDE, qualite.INVALIDE),   # rien à extraire
    ("", qualite.INVALIDE, qualite.INVALIDE),
    (" #### abc", qualite.INVALIDE, qualite.INVALIDE),                      # « #### » sans nombre
    (" ####18", qualite.INVALIDE, "18"),                                    # il faut « #### » suivi d'une espace
    (" a #### 3.5 b #### 4", "3.5", "4"),                                   # deux « #### » : strict garde le premier
])
def test_les_deux_filtres_extraient_comme_lm_eval(texte, strict, flexible):
    assert qualite.extraire_strict(texte) == strict
    assert qualite.extraire_flexible(texte) == flexible


@pytest.mark.parametrize("texte, strict, flexible", [
    (" #### 18", True, True),
    (" donc 18.", False, True),                       # pas de ####, mais le dernier nombre est le bon (point final ignoré)
    (" #### 18 et le reste 20", True, False),         # le cas vu sur gemma : bon à #### mais le dernier nombre est faux
    (" #### 20", False, False),
    (" 18,000", False, False),                        # attendu 18 : « 18000 » ne vaut pas 18
    (" aucune idée", False, False),
])
def test_la_note_compare_apres_normalisation(texte, strict, flexible):
    resultat = qualite.noter(texte, BONNE_REPONSE)
    assert (resultat["strict"], resultat["flexible"]) == (strict, flexible)
    assert resultat["attendu"] == "18"


def test_les_nombres_sont_compares_sans_virgule_ni_dollar_ni_point_final():
    attendu = "Beaucoup de calcul. #### 70000"
    assert qualite.noter(" #### 70,000", attendu)["strict"] is True
    assert qualite.noter(" cela fait $70,000.", attendu)["flexible"] is True
    assert qualite.noter(" #### 7000", attendu)["strict"] is False


def test_la_note_donne_ce_qui_a_ete_extrait():
    resultat = qualite.noter(" #### 18 puis 20", BONNE_REPONSE)
    assert (resultat["extrait_strict"], resultat["extrait_flexible"]) == ("18", "20")


# ----- Le prompt -----

def test_le_prompt_a_le_format_de_lm_eval():
    exemples = [Question("Combien font 1 et 1 ?", "1 + 1 = 2. #### 2"), Question("Combien font 2 et 2 ?", "2 + 2 = 4. #### 4")]
    assert qualite.construire_prompt(exemples, "Combien font 3 et 3 ?") == (
        "Question: Combien font 1 et 1 ?\nAnswer: 1 + 1 = 2. #### 2\n\n"
        "Question: Combien font 2 et 2 ?\nAnswer: 2 + 2 = 4. #### 4\n\n"
        "Question: Combien font 3 et 3 ?\nAnswer:")


def test_le_protocole_est_ecrit_en_clair_et_serialisable():
    protocole = qualite.PROTOCOLE
    json.dumps(protocole)  # s'enregistre avec chaque mesure
    assert protocole["mode"] == "complétion brute, sans gabarit de chat, sans réflexion"
    assert protocole["arrets"] == ["Question:", "</s>", "<|im_end|>"] and protocole["jetons_max"] == 256
    assert protocole["temperature"] == 0 and protocole["graine"] == 1234
    assert protocole["exemples"] == {"nombre": 5, "choix": "les 5 premiers de train.jsonl"}
    assert qualite.ARRETS == ("Question:", "</s>", "<|im_end|>")


# ----- La lecture des jeux -----

def test_lire_jsonl_lit_les_questions(tmp_path):
    chemin = tmp_path / "t.jsonl"
    chemin.write_bytes(CONTENUS["test.jsonl"])
    questions = qualite.lire_jsonl(chemin)
    assert len(questions) == 4
    assert questions[0] == Question("test 1 ?", "On calcule test 1 ?. #### 10")


@pytest.mark.parametrize("contenu, extrait", [
    ("pas du json\n", r"t\.jsonl, ligne 1"),
    ('{"question": "q"}\n', r"ligne 1 : ce n'est pas une question"),
    ('{"question": "q", "answer": "pas de dièses"}\n', r"ligne 1 : la réponse n'est pas de la forme"),
    ('{"question": 3, "answer": "x #### 1"}\n', r"ligne 1 : la réponse n'est pas de la forme"),
    ('[1, 2]\n', r"ligne 1 : ce n'est pas une question"),
    (ligne("a", "1") + "\n\n" + ligne("b", "2") + "\n", r"ligne 2 : ce n'est pas une question"),  # une ligne vide au milieu
])
def test_une_ligne_qui_n_a_pas_la_forme_attendue_est_refusee_avec_son_numero(tmp_path, contenu, extrait):
    chemin = tmp_path / "t.jsonl"
    chemin.write_text(contenu, encoding="utf-8")
    with pytest.raises(ErreurQualite, match=extrait):
        qualite.lire_jsonl(chemin)


def test_un_fichier_vide_absent_ou_illisible(tmp_path):
    vide = tmp_path / "vide.jsonl"
    vide.write_bytes(b"")
    assert qualite.lire_jsonl(vide) == []
    with pytest.raises(ErreurQualite, match="Impossible de lire"):
        qualite.lire_jsonl(tmp_path / "absent.jsonl")
    binaire = tmp_path / "binaire.jsonl"
    binaire.write_bytes(b"\xff\xfe\x00 pas de l'utf-8")
    with pytest.raises(ErreurQualite, match="Impossible de lire"):
        qualite.lire_jsonl(binaire)


@pytest.fixture
def racine_avec_jeu(tmp_path):
    jeux.telecharger(JEU_ESSAI, tmp_path, accord=True, lire=FauxReseau())
    return tmp_path


def test_charger_donne_les_5_premiers_exemples_et_les_n_premieres_questions(racine_avec_jeu):
    exemples, questions = qualite.charger_gsm8k(3, racine_avec_jeu, JEU_ESSAI)
    assert [e.question for e in exemples] == [f"exemple {i} ?" for i in range(1, 6)]  # le 6e d'entraînement n'est pas pris
    assert [q.question for q in questions] == ["test 1 ?", "test 2 ?", "test 3 ?"]  # dans l'ordre du fichier, toujours les mêmes


def test_charger_refuse_un_jeu_absent_ou_modifie(tmp_path, racine_avec_jeu):
    with pytest.raises(ErreurQualite, match="n'est pas téléchargé.*jeux telecharger gsm8k"):
        qualite.charger_gsm8k(3, tmp_path / "ailleurs", JEU_ESSAI)
    (racine_avec_jeu / "jeux" / "essai" / "test.jsonl").write_bytes(CONTENUS["test.jsonl"][:-1] + b"X")
    with pytest.raises(ErreurQualite, match="ne correspond plus à l'empreinte"):
        qualite.charger_gsm8k(3, racine_avec_jeu, JEU_ESSAI)


@pytest.mark.parametrize("nombre", [0, -1, 1320, 2.5, "10", True, None])
def test_charger_refuse_un_nombre_de_questions_absurde(racine_avec_jeu, nombre):
    with pytest.raises(ErreurQualite, match="entre 1 et 1319"):
        qualite.charger_gsm8k(nombre, racine_avec_jeu, JEU_ESSAI)


def test_charger_refuse_un_jeu_plus_court_que_prevu(racine_avec_jeu):
    with pytest.raises(ErreurQualite, match="plus court que prévu"):
        qualite.charger_gsm8k(5, racine_avec_jeu, JEU_ESSAI)  # le jeu d'essai n'a que 4 questions de test


# ----- L'incertitude -----

def test_wilson_aux_bornes_a_une_formule_fermee():
    z2 = qualite.Z_95 ** 2
    bas, haut = qualite.intervalle_wilson(0, 20)
    assert bas == 0.0 and haut == pytest.approx(z2 / (20 + z2))
    bas, haut = qualite.intervalle_wilson(20, 20)
    assert haut == 1.0 and bas == pytest.approx(20 / (20 + z2))


def test_wilson_valeurs_de_reference():
    bas, haut = qualite.intervalle_wilson(10, 20)
    assert (round(bas, 4), round(haut, 4)) == (0.2993, 0.7007)
    bas, haut = qualite.intervalle_wilson(1, 20)
    assert (round(bas, 3), round(haut, 3)) == (0.009, 0.236)


@pytest.mark.parametrize("bonnes, total", [(0, 1), (1, 1), (3, 10), (37, 50), (120, 200), (1000, 1319)])
def test_wilson_est_coherent(bonnes, total):
    bas, haut = qualite.intervalle_wilson(bonnes, total)
    p = bonnes / total
    assert 0.0 <= bas <= p <= haut <= 1.0
    miroir_bas, miroir_haut = qualite.intervalle_wilson(total - bonnes, total)
    assert (bas, haut) == pytest.approx((1 - miroir_haut, 1 - miroir_bas))  # symétrique : réussites et échecs


def test_plus_de_questions_donne_un_intervalle_plus_etroit():
    largeurs = [h - b for b, h in (qualite.intervalle_wilson(n // 2, n) for n in (20, 100, 200, 1319))]
    assert largeurs == sorted(largeurs, reverse=True) and largeurs[0] > 0.4 and largeurs[-1] < 0.06


@pytest.mark.parametrize("bonnes, total", [(0, 0), (-1, 10), (11, 10), (1, -5)])
def test_une_proportion_impossible_est_refusee(bonnes, total):
    with pytest.raises(ErreurQualite, match="n'existe pas"):
        qualite.intervalle_wilson(bonnes, total)


def test_le_score_donne_les_bonnes_reponses_la_proportion_et_l_intervalle():
    notes = [{"strict": True, "flexible": True}] * 10 + [{"strict": False, "flexible": True}] * 5 \
        + [{"strict": False, "flexible": False}] * 5
    strict, flexible = qualite.score(notes, "strict"), qualite.score(notes, "flexible")
    assert (strict["bonnes"], strict["total"], strict["proportion"]) == (10, 20, 0.5)
    assert (flexible["bonnes"], flexible["proportion"]) == (15, 0.75)
    assert strict["wilson_bas"] < 0.5 < strict["wilson_haut"] and math.isclose(strict["wilson_bas"], 0.2993, abs_tol=1e-4)
    with pytest.raises(ErreurQualite, match="Filtre inconnu"):
        qualite.score(notes, "autre")
    with pytest.raises(ErreurQualite):
        qualite.score([], "strict")
