"""appli/jeux.py : rien ne se télécharge sans accord, rien de non conforme n'est gardé, un seul hôte est joignable."""

import re
import sys
import urllib.error
import urllib.request

import pytest

from appli import __main__ as lancement
from appli import donnees, identite, jeux
from appli.jeux import ErreurJeu
from tests.outils_jeux import CONTENUS, JEU_ESSAI, URL, FauxReseau


def test_le_jeu_gsm8k_est_epingle_precisement():
    assert jeux.JEUX == {"gsm8k": jeux.GSM8K}
    assert [f.nom for f in jeux.GSM8K.fichiers] == ["test.jsonl", "train.jsonl"]
    for fichier in jeux.GSM8K.fichiers:
        assert re.fullmatch(r"[0-9a-f]{64}", fichier.sha256)
        assert fichier.taille > 0
        assert fichier.url.startswith("https://api.github.com/repos/openai/grade-school-math/contents/")
        assert fichier.url.endswith("?ref=3101c7d5072418e28b9008a6636bde82a006892c")  # le commit, pas une branche mobile
        jeux._valider_url(fichier.url)  # toutes ses adresses sont permises
    assert jeux.GSM8K.licence.startswith("MIT")


@pytest.mark.parametrize("url", [
    "http://api.github.com/x", "https://example.com/x", "https://api.github.com.evil.example/x",
    "https://evil.example/@api.github.com/x", "https://user@evil.example/x", "ftp://api.github.com/x", "file:///c:/x",
    "//api.github.com/x", "api.github.com/x",
])
def test_seul_https_vers_api_github_com_est_permis(url):
    with pytest.raises(ErreurJeu, match="Adresse refusée"):
        jeux._valider_url(url)


def test_une_redirection_vers_un_autre_hote_est_refusee():
    gestionnaire = jeux._RedirectionRestreinte()
    requete = urllib.request.Request(URL.format("test.jsonl"))
    for destination in ("https://raw.githubusercontent.com/x", "http://api.github.com/x", "https://evil.example/x"):
        with pytest.raises(ErreurJeu, match="Adresse refusée"):
            gestionnaire.redirect_request(requete, None, 302, "Found", {}, destination)
    suivante = gestionnaire.redirect_request(requete, None, 302, "Found", {}, "https://api.github.com/autre")
    assert suivante is not None and suivante.full_url == "https://api.github.com/autre"


def test_sans_accord_rien_n_est_telecharge_ni_cree(tmp_path):
    reseau = FauxReseau()
    with pytest.raises(ErreurJeu, match="accord explicite"):
        jeux.telecharger(JEU_ESSAI, tmp_path, lire=reseau)
    with pytest.raises(ErreurJeu, match="accord explicite"):
        jeux.telecharger(JEU_ESSAI, tmp_path, accord=False, lire=reseau)
    assert reseau.appels == [] and list(tmp_path.iterdir()) == []


def test_le_telechargement_range_des_fichiers_verifies_et_le_note_au_journal(tmp_path):
    reseau = FauxReseau()
    messages = []
    ecrits = jeux.telecharger(JEU_ESSAI, tmp_path, accord=True, lire=reseau, progression=messages.append)
    dossier = tmp_path / "jeux" / "essai"
    assert sorted(p.name for p in ecrits) == ["test.jsonl", "train.jsonl"]
    assert sorted(p.name for p in dossier.iterdir()) == ["test.jsonl", "train.jsonl"]  # aucun fichier .partiel
    assert (dossier / "test.jsonl").read_bytes() == CONTENUS["test.jsonl"]
    assert [(url, taille) for url, taille in reseau.appels] == [(f.url, f.taille) for f in JEU_ESSAI.fichiers]
    assert jeux.etat(JEU_ESSAI, tmp_path) == {"test.jsonl": "présent", "train.jsonl": "présent"}
    assert len(messages) == 2 and "test.jsonl" in " ".join(messages) and "train.jsonl" in " ".join(messages)
    derniere = donnees.lire_journal(tmp_path)[-1]
    assert derniere["type"] == "jeux" and "api.github.com" in derniere["bilan"] and "essai" in derniere["bilan"]


def test_un_jeu_deja_la_n_est_pas_retelecharge(tmp_path):
    jeux.telecharger(JEU_ESSAI, tmp_path, accord=True, lire=FauxReseau())
    reseau = FauxReseau()
    assert jeux.telecharger(JEU_ESSAI, tmp_path, accord=True, lire=reseau) == []
    assert reseau.appels == []


def test_seul_ce_qui_manque_est_telecharge(tmp_path):
    (tmp_path / "jeux" / "essai").mkdir(parents=True)
    (tmp_path / "jeux" / "essai" / "train.jsonl").write_bytes(CONTENUS["train.jsonl"])
    reseau = FauxReseau()
    ecrits = jeux.telecharger(JEU_ESSAI, tmp_path, accord=True, lire=reseau)
    assert [p.name for p in ecrits] == ["test.jsonl"]
    assert [url.split("?")[0].rsplit("/", 1)[-1] for url, _ in reseau.appels] == ["test.jsonl"]


def test_un_fichier_modifie_apres_coup_est_dit_corrompu_puis_remplace(tmp_path):
    jeux.telecharger(JEU_ESSAI, tmp_path, accord=True, lire=FauxReseau())
    cible = tmp_path / "jeux" / "essai" / "test.jsonl"
    cible.write_bytes(CONTENUS["test.jsonl"][:-1] + b"X")  # même taille, un octet changé
    assert jeux.etat(JEU_ESSAI, tmp_path)["test.jsonl"] == "corrompu"
    assert [f.nom for f in jeux.a_telecharger(JEU_ESSAI, tmp_path)] == ["test.jsonl"]
    jeux.telecharger(JEU_ESSAI, tmp_path, accord=True, lire=FauxReseau())
    assert cible.read_bytes() == CONTENUS["test.jsonl"]


def test_une_empreinte_differente_est_refusee_et_le_fichier_fautif_n_est_pas_garde(tmp_path):
    reseau = FauxReseau({**CONTENUS, "test.jsonl": b"x" * len(CONTENUS["test.jsonl"])})  # bonne taille, mauvais contenu
    with pytest.raises(ErreurJeu, match="SHA-256 ne correspond pas"):
        jeux.telecharger(JEU_ESSAI, tmp_path, accord=True, lire=reseau)
    dossier = tmp_path / "jeux" / "essai"
    assert sorted(p.name for p in dossier.iterdir()) == ["train.jsonl"]  # le fichier fautif n'est pas là, ni aucun « .partiel »
    assert jeux.etat(JEU_ESSAI, tmp_path) == {"train.jsonl": "présent", "test.jsonl": "absent"}  # le bon fichier reste valide


def test_une_taille_differente_est_refusee(tmp_path):
    reseau = FauxReseau({**CONTENUS, "test.jsonl": CONTENUS["test.jsonl"] + b"en trop"})
    with pytest.raises(ErreurJeu, match="octets reçus au lieu de"):
        jeux.telecharger(JEU_ESSAI, tmp_path, accord=True, lire=reseau)
    assert jeux.etat(JEU_ESSAI, tmp_path)["test.jsonl"] == "absent"


def test_une_ecriture_interrompue_ne_laisse_ni_fichier_ni_reste(tmp_path, monkeypatch):
    def echoue(*_args):
        raise OSError("disque plein")

    monkeypatch.setattr(jeux.os, "replace", echoue)
    with pytest.raises(ErreurJeu, match="Impossible d'écrire"):
        jeux.telecharger(JEU_ESSAI, tmp_path, accord=True, lire=FauxReseau())
    assert list((tmp_path / "jeux" / "essai").iterdir()) == []  # ni fichier, ni « .partiel »


def test_un_fichier_absent_ou_un_dossier_a_la_place_est_dit_absent_ou_corrompu(tmp_path):
    assert set(jeux.etat(JEU_ESSAI, tmp_path).values()) == {"absent"}
    (tmp_path / "jeux" / "essai" / "test.jsonl").mkdir(parents=True)  # un dossier de ce nom
    assert jeux.etat(JEU_ESSAI, tmp_path)["test.jsonl"] == "corrompu"


# ----- Le réseau lui-même (remplacé par un faux ouvreur : aucun test ne sort de la machine) -----

class _FauxOuvreur:
    def __init__(self, comportement):
        self.comportement = comportement
        self.requete = None

    def open(self, requete, timeout=None):
        self.requete = requete
        self.delai = timeout
        return self.comportement()


class _Reponse:
    def __init__(self, contenu: bytes):
        self.contenu = contenu

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, n: int) -> bytes:
        return self.contenu[:n]


def _ouvreur_de_test(monkeypatch, comportement) -> _FauxOuvreur:
    faux = _FauxOuvreur(comportement)
    monkeypatch.setattr(jeux.urllib.request, "build_opener", lambda *_gestionnaires: faux)
    return faux


def test_la_requete_est_anonyme_et_demande_le_contenu_brut(monkeypatch):
    faux = _ouvreur_de_test(monkeypatch, lambda: _Reponse(b"abc"))
    assert jeux._lire_url(URL.format("test.jsonl"), 10) == b"abc"
    entetes = {cle.lower(): valeur for cle, valeur in faux.requete.header_items()}
    assert entetes["accept"] == "application/vnd.github.raw+json"
    assert entetes["user-agent"] == f"{identite.EXE}/{identite.version()}"  # le nom et la version de l'appli, rien d'autre
    assert set(entetes) == {"accept", "user-agent"}  # ni jeton, ni cookie, ni identifiant
    assert faux.delai == jeux.DELAI_RESEAU


def test_un_fichier_plus_gros_que_prevu_est_refuse(monkeypatch):
    _ouvreur_de_test(monkeypatch, lambda: _Reponse(b"x" * 11))
    with pytest.raises(ErreurJeu, match="plus gros que prévu"):
        jeux._lire_url(URL.format("test.jsonl"), 10)


@pytest.mark.parametrize("erreur, extrait", [
    (urllib.error.HTTPError(URL, 403, "rate limit exceeded", {}, None), "limite de requêtes"),
    (urllib.error.HTTPError(URL, 404, "Not Found", {}, None), "404"),
    (urllib.error.URLError("pas de réseau"), "Impossible de joindre api.github.com"),
    (TimeoutError("trop long"), "Impossible de joindre api.github.com"),
])
def test_les_pannes_reseau_sont_dites_en_clair(monkeypatch, erreur, extrait):
    def echoue():
        raise erreur

    _ouvreur_de_test(monkeypatch, echoue)
    with pytest.raises(ErreurJeu, match=extrait):
        jeux._lire_url(URL.format("test.jsonl"), 10)


def test_une_adresse_non_permise_est_refusee_avant_toute_connexion(monkeypatch):
    def interdit(*_args):
        raise AssertionError("aucune connexion ne doit être tentée")

    monkeypatch.setattr(jeux.urllib.request, "build_opener", interdit)
    with pytest.raises(ErreurJeu, match="Adresse refusée"):
        jeux._lire_url("https://evil.example/x", 10)


# ----- La commande -----

@pytest.fixture
def commande(monkeypatch, tmp_path):
    """La commande travaille sur un dossier de données vide, avec le jeu d'essai, et ne peut pas sortir sur le réseau."""
    monkeypatch.setattr(identite, "dossier_donnees", lambda: tmp_path)
    monkeypatch.setattr(jeux, "JEUX", {"essai": JEU_ESSAI})
    reseau = FauxReseau()
    monkeypatch.setattr(jeux, "_lire_url", reseau)
    return reseau


def test_la_liste_ne_se_connecte_pas_et_ne_cree_rien(tmp_path, commande, capsys):
    assert jeux.main([]) == 0
    sortie = capsys.readouterr().out
    assert "essai : Jeu d'essai" in sortie and "MIT, pour les tests" in sortie and "absent" in sortie
    assert commande.appels == [] and list(tmp_path.iterdir()) == []


def test_telecharger_sans_accord_montre_le_plan_et_ne_fait_rien(tmp_path, commande, capsys):
    assert jeux.main(["telecharger", "essai"]) == 0
    sortie = capsys.readouterr().out
    assert "Voici ce qui serait téléchargé" in sortie and "SHA-256" in sortie and "api.github.com" in sortie
    assert str(tmp_path / "jeux" / "essai") in sortie and "Rien n'a été fait" in sortie
    assert commande.appels == [] and list(tmp_path.iterdir()) == []  # le dossier de données n'est même pas créé


def test_telecharger_avec_accord_telecharge_une_fois(tmp_path, commande, capsys):
    assert jeux.main(["telecharger", "essai", "--accord"]) == 0
    sortie, erreur = capsys.readouterr()
    assert "2 fichier(s) téléchargé(s) et vérifié(s)" in sortie and "Téléchargement de test.jsonl" in erreur
    assert len(commande.appels) == 2
    assert jeux.main(["telecharger", "essai", "--accord"]) == 0
    assert "déjà là, complet et conforme" in capsys.readouterr().out
    assert len(commande.appels) == 2  # pas de nouvelle connexion
    assert jeux.main([]) == 0 and "présent" in capsys.readouterr().out


def test_un_echec_de_telechargement_donne_le_code_1(tmp_path, commande, capsys, monkeypatch):
    def echoue(url, taille_max):
        raise ErreurJeu("GitHub a répondu 503 (Service Unavailable) : réessaie plus tard.")

    monkeypatch.setattr(jeux, "_lire_url", echoue)
    assert jeux.main(["telecharger", "essai", "--accord"]) == 1
    assert "Téléchargement impossible : GitHub a répondu 503" in capsys.readouterr().err


@pytest.mark.parametrize("arguments, extrait", [
    (["zzz"], "Sous-commande inconnue : zzz"), (["telecharger"], "un seul jeu"), (["telecharger", "a", "b"], "un seul jeu"),
    (["telecharger", "essai", "--vite"], "un seul jeu"), (["telecharger", "inconnu"], "Jeu inconnu : inconnu"),
])
def test_les_erreurs_d_usage(commande, capsys, arguments, extrait):
    assert jeux.main(arguments) == 2
    assert extrait in capsys.readouterr().err


def test_l_aide(commande, capsys):
    assert jeux.main(["--aide"]) == 0
    assert "telecharger JEU --accord" in capsys.readouterr().out


def test_la_commande_se_lance_avant_le_serveur_et_le_dossier_de_donnees(commande, monkeypatch, capsys):
    def interdit(*_a, **_k):
        raise AssertionError("le dossier de données ne doit pas être touché par « jeux »")

    monkeypatch.setattr(lancement.donnees, "assurer", interdit)
    monkeypatch.setattr(sys, "argv", ["Aria-Etabli", "jeux"])
    assert lancement.main() == 0
    assert "essai" in capsys.readouterr().out
