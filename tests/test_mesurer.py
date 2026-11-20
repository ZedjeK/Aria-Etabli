"""appli/mesurer.py : la commande « mesurer bench », de la ligne de commande au code de retour."""

import json
import pathlib
import re
import sys

import pytest

from appli import __main__ as lancement
from appli import mesurer, mesures, vitesse_serveur
from tests.outils_mesure import modele_gguf, modele_moe, moteur_faux


@pytest.fixture(autouse=True)
def moteur_de_test(monkeypatch, tmp_path):
    """« --moteur » donne un dossier : ici, il mène au faux moteur. La carte graphique n'est pas consultée."""
    monkeypatch.setattr(mesures, "occupation_gpu", lambda: None)
    monkeypatch.setattr(mesures, "nom_gpu", lambda: "GPU d'essai")
    monkeypatch.setattr(vitesse_serveur, "SONDAGE", 0.05)
    monkeypatch.setattr(mesurer.moteurs, "reconnaitre", lambda dossier: moteur_faux(tmp_path))


def test_la_commande_en_texte(tmp_path, capsys):
    assert mesurer.main(["bench", str(modele_gguf(tmp_path)), "--moteur", str(tmp_path), "-r", "4"]) == 0
    sortie = capsys.readouterr().out
    for attendu in ("essai.gguf (llama, F16", "build 12007 (commit fb5997274), CUDA", "CPU d'essai ; GPU d'essai",
                    "n_gpu_layers = -1", "llama-bench, 4 répétition(s)", "pp512", "1000,00", "10,00", "tg128", "100,00"):
        assert attendu in sortie


def test_la_commande_en_json(tmp_path, capsys):
    assert mesurer.main(["bench", str(modele_gguf(tmp_path)), "--moteur", "x", "--json", "--profondeur", "1024",
                         "--ngl", "0", "--prompt", "64", "--generation", "0"]) == 0
    donnees = json.loads(capsys.readouterr().out)
    assert [m["test"] for m in donnees["mesures"]] == ["pp64@d1024"]
    assert donnees["configuration"]["outil"] == "bench"
    assert donnees["configuration"]["reglages"]["n_gpu_layers"] == 0
    assert donnees["avertissements"] == []  # 5 répétitions par défaut, carte non consultée : rien à signaler


def test_sortie_ecrit_le_json_et_refuse_d_ecraser(tmp_path, capsys):
    modele, sortie = modele_gguf(tmp_path), tmp_path / "resultat.json"
    assert mesurer.main(["bench", str(modele), "--moteur", "x", "--sortie", str(sortie)]) == 0
    capsys.readouterr()
    assert json.loads(sortie.read_text(encoding="utf-8"))["configuration"]["outil"] == "bench"
    contenu = sortie.read_bytes()
    assert mesurer.main(["bench", str(modele), "--moteur", "x", "--sortie", str(sortie)]) == 1
    assert "existe déjà" in capsys.readouterr().err
    assert sortie.read_bytes() == contenu


def test_les_codes_de_retour(tmp_path, capsys):
    modele = str(modele_gguf(tmp_path))
    assert mesurer.main(["--aide"]) == 0
    assert "Usage : python -m appli mesurer" in capsys.readouterr().out
    assert mesurer.main(["bench", "--aide"]) == 0
    capsys.readouterr()

    for arguments, attendu in [
        ([], "sous-commande"), (["zzz"], "Sous-commande inconnue : zzz"), (["bench"], "un seul fichier modèle"),
        (["bench", modele], "--moteur DOSSIER est obligatoire"), (["bench", modele, "--moteur", "x", "--bizarre"], "Option inconnue"),
        (["bench", modele, "--moteur"], "attend une valeur"), (["bench", modele, "--moteur", "x", "--ngl", "beaucoup"], "nombre entier"),
        (["bench", modele, "autre.gguf", "--moteur", "x"], "un seul fichier modèle"),
    ]:
        assert mesurer.main(arguments) == 2
        assert attendu in capsys.readouterr().err

    assert mesurer.main(["bench", str(tmp_path / "absent.gguf"), "--moteur", "x"]) == 1
    assert "Mesure impossible" in capsys.readouterr().err
    assert mesurer.main(["bench", modele, "--moteur", "x", "--ngl", "-5"]) == 1  # refusé par la vérification des bornes
    assert "couches sur la carte" in capsys.readouterr().err
    assert mesurer.main(["bench", str(modele_gguf(tmp_path, "plante.gguf")), "--moteur", "x"]) == 1
    assert "failed to load model" in capsys.readouterr().err


def test_un_nombre_negatif_est_une_valeur_pas_une_option(tmp_path, capsys):
    assert mesurer.main(["bench", str(modele_gguf(tmp_path)), "--moteur", "x", "--ngl", "-1", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["configuration"]["reglages"]["n_gpu_layers"] == -1


def test_un_dossier_qui_n_est_pas_un_moteur_est_dit(tmp_path, monkeypatch, capsys):
    monkeypatch.undo()  # retire le faux moteur : la vraie reconnaissance, sur un dossier vide
    monkeypatch.setattr(mesures, "occupation_gpu", lambda: None)
    assert mesurer.main(["bench", str(modele_gguf(tmp_path)), "--moteur", str(tmp_path)]) == 1
    assert "ni llama-bench" in capsys.readouterr().err


ARGS_EXPERTS = ["--moteur", "x", "--prompt", "32", "--generation", "8", "-r", "3", "--echauffement", "0", "--contexte", "128"]


def test_experts_en_texte_avec_la_comparaison_au_fichier(tmp_path, capsys):
    assert mesurer.main(["experts", str(modele_moe(tmp_path)), "--experts", "2,4,8", *ARGS_EXPERTS]) == 0
    sortie, erreur = capsys.readouterr()
    for attendu in ("moe.gguf (qwen35moe, Q4_K_M", "16 au total, 4 actifs dans le fichier (qwen35moe.expert_used_count)",
                    "llama-server (--override-kv), 3 répétition(s) après 0 d'échauffement", "GPU d'essai",
                    "pp jetons/s", "tg jetons/s", "référence", "Ces vitesses ne disent rien de la qualité"):
        assert attendu in sortie
    lignes = {int(ligne.split()[0]): ligne for ligne in sortie.splitlines() if ligne.strip()[:1].isdigit() and "jetons" not in ligne}
    assert set(lignes) == {2, 4, 8}
    assert "référence" in lignes[4]
    assert re.search(r"\+\d+,\d %", lignes[2]) and re.search(r"-\d+,\d %", lignes[8])  # moins d'experts : plus vite
    assert "démarrage de llama-server" in erreur  # la progression va sur la sortie d'erreur


def test_experts_en_json_laisse_la_sortie_standard_propre(tmp_path, capsys):
    assert mesurer.main(["experts", str(modele_moe(tmp_path)), "--experts", "4,6", *ARGS_EXPERTS, "--json"]) == 0
    sortie, erreur = capsys.readouterr()
    donnees = json.loads(sortie)  # que du JSON : la progression n'y est pas mélangée
    assert donnees["comparables"] is True
    assert [r["configuration"]["reglages"]["experts_actifs"] for r in donnees["resultats"]] == [4, 6]
    assert donnees["resultats"][0]["configuration"]["outil"] == "serveur"
    assert "mesure 3/3" in erreur


def test_experts_sortie_ecrit_le_json_et_refuse_d_ecraser(tmp_path, capsys):
    modele, sortie = modele_moe(tmp_path), tmp_path / "experts.json"
    assert mesurer.main(["experts", str(modele), "--experts", "4", *ARGS_EXPERTS, "--sortie", str(sortie)]) == 0
    capsys.readouterr()
    assert json.loads(sortie.read_text(encoding="utf-8"))["comparables"] is True
    contenu = sortie.read_bytes()
    assert mesurer.main(["experts", str(modele), "--experts", "4", *ARGS_EXPERTS, "--sortie", str(sortie)]) == 1
    assert "existe déjà" in capsys.readouterr().err
    assert sortie.read_bytes() == contenu


def test_experts_les_erreurs_d_usage_et_les_echecs(tmp_path, capsys):
    modele = str(modele_moe(tmp_path))
    for arguments, attendu in [
        (["experts", modele, "--moteur", "x"], "--experts 4,6,8 est obligatoire"),
        (["experts", modele, "--moteur", "x", "--experts", "4,x"], "séparés par des virgules"),
        (["experts", modele, "--experts", "4"], "--moteur DOSSIER est obligatoire"),
        (["experts", "--moteur", "x", "--experts", "4"], "un seul fichier modèle"),
        (["experts", modele, "--moteur", "x", "--experts", "4", "--profondeur", "1"], "Option inconnue : --profondeur"),
    ]:
        assert mesurer.main(arguments) == 2
        assert attendu in capsys.readouterr().err
    assert mesurer.main(["experts", str(modele_gguf(tmp_path)), "--experts", "4", *ARGS_EXPERTS]) == 1
    assert "pas un mélange d'experts" in capsys.readouterr().err
    assert mesurer.main(["experts", modele, "--experts", "99", *ARGS_EXPERTS]) == 1
    assert "entre 1 et 16" in capsys.readouterr().err
    assert mesurer.main(["experts", str(modele_moe(tmp_path, "plante.gguf")), "--experts", "4", *ARGS_EXPERTS]) == 1
    assert "failed to load model" in capsys.readouterr().err


def test_experts_refuse_de_presenter_des_mesures_non_comparables(tmp_path, monkeypatch, capsys):
    def melangees(*_a, **_k):
        raise mesures.ErreurMesure("Ces mesures ne sont pas comparables, leurs configurations diffèrent : test.")

    monkeypatch.setattr(mesurer.mesures, "verifier_comparables", melangees)
    assert mesurer.main(["experts", str(modele_moe(tmp_path)), "--experts", "4,6", *ARGS_EXPERTS]) == 1
    sortie, erreur = capsys.readouterr()
    assert "ne sont pas comparables" in erreur
    assert "tg jetons/s" not in sortie  # aucun tableau n'est affiché


def test_l_aide_decrit_les_deux_sous_commandes(capsys):
    assert mesurer.main(["--aide"]) == 0
    sortie = capsys.readouterr().out
    assert "mesurer bench" in sortie and "mesurer experts" in sortie and "--experts 4,6,8" in sortie
    assert mesurer.main(["experts", "--aide"]) == 0
    assert "--override-kv" in capsys.readouterr().out


def test_la_commande_ne_touche_pas_au_dossier_de_donnees(tmp_path, monkeypatch, capsys):
    maison = tmp_path / "maison"
    maison.mkdir()
    monkeypatch.setattr(pathlib.Path, "home", lambda: maison)
    modele = modele_gguf(tmp_path)
    avant = modele.read_bytes()
    assert mesurer.main(["bench", str(modele), "--moteur", "x"]) == 0
    capsys.readouterr()
    assert modele.read_bytes() == avant
    assert list(maison.iterdir()) == []


def test_la_commande_se_lance_avant_le_serveur_et_le_dossier_de_donnees(tmp_path, monkeypatch, capsys):
    def interdit(*_a, **_k):
        raise AssertionError("le dossier de données ne doit pas être touché par « mesurer »")

    monkeypatch.setattr(lancement.donnees, "assurer", interdit)
    monkeypatch.setattr(sys, "argv", ["Aria-Etabli", "mesurer", "bench", str(modele_gguf(tmp_path)), "--moteur", "x"])
    assert lancement.main() == 0
    assert "pp512" in capsys.readouterr().out
