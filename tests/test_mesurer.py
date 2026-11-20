"""appli/mesurer.py : la commande « mesurer bench », de la ligne de commande au code de retour."""

import json
import pathlib
import sys

import pytest

from appli import __main__ as lancement
from appli import mesurer, mesures
from tests.outils_mesure import modele_gguf, moteur_faux


@pytest.fixture(autouse=True)
def moteur_de_test(monkeypatch, tmp_path):
    """« --moteur » donne un dossier : ici, il mène au faux moteur. La carte graphique n'est pas consultée."""
    monkeypatch.setattr(mesures, "occupation_gpu", lambda: None)
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
