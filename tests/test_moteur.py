"""appli/moteur.py : un dossier est un moteur s'il contient llama-bench ou llama-server."""

import pytest

from appli import moteur


def test_un_dossier_avec_les_deux_executables_est_reconnu(tmp_path):
    (tmp_path / f"llama-bench{moteur.SUFFIXE}").write_bytes(b"")
    (tmp_path / f"llama-server{moteur.SUFFIXE}").write_bytes(b"")
    reconnu = moteur.reconnaitre(tmp_path)
    assert reconnu.dossier == tmp_path
    assert reconnu.commande_bench == (str(tmp_path / f"llama-bench{moteur.SUFFIXE}"),)
    assert reconnu.commande_serveur == (str(tmp_path / f"llama-server{moteur.SUFFIXE}"),)


def test_un_moteur_peut_n_avoir_que_l_un_des_deux(tmp_path):
    (tmp_path / f"llama-server{moteur.SUFFIXE}").write_bytes(b"")
    reconnu = moteur.reconnaitre(tmp_path)
    assert reconnu.commande_bench is None and reconnu.commande_serveur is not None


def test_un_dossier_sans_executable_est_refuse_en_clair(tmp_path):
    with pytest.raises(moteur.ErreurMoteur, match="ni llama-bench"):
        moteur.reconnaitre(tmp_path)


def test_un_chemin_qui_n_est_pas_un_dossier_est_refuse(tmp_path):
    fichier = tmp_path / "x.txt"
    fichier.write_text("x")
    for chemin in (fichier, tmp_path / "absent"):
        with pytest.raises(moteur.ErreurMoteur, match="n'est pas un dossier"):
            moteur.reconnaitre(chemin)
