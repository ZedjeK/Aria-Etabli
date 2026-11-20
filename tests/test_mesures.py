"""appli/mesures.py : la configuration complète, et le refus de comparer l'incomparable."""

import subprocess

import pytest

from appli import gguf, mesures
from appli.mesures import Configuration, ErreurMesure
from tests.outils_gguf import STRING, TENSEUR_F32, TENSEUR_Q8_0, U32, ecrire


def configuration(**changements) -> Configuration:
    modele = {"fichier": "m.gguf", "quantification": "Q4_K_M", "architecture": "qwen35moe"}
    moteur = {"build": 12007, "commit": "fb5997274", "backends": "CUDA"}
    reglages = {"n_gpu_layers": -1, "n_depth": 0, "flash_attn": -1}
    outil = "bench"
    for cle, valeur in changements.items():
        bloc, _, nom = cle.partition("__")
        if bloc == "outil":
            outil = valeur
        else:
            {"modele": modele, "moteur": moteur, "reglages": reglages}[bloc][nom] = valeur
    return Configuration(outil, modele, moteur, reglages)


def test_deux_configurations_identiques_ne_different_pas():
    assert mesures.differences(configuration(), configuration()) == {}
    mesures.verifier_comparables([configuration(), configuration(), configuration()])


def test_les_differences_nomment_la_cle_et_les_deux_valeurs():
    assert mesures.differences(configuration(), configuration(reglages__n_gpu_layers=20, modele__quantification="Q8_0")) == {
        "modele.quantification": ("Q4_K_M", "Q8_0"), "reglages.n_gpu_layers": (-1, 20)}


def test_une_cle_presente_d_un_seul_cote_est_une_difference():
    avec_en_plus = configuration(reglages__repack=True)
    assert mesures.differences(configuration(), avec_en_plus) == {"reglages.repack": (mesures.ABSENT, True)}


def test_on_peut_comparer_quand_seule_la_variable_attendue_change():
    mesures.verifier_comparables([configuration(reglages__n_gpu_layers=v) for v in (10, 20, 30)],
                                 variables=("reglages.n_gpu_layers",))


def test_une_autre_difference_que_la_variable_est_refusee_en_clair():
    melangees = [configuration(), configuration(reglages__n_gpu_layers=20, moteur__build=12100)]
    with pytest.raises(ErreurMesure, match="ne sont pas comparables") as erreur:
        mesures.verifier_comparables(melangees, variables=("reglages.n_gpu_layers",))
    assert "moteur.build : 12007 contre 12100" in str(erreur.value)
    assert "n_gpu_layers" not in str(erreur.value)


def test_deux_outils_differents_ne_se_comparent_pas():
    with pytest.raises(ErreurMesure, match="outil : 'bench' contre 'serveur'"):
        mesures.verifier_comparables([configuration(), configuration(outil="serveur")])


def test_le_modele_et_le_materiel_comptent_dans_la_comparaison():
    with pytest.raises(ErreurMesure, match="modele.quantification"):
        mesures.verifier_comparables([configuration(), configuration(modele__quantification="Q2_K")])
    with pytest.raises(ErreurMesure, match="moteur.backends"):
        mesures.verifier_comparables([configuration(), configuration(moteur__backends="CPU")])


def test_la_configuration_s_ecrit_a_plat_et_en_blocs():
    c = configuration()
    assert c.plat()["reglages.n_depth"] == 0 and c.plat()["outil"] == "bench"
    assert c.en_dict() == {"outil": "bench", "modele": c.modele, "moteur": c.moteur, "reglages": c.reglages}


def test_le_modele_est_decrit_d_apres_l_inspecteur(tmp_path):
    chemin = ecrire(tmp_path / "m.gguf", [("general.architecture", STRING, "llama"), ("general.file_type", U32, 7)],
                    [("a", (64, 2), TENSEUR_Q8_0, bytes(136)), ("b", (4,), TENSEUR_F32, bytes(16))])
    description = mesures.decrire_modele(gguf.lire(chemin))
    assert description == {"fichier": "m.gguf", "taille_octets": chemin.stat().st_size, "architecture": "llama",
                           "quantification": "Q8_0", "types_de_tenseurs": ["Q8_0", "F32"], "parametres": 132}


@pytest.mark.parametrize("sortie, attendu", [("47\n", 47), ("0\n", 0), ("[N/A]\n", None), ("", None)])
def test_occupation_gpu_lit_nvidia_smi(monkeypatch, sortie, attendu):
    monkeypatch.setattr(mesures.shutil, "which", lambda nom: "nvidia-smi.exe")
    monkeypatch.setattr(mesures.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, sortie, ""))
    assert mesures.occupation_gpu() == attendu


def test_occupation_gpu_est_none_sans_nvidia_smi_ou_s_il_plante(monkeypatch):
    monkeypatch.setattr(mesures.shutil, "which", lambda nom: None)
    assert mesures.occupation_gpu() is None
    monkeypatch.setattr(mesures.shutil, "which", lambda nom: "nvidia-smi.exe")

    def plante(*_a, **_k):
        raise OSError("introuvable")

    monkeypatch.setattr(mesures.subprocess, "run", plante)
    assert mesures.occupation_gpu() is None


def test_l_avertissement_gpu_ne_vaut_que_si_la_carte_sert_et_est_occupee():
    assert mesures.avertissement_gpu(80, True) != []
    assert mesures.avertissement_gpu(80, False) == []
    assert mesures.avertissement_gpu(mesures.SEUIL_GPU_OCCUPE - 1, True) == []
    assert mesures.avertissement_gpu(None, True) == []
    assert "80 %" in mesures.avertissement_gpu(80, True)[0]
