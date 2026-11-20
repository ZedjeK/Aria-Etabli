"""appli/vitesse_bench.py : le pilote llama-bench, essayé avec un faux moteur qui imite le vrai."""

import json
import sys
from pathlib import Path

import pytest

from appli import mesures, vitesse_bench
from appli.mesures import ErreurMesure
from appli.moteur import Moteur
from appli.vitesse_bench import OptionsBench
from tests.outils_gguf import STRING, TENSEUR_INCONNU, ecrire
from tests.outils_mesure import modele_gguf, moteur_faux

REEL = Path(__file__).parent / "donnees" / "llama-bench-reel.json"


@pytest.fixture(autouse=True)
def carte_libre(monkeypatch):
    """Aucun test ne doit dépendre de la carte graphique de la machine qui le lance."""
    monkeypatch.setattr(mesures, "occupation_gpu", lambda: None)


def test_la_ligne_de_commande_est_une_liste_sans_shell(tmp_path):
    modele = tmp_path / "mon modèle.gguf"
    commande = vitesse_bench.construire_commande(
        Moteur(tmp_path, ("C:\\moteur\\llama-bench.exe",), None), modele,
        OptionsBench(couches_gpu=0, experts_cpu=2, profondeur=4096, prompt=64, generation=16, repetitions=3,
                     flash_attn="on", threads=4))
    assert commande == ["C:\\moteur\\llama-bench.exe", "-m", str(modele), "-o", "json", "-r", "3", "-p", "64", "-n", "16",
                        "-d", "4096", "-ngl", "0", "-ncmoe", "2", "-fa", "on", "-t", "4"]


def test_sans_option_facultative_le_moteur_garde_ses_defauts(tmp_path):
    moteur = moteur_faux(tmp_path)
    commande = vitesse_bench.construire_commande(moteur, tmp_path / "m.gguf", OptionsBench())
    assert commande == [*moteur.commande_bench, "-m", str(tmp_path / "m.gguf"), "-o", "json", "-r", "5", "-p", "512",
                        "-n", "128", "-d", "0"]


def test_une_mesure_complete(tmp_path):
    modele = modele_gguf(tmp_path)
    resultat = vitesse_bench.mesurer(moteur_faux(tmp_path), modele, OptionsBench(repetitions=4))
    assert [m.test for m in resultat.mesures] == ["pp512", "tg128"]
    pp, tg = resultat.mesures
    assert (pp.jetons_par_seconde, pp.ecart_type, len(pp.echantillons)) == (1000.0, 10.0, 4)
    assert (tg.jetons_par_seconde, tg.ecart_type) == (100.0, 1.0)

    c = resultat.configuration
    assert c.outil == "bench"
    assert c.modele["fichier"] == "essai.gguf" and c.modele["architecture"] == "llama"  # le nom seulement, pas le chemin
    assert tmp_path.name not in json.dumps(c.en_dict())  # aucun dossier de la machine dans la configuration
    assert c.moteur == {"commit": "fb5997274", "build": 12007, "backends": "CUDA", "processeur": "CPU d'essai",
                        "carte_graphique": "GPU d'essai"}
    assert c.reglages["n_gpu_layers"] == -1 and c.reglages["n_depth"] == 0 and c.reglages["repack"] is True
    for exclu in ("model_filename", "avg_ts", "samples_ts", "n_prompt", "build_number", "cpu_info"):
        assert exclu not in c.reglages
    assert resultat.date.startswith("20")
    assert resultat.avertissements == ()


def test_les_options_arrivent_au_moteur_et_a_la_configuration(tmp_path, monkeypatch):
    fichier_args = tmp_path / "args.json"
    monkeypatch.setenv("FAUX_ARGS", str(fichier_args))
    options = OptionsBench(couches_gpu=0, experts_cpu=3, profondeur=4096, prompt=64, generation=0, flash_attn="on", threads=4)
    resultat = vitesse_bench.mesurer(moteur_faux(tmp_path), modele_gguf(tmp_path), options)
    assert [m.test for m in resultat.mesures] == ["pp64@d4096"]  # pas de génération demandée, profondeur dans le nom
    r = resultat.configuration.reglages
    assert (r["n_gpu_layers"], r["n_cpu_moe"], r["n_depth"], r["n_threads"], r["flash_attn"]) == (0, 3, 4096, 4, 1)
    recu = json.loads(fichier_args.read_text(encoding="utf-8"))
    assert recu[recu.index("-ngl") + 1] == "0" and recu[recu.index("-d") + 1] == "4096"


def test_generation_seule(tmp_path):
    resultat = vitesse_bench.mesurer(moteur_faux(tmp_path), modele_gguf(tmp_path), OptionsBench(prompt=0, generation=32))
    assert [m.test for m in resultat.mesures] == ["tg32"]


def test_le_vrai_format_de_llama_bench_est_lu():
    enregistrements = vitesse_bench._lire_json(REEL.read_text(encoding="utf-8"))
    pp, tg = (vitesse_bench._mesure(e) for e in enregistrements)
    assert (pp.test, round(pp.jetons_par_seconde, 1), len(pp.echantillons)) == ("pp32", 744.9, 2)
    assert (tg.test, round(tg.jetons_par_seconde, 1), round(tg.ecart_type, 1)) == ("tg16", 163.4, 0.2)


def test_des_lignes_avant_et_apres_le_json_ne_gênent_pas(tmp_path):
    resultat = vitesse_bench.mesurer(moteur_faux(tmp_path), modele_gguf(tmp_path, "bruit.gguf"), OptionsBench())
    assert [m.test for m in resultat.mesures] == ["pp512", "tg128"]


def test_un_moteur_qui_plante_est_rapporte_avec_son_message(tmp_path):
    with pytest.raises(ErreurMesure, match="code 3.*failed to load model"):
        vitesse_bench.mesurer(moteur_faux(tmp_path), modele_gguf(tmp_path, "plante.gguf"))


def test_un_echec_sur_un_type_de_tenseur_inconnu_donne_une_piste(tmp_path):
    inconnu = ecrire(tmp_path / "plante.gguf", [("general.architecture", STRING, "qwen35")],
                     [("w", (256,), TENSEUR_INCONNU, bytes(64))])
    with pytest.raises(ErreurMesure, match=r"failed to load model.*Indice.*inconnu\(142\).*fork"):
        vitesse_bench.mesurer(moteur_faux(tmp_path), inconnu)
    connu = modele_gguf(tmp_path, "plante.gguf")  # même échec du moteur, mais tous les types sont connus : pas d'indice
    with pytest.raises(ErreurMesure) as erreur:
        vitesse_bench.mesurer(moteur_faux(tmp_path), connu)
    assert "Indice" not in str(erreur.value)


def test_une_sortie_sans_json_est_refusee(tmp_path):
    with pytest.raises(ErreurMesure, match="pas de tableau JSON"):
        vitesse_bench.mesurer(moteur_faux(tmp_path), modele_gguf(tmp_path, "pasjson.gguf"))


def test_un_champ_manquant_dans_la_sortie_est_dit():
    sans_moyenne = json.dumps([{"n_prompt": 1, "n_gen": 0, "stddev_ts": 1.0, "samples_ts": [1.0]}])
    with pytest.raises(ErreurMesure, match="avg_ts"):
        vitesse_bench._mesure(vitesse_bench._lire_json(sans_moyenne)[0])


def test_un_moteur_trop_lent_est_arrete():
    with pytest.raises(ErreurMesure, match="pas fini en 1 s"):
        vitesse_bench._lancer([sys.executable, "-c", "import time; time.sleep(30)"], 1)


def test_un_executable_introuvable_est_dit(tmp_path):
    moteur = Moteur(tmp_path, (str(tmp_path / "absent.exe"),), None)
    with pytest.raises(ErreurMesure, match="Impossible de lancer"):
        vitesse_bench.mesurer(moteur, modele_gguf(tmp_path))


def test_rien_n_est_lance_si_le_modele_est_absent_ou_illisible(tmp_path, monkeypatch):
    fichier_args = tmp_path / "args.json"
    monkeypatch.setenv("FAUX_ARGS", str(fichier_args))
    with pytest.raises(ErreurMesure, match="n'existe pas"):
        vitesse_bench.mesurer(moteur_faux(tmp_path), tmp_path / "absent.gguf")
    faux = tmp_path / "faux.gguf"
    faux.write_bytes(b"pas un gguf, mais assez long pour passer la taille minimale du fichier")
    with pytest.raises(ErreurMesure, match="GGUF lisible"):
        vitesse_bench.mesurer(moteur_faux(tmp_path), faux)
    assert not fichier_args.exists()


def test_un_moteur_sans_bench_est_refuse(tmp_path):
    with pytest.raises(ErreurMesure, match="pas de llama-bench"):
        vitesse_bench.mesurer(Moteur(tmp_path, None, ("serveur.exe",)), modele_gguf(tmp_path))


@pytest.mark.parametrize("options", [
    OptionsBench(couches_gpu=-2), OptionsBench(experts_cpu=-1), OptionsBench(profondeur=-1), OptionsBench(prompt=-5),
    OptionsBench(generation=10 ** 9), OptionsBench(repetitions=0), OptionsBench(threads=0), OptionsBench(delai_max=1),
    OptionsBench(prompt=0, generation=0), OptionsBench(flash_attn="peut-etre"), OptionsBench(couches_gpu=True),
    OptionsBench(profondeur="4096"),
])
def test_des_options_absurdes_sont_refusees_avant_de_lancer(tmp_path, monkeypatch, options):
    fichier_args = tmp_path / "args.json"
    monkeypatch.setenv("FAUX_ARGS", str(fichier_args))
    with pytest.raises(ErreurMesure):
        vitesse_bench.mesurer(moteur_faux(tmp_path), modele_gguf(tmp_path), options)
    assert not fichier_args.exists()


def test_une_carte_occupee_est_signalee_seulement_si_la_mesure_l_utilise(tmp_path, monkeypatch):
    monkeypatch.setattr(mesures, "occupation_gpu", lambda: 80)
    sur_la_carte = vitesse_bench.mesurer(moteur_faux(tmp_path), modele_gguf(tmp_path))
    assert sur_la_carte.occupation_gpu_avant == 80
    assert any("80 %" in a for a in sur_la_carte.avertissements)
    sur_le_processeur = vitesse_bench.mesurer(moteur_faux(tmp_path), modele_gguf(tmp_path), OptionsBench(couches_gpu=0))
    assert sur_le_processeur.occupation_gpu_avant == 80
    assert sur_le_processeur.avertissements == ()


def test_une_ou_deux_repetitions_sont_signalees(tmp_path):
    resultat = vitesse_bench.mesurer(moteur_faux(tmp_path), modele_gguf(tmp_path), OptionsBench(repetitions=1))
    assert any("1 répétition" in a for a in resultat.avertissements)


def test_la_mesure_ne_modifie_pas_le_modele_et_n_ecrit_rien(tmp_path):
    modele = modele_gguf(tmp_path)
    avant = modele.read_bytes()
    vitesse_bench.mesurer(moteur_faux(tmp_path), modele)
    assert modele.read_bytes() == avant
    assert sorted(p.name for p in tmp_path.iterdir()) == ["essai.gguf"]
