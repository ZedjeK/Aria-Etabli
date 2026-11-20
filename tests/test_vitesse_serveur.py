"""appli/vitesse_serveur.py : le pilote llama-server, essayé avec un faux serveur qui imite le vrai."""

import json
import socket
import time

import pytest

from appli import mesures, vitesse_serveur
from appli.mesures import ErreurMesure
from appli.moteur import Moteur
from appli.vitesse_serveur import OptionsServeur
from tests.outils_gguf import STRING, TENSEUR_INCONNU, ecrire
from tests.outils_mesure import modele_gguf, modele_moe, moteur_faux

CLE_EXPERTS = "qwen35moe.expert_used_count"


@pytest.fixture(autouse=True)
def environnement(monkeypatch):
    """Carte graphique non consultée, sondage rapide : aucun test ne dépend de la machine qui le lance."""
    monkeypatch.setattr(mesures, "occupation_gpu", lambda: None)
    monkeypatch.setattr(mesures, "nom_gpu", lambda: "GPU d'essai")
    monkeypatch.setattr(vitesse_serveur, "SONDAGE", 0.05)


@pytest.fixture
def traces(monkeypatch, tmp_path):
    """Fichiers où le faux serveur note ses lancements et ses requêtes."""
    lancements, requetes = tmp_path / "lancements.jsonl", tmp_path / "requetes.jsonl"
    monkeypatch.setenv("FAUX_ARGS", str(lancements))
    monkeypatch.setenv("FAUX_REQUETES", str(requetes))

    class Traces:
        def lancements(self):
            return [json.loads(ligne) for ligne in lancements.read_text(encoding="utf-8").splitlines()] if lancements.exists() else []

        def requetes(self):
            return [json.loads(ligne) for ligne in requetes.read_text(encoding="utf-8").splitlines()] if requetes.exists() else []

    return Traces()


def port_ferme(port: int, delai: float = 5.0) -> bool:
    """Vrai quand plus rien n'écoute sur ce port : le serveur a bien été arrêté.

    On tente de s'y lier : cela n'aboutit que si plus personne ne l'occupe. Se connecter ne convient pas : sur Windows, une
    connexion vers un port fermé met environ une seconde à être refusée, et vers un port de la plage éphémère elle peut aboutir
    à une auto-connexion (le client reçoit le même port que sa cible), donc à un faux « ouvert ». Le système peut aussi mettre
    quelques millisecondes à libérer le port d'un processus tué : on réessaie jusqu'à `delai` secondes, et un port encore pris
    après cela est un vrai processus qui survit.
    """
    fin = time.monotonic() + delai
    while True:
        with socket.socket() as essai:
            try:
                essai.bind(("127.0.0.1", port))
                return True
            except OSError:
                pass
        if time.monotonic() >= fin:
            return False
        time.sleep(0.05)


def test_l_instrument_de_fermeture_voit_un_port_occupe_et_un_port_libre():
    with socket.socket() as ecoute:
        ecoute.bind(("127.0.0.1", 0))
        ecoute.listen()
        assert port_ferme(ecoute.getsockname()[1], delai=0.3) is False  # quelqu'un écoute : pas fermé
        port = ecoute.getsockname()[1]
    assert port_ferme(port) is True  # le socket est fermé : le port est libre


def valeur_de(commande: list[str], option: str) -> str:
    return commande[commande.index(option) + 1]


def test_la_ligne_de_commande_ecoute_en_local_avec_une_cle_et_le_reglage_d_experts(tmp_path):
    commande = vitesse_serveur.construire_commande(
        Moteur(tmp_path, None, ("C:\\moteur\\llama-server.exe",)), tmp_path / "m.gguf", 51234, "cle-secrete",
        OptionsServeur(couches_gpu=20, experts_cpu=2, contexte=2048, flash_attn="on", threads=4), CLE_EXPERTS, 6)
    assert commande == ["C:\\moteur\\llama-server.exe", "-m", str(tmp_path / "m.gguf"), "--host", "127.0.0.1", "--port",
                        "51234", "--api-key", "cle-secrete", "-np", "1", "--no-webui", "-c", "2048", "--override-kv",
                        f"{CLE_EXPERTS}=int:6", "-ngl", "20", "-ncmoe", "2", "-fa", "on", "-t", "4"]


def test_une_mesure_par_nombre_d_experts_avec_la_configuration_complete(tmp_path):
    modele = modele_moe(tmp_path)
    resultats = vitesse_serveur.mesurer_experts(moteur_faux(tmp_path), modele, (2, 4, 8),
                                                OptionsServeur(prompt=64, generation=16, repetitions=3, contexte=256))
    assert [r.configuration.reglages["experts_actifs"] for r in resultats] == [2, 4, 8]
    for resultat, experts in zip(resultats, (2, 4, 8)):
        pp, tg = resultat.mesures
        assert (pp.test, tg.test) == ("pp64", "tg16")
        # le faux serveur fait baisser tg avec les experts, à 0, 1 ou 2 jetons/s près (son écart d'une requête à l'autre)
        assert abs(tg.jetons_par_seconde - (300 - 10 * experts)) <= 2
        assert len(tg.echantillons) == 3 and tg.ecart_type > 0
        c = resultat.configuration
        assert c.outil == "serveur"
        assert c.modele["architecture"] == "qwen35moe" and c.modele["fichier"] == "moe.gguf"
        assert c.moteur["build"] == 12007 and c.moteur["commit"] == "fb5997274" and c.moteur["carte_graphique"] == "GPU d'essai"
        r = c.reglages
        assert (r["experts_actifs_fichier"], r["experts_totaux"], r["cle_experts"]) == (4, 16, CLE_EXPERTS)
        assert (r["contexte"], r["prompt_jetons"], r["generation_jetons"], r["temperature"], r["cache_prompt"],
                r["ignore_eos"]) == (256, 64, 16, 0, False, True)
    # Seul le nombre d'experts actifs diffère d'une mesure à l'autre : c'est ce que la règle de comparaison exige
    mesures.verifier_comparables([r.configuration for r in resultats], variables=("reglages.experts_actifs",))
    assert set(mesures.differences(resultats[0].configuration, resultats[1].configuration)) == {"reglages.experts_actifs"}


def test_le_serveur_recoit_le_reglage_sur_127_0_0_1_avec_une_cle_et_la_cle_ne_fuit_pas(tmp_path, traces):
    resultats = vitesse_serveur.mesurer_experts(moteur_faux(tmp_path), modele_moe(tmp_path), (6,),
                                                OptionsServeur(prompt=32, generation=8, repetitions=1, echauffement=0, contexte=128))
    (lancement,) = traces.lancements()
    assert valeur_de(lancement, "--override-kv") == f"{CLE_EXPERTS}=int:6"
    assert valeur_de(lancement, "--host") == "127.0.0.1"
    assert int(valeur_de(lancement, "--port")) > 1024
    assert "--no-webui" in lancement and valeur_de(lancement, "-np") == "1"
    cle_api = valeur_de(lancement, "--api-key")
    assert len(cle_api) >= 20
    assert cle_api not in json.dumps(resultats[0].en_dict())  # la clé de ce lancement n'est écrite nulle part


def test_chaque_valeur_a_son_serveur_et_sa_cle_et_le_port_est_ferme_ensuite(tmp_path, traces):
    vitesse_serveur.mesurer_experts(moteur_faux(tmp_path), modele_moe(tmp_path), (2, 8),
                                    OptionsServeur(prompt=32, generation=8, repetitions=1, echauffement=0, contexte=128))
    lancements = traces.lancements()
    assert [valeur_de(arguments, "--override-kv") for arguments in lancements] == [f"{CLE_EXPERTS}=int:2",
                                                                                   f"{CLE_EXPERTS}=int:8"]
    assert len({valeur_de(arguments, "--api-key") for arguments in lancements}) == 2
    assert all(port_ferme(int(valeur_de(arguments, "--port"))) for arguments in lancements)


def test_la_requete_de_mesure_est_exacte_et_l_echauffement_est_ignore(tmp_path, traces):
    resultat, = vitesse_serveur.mesurer_experts(
        moteur_faux(tmp_path), modele_moe(tmp_path), (4,),
        OptionsServeur(prompt=100, generation=40, repetitions=3, echauffement=2, contexte=512))
    requetes = traces.requetes()
    assert len(requetes) == 5  # 2 d'échauffement + 3 mesurées
    assert len(resultat.mesures[1].echantillons) == 3  # seules les 3 dernières comptent
    for requete in requetes:
        assert requete == {"prompt": 100, "n_predict": 40, "temperature": 0, "seed": 1, "cache_prompt": False,
                           "ignore_eos": True, "stream": False}


def test_un_proxy_configure_sur_la_machine_n_est_pas_utilise(tmp_path, monkeypatch):
    for nom in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(nom, "http://127.0.0.1:9")  # un proxy qui n'existe pas : tout passage par lui échouerait
    resultat, = vitesse_serveur.mesurer_experts(moteur_faux(tmp_path), modele_moe(tmp_path), (4,),
                                                OptionsServeur(prompt=32, generation=8, repetitions=1, echauffement=0, contexte=128))
    assert resultat.mesures[1].jetons_par_seconde > 0


def test_la_progression_est_rapportee(tmp_path):
    messages = []
    vitesse_serveur.mesurer_experts(moteur_faux(tmp_path), modele_moe(tmp_path), (4,),
                                    OptionsServeur(prompt=32, generation=8, repetitions=2, echauffement=1, contexte=128),
                                    progression=messages.append)
    assert messages == ["4 experts actifs : démarrage de llama-server...", "4 experts actifs : échauffement 1/1",
                        "4 experts actifs : mesure 1/2", "4 experts actifs : mesure 2/2"]


def test_les_avertissements_disent_la_vitesse_sans_la_qualite(tmp_path):
    resultats = vitesse_serveur.mesurer_experts(
        moteur_faux(tmp_path), modele_moe(tmp_path), (2, 4, 8),
        OptionsServeur(prompt=32, generation=8, repetitions=3, echauffement=0, contexte=128))
    moins, egal, plus = (r.avertissements for r in resultats)
    assert any("2 experts actifs au lieu de 4" in a and "qualité baisse" in a for a in moins)
    assert egal == ()
    assert any("8 experts actifs dépassent les 4" in a and "pas été entraîné" in a for a in plus)


def test_une_carte_occupee_est_signalee(tmp_path, monkeypatch):
    monkeypatch.setattr(mesures, "occupation_gpu", lambda: 80)
    resultat, = vitesse_serveur.mesurer_experts(moteur_faux(tmp_path), modele_moe(tmp_path), (4,),
                                                OptionsServeur(prompt=32, generation=8, repetitions=3, echauffement=0, contexte=128))
    assert resultat.occupation_gpu_avant == 80 and any("80 %" in a for a in resultat.avertissements)


def test_un_modele_dense_est_refuse_avant_de_lancer_quoi_que_ce_soit(tmp_path, traces):
    with pytest.raises(ErreurMesure, match="pas un mélange d'experts"):
        vitesse_serveur.mesurer_experts(moteur_faux(tmp_path), modele_gguf(tmp_path), (4,))
    assert traces.lancements() == []


@pytest.mark.parametrize("valeurs, extrait", [
    ((), "au moins une valeur"), ((0,), "entre 1 et 16"), ((17,), "entre 1 et 16"), ((-3,), "entre 1 et 16"),
    ((4, "8"), "entre 1 et 16"), ((True,), "entre 1 et 16"), (tuple(range(1, 34)), "Au plus 32"),
])
def test_des_valeurs_d_experts_absurdes_sont_refusees_avant_de_lancer(tmp_path, traces, valeurs, extrait):
    with pytest.raises(ErreurMesure, match=extrait):
        vitesse_serveur.mesurer_experts(moteur_faux(tmp_path), modele_moe(tmp_path), valeurs)
    assert traces.lancements() == []


def test_les_doublons_sont_retires_dans_l_ordre_donne():
    assert vitesse_serveur.verifier_valeurs([8, 4, 8, 2, 4], 16) == [8, 4, 2]


@pytest.mark.parametrize("options", [
    OptionsServeur(contexte=100, prompt=90, generation=20), OptionsServeur(repetitions=0), OptionsServeur(echauffement=-1),
    OptionsServeur(prompt=0), OptionsServeur(generation=0), OptionsServeur(delai_demarrage=1), OptionsServeur(threads=0),
    OptionsServeur(couches_gpu=-2), OptionsServeur(flash_attn="peut-etre"), OptionsServeur(contexte="4096"),
])
def test_des_options_absurdes_sont_refusees_avant_de_lancer(tmp_path, traces, options):
    with pytest.raises(ErreurMesure):
        vitesse_serveur.mesurer_experts(moteur_faux(tmp_path), modele_moe(tmp_path), (4,), options)
    assert traces.lancements() == []


def test_rien_n_est_lance_si_le_modele_est_absent_illisible_ou_le_moteur_sans_serveur(tmp_path, traces):
    with pytest.raises(ErreurMesure, match="n'existe pas"):
        vitesse_serveur.mesurer_experts(moteur_faux(tmp_path), tmp_path / "absent.gguf", (4,))
    faux = tmp_path / "faux.gguf"
    faux.write_bytes(b"pas un gguf, mais assez long pour passer la taille minimale du fichier")
    with pytest.raises(ErreurMesure, match="GGUF lisible"):
        vitesse_serveur.mesurer_experts(moteur_faux(tmp_path), faux, (4,))
    with pytest.raises(ErreurMesure, match="pas de llama-server"):
        vitesse_serveur.mesurer_experts(Moteur(tmp_path, ("bench.exe",), None), modele_moe(tmp_path), (4,))
    assert traces.lancements() == []


def test_un_serveur_qui_plante_au_demarrage_est_rapporte_avec_ses_dernieres_lignes(tmp_path):
    with pytest.raises(ErreurMesure, match=r"code 3.*avant d'être prêt.*failed to load model"):
        vitesse_serveur.mesurer_experts(moteur_faux(tmp_path), modele_moe(tmp_path, "plante.gguf"), (4,),
                                        OptionsServeur(prompt=32, generation=8, contexte=128))


def test_un_echec_au_demarrage_sur_un_type_de_tenseur_inconnu_donne_une_piste(tmp_path):
    inconnu = ecrire(tmp_path / "plante.gguf", [
        ("general.architecture", STRING, "qwen35moe"), ("qwen35moe.expert_count", 4, 16),
        ("qwen35moe.expert_used_count", 4, 4)], [("w", (256,), TENSEUR_INCONNU, bytes(64))])
    with pytest.raises(ErreurMesure, match=r"failed to load model.*Indice.*inconnu\(142\).*fork"):
        vitesse_serveur.mesurer_experts(moteur_faux(tmp_path), inconnu, (4,), OptionsServeur(prompt=32, generation=8, contexte=128))


def test_un_serveur_jamais_pret_est_arrete_apres_le_delai(tmp_path):
    serveur_lent = modele_moe(tmp_path, "lent.gguf")
    moteur = moteur_faux(tmp_path)
    commande = vitesse_serveur.construire_commande(moteur, serveur_lent, vitesse_serveur._port_libre(), "cle",
                                                   OptionsServeur(), CLE_EXPERTS, 4)
    port = int(valeur_de(commande, "--port"))
    debut = time.monotonic()
    with pytest.raises(ErreurMesure, match="pas prêt après 1 s"):
        with vitesse_serveur.Serveur(commande, port, "cle") as serveur:
            serveur.attendre_pret(1)
    assert time.monotonic() - debut < 15
    assert port_ferme(port)  # arrêté malgré l'erreur


@pytest.mark.parametrize("nom, extrait", [
    ("courtjetons", "au lieu des 8 demandés"), ("sanstimings", "pas de champ « timings »"), ("pasdejson", "Pas de réponse"),
])
def test_une_reponse_inattendue_est_refusee_et_le_serveur_est_quand_meme_arrete(tmp_path, traces, nom, extrait):
    with pytest.raises(ErreurMesure, match=extrait):
        vitesse_serveur.mesurer_experts(moteur_faux(tmp_path), modele_moe(tmp_path, f"{nom}.gguf"), (4,),
                                        OptionsServeur(prompt=32, generation=8, repetitions=2, echauffement=0, contexte=128))
    (lancement,) = traces.lancements()
    assert port_ferme(int(valeur_de(lancement, "--port")))


def test_un_executable_introuvable_est_dit(tmp_path):
    moteur = Moteur(tmp_path, None, (str(tmp_path / "absent.exe"),))
    with pytest.raises(ErreurMesure, match="Impossible de lancer llama-server"):
        vitesse_serveur.mesurer_experts(moteur, modele_moe(tmp_path), (4,), OptionsServeur(prompt=32, generation=8, contexte=128))


def test_une_mauvaise_cle_d_api_est_refusee_par_le_serveur(tmp_path):
    moteur = moteur_faux(tmp_path)
    port = vitesse_serveur._port_libre()
    commande = vitesse_serveur.construire_commande(moteur, modele_moe(tmp_path), port, "la-bonne-cle", OptionsServeur(), CLE_EXPERTS, 4)
    with vitesse_serveur.Serveur(commande, port, "une-autre-cle") as serveur:
        serveur.attendre_pret(30)  # /health n'exige pas de clé
        with pytest.raises(ErreurMesure, match="a répondu 401 à /props"):
            serveur.appeler("/props")

