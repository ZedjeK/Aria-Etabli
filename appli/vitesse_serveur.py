"""Le pilote de vitesse avec llama-server : fait varier le nombre d'experts actifs d'un modèle MoE.

llama-bench n'a pas d'--override-kv ; llama-server, si. Pour chaque valeur d'experts actifs, ce module lance un llama-server en
processus enfant avec `--override-kv <arch>.expert_used_count=int:N` (le modèle n'est jamais écrit : le réglage vit dans les
arguments de lancement), mesure, puis l'arrête.

Le serveur n'écoute que sur 127.0.0.1, sur un port libre, avec une clé d'API tirée au hasard pour ce seul lancement et jamais
écrite nulle part. Les requêtes ne passent par aucun proxy. Le serveur est arrêté dans tous les cas, même sur une erreur. Sa
sortie est lue en continu (un tuyau qu'on ne vide pas finit par bloquer un serveur bavard) : seules ses dernières lignes sont
gardées, pour expliquer une panne.

La mesure : /completion avec un prompt donné en jetons (taille exacte), `ignore_eos` (la génération va jusqu'au bout),
température 0, sans cache de prompt. Les vitesses sont celles que le serveur déclare dans `timings`. Ce protocole n'est pas celui de
llama-bench : les deux familles de chiffres ne se comparent pas (champ « outil » de la configuration).
"""

import collections
import json
import math
import platform
import re
import secrets
import socket
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from appli import gguf, inspecteur, mesures
from appli.mesures import Configuration, ErreurMesure, Mesure, Resultat
from appli.moteur import Moteur

HOTE = "127.0.0.1"
SEED = 1
SONDAGE = 0.5  # secondes entre deux demandes « le serveur est-il prêt ? »
MAX_VALEURS = 32
TEXTE_BASE = "Le chat dort sur le canapé pendant que la pluie tombe doucement sur la ville. "
FLASH_ATTN = ("on", "off", "auto")

def _ouvreur() -> urllib.request.OpenerDirector:
    """Jamais de proxy : les requêtes ne vont que vers 127.0.0.1, et un proxy configuré sur la machine n'a rien à y voir.

    Créé à chaque requête : l'ouvreur par défaut lit les variables de proxy à sa création, pas à l'usage."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


@dataclass(frozen=True)
class OptionsServeur:
    couches_gpu: int | None = None  # -ngl ; None : le défaut du moteur
    experts_cpu: int | None = None  # -ncmoe
    contexte: int = 4096  # -c : doit contenir le prompt et la génération
    prompt: int = 512  # jetons de prompt de chaque requête
    generation: int = 128  # jetons générés par requête
    repetitions: int = 5  # requêtes mesurées
    echauffement: int = 1  # requêtes faites avant, et ignorées
    flash_attn: str | None = None
    threads: int | None = None
    delai_demarrage: int = 600  # secondes pour que le modèle soit chargé
    delai_requete: int = 900  # secondes pour une requête

    def verifier(self) -> None:
        def borne(nom: str, valeur: Any, bas: int, haut: int) -> None:
            if valeur is None:
                return
            if isinstance(valeur, bool) or not isinstance(valeur, int) or not bas <= valeur <= haut:
                raise ErreurMesure(f"{nom} doit être un entier entre {bas} et {haut} (reçu : {valeur!r}).")

        borne("Les couches sur la carte", self.couches_gpu, -1, 10_000)
        borne("Les experts sur le processeur", self.experts_cpu, 0, 10_000)
        borne("Le contexte", self.contexte, 64, 10_000_000)
        borne("Le nombre de jetons de prompt", self.prompt, 1, 10_000_000)
        borne("Le nombre de jetons à générer", self.generation, 1, 10_000_000)
        borne("Le nombre de répétitions", self.repetitions, 1, 1000)
        borne("Le nombre de requêtes d'échauffement", self.echauffement, 0, 100)
        borne("Le nombre de threads", self.threads, 1, 1024)
        borne("Le délai de démarrage", self.delai_demarrage, 10, 86_400)
        borne("Le délai d'une requête", self.delai_requete, 10, 86_400)
        if isinstance(self.contexte, int) and isinstance(self.prompt, int) and isinstance(self.generation, int) \
                and self.contexte < self.prompt + self.generation + 16:
            raise ErreurMesure(f"Le contexte ({self.contexte}) doit contenir le prompt ({self.prompt}) et la génération "
                               f"({self.generation}) : au moins {self.prompt + self.generation + 16}.")
        if self.flash_attn is not None and self.flash_attn not in FLASH_ATTN:
            raise ErreurMesure(f"La flash attention doit être « on », « off » ou « auto » (reçu : {self.flash_attn!r}).")


def cles_experts(fichier: gguf.Fichier) -> tuple[str, int | None, int]:
    """(clé du nombre d'experts actifs, sa valeur dans le fichier, nombre d'experts totaux), lus dans le fichier."""
    experts = inspecteur.synthese(fichier)["experts"]
    cle, totaux = experts["actifs"]["cle"], experts["totaux"]["valeur"]
    if not experts["moe"] or cle is None or not totaux:
        raise ErreurMesure("Ce modèle n'est pas un mélange d'experts : le fichier n'indique pas de nombre d'experts actifs "
                           "(clé <architecture>.expert_used_count) ni d'experts totaux.")
    return cle, experts["actifs"]["valeur"], totaux


def verifier_valeurs(valeurs, totaux: int) -> list[int]:
    liste = list(valeurs)
    if not liste:
        raise ErreurMesure("Donne au moins une valeur d'experts actifs.")
    if len(liste) > MAX_VALEURS:
        raise ErreurMesure(f"Au plus {MAX_VALEURS} valeurs d'experts actifs à la fois (reçu : {len(liste)}).")
    for valeur in liste:
        if isinstance(valeur, bool) or not isinstance(valeur, int) or not 1 <= valeur <= totaux:
            raise ErreurMesure(f"Un nombre d'experts actifs doit être un entier entre 1 et {totaux}, le nombre d'experts du "
                               f"fichier (reçu : {valeur!r}).")
    return list(dict.fromkeys(liste))  # sans doublon, dans l'ordre donné


def construire_commande(moteur: Moteur, modele: Path, port: int, cle_api: str, options: OptionsServeur,
                        cle_experts: str, experts: int) -> list[str]:
    if moteur.commande_serveur is None:
        raise ErreurMesure(f"Ce moteur n'a pas de llama-server (dossier {moteur.dossier}).")
    commande = [*moteur.commande_serveur, "-m", str(modele), "--host", HOTE, "--port", str(port), "--api-key", cle_api,
                "-np", "1", "--no-webui", "-c", str(options.contexte), "--override-kv", f"{cle_experts}=int:{experts}"]
    for drapeau, valeur in (("-ngl", options.couches_gpu), ("-ncmoe", options.experts_cpu), ("-fa", options.flash_attn),
                            ("-t", options.threads)):
        if valeur is not None:
            commande += [drapeau, str(valeur)]
    return commande


def _port_libre() -> int:
    with socket.socket() as essai:
        essai.bind((HOTE, 0))
        return essai.getsockname()[1]


class Serveur:
    """Un llama-server enfant : se lance avec `with`, et est arrêté à la sortie du bloc quoi qu'il arrive."""

    def __init__(self, commande: list[str], port: int, cle_api: str):
        self.commande, self.port, self.cle_api = commande, port, cle_api
        self.journal: collections.deque[str] = collections.deque(maxlen=200)
        self.processus: subprocess.Popen | None = None

    def __enter__(self) -> "Serveur":
        try:
            self.processus = subprocess.Popen(
                self.commande, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True,
                encoding="utf-8", errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as erreur:
            raise ErreurMesure(f"Impossible de lancer llama-server : {erreur.strerror or erreur}") from erreur
        threading.Thread(target=self._lire_sortie, daemon=True).start()
        return self

    def __exit__(self, *_exception) -> None:
        self.arreter()

    def _lire_sortie(self) -> None:
        for ligne in self.processus.stdout:
            self.journal.append(ligne.rstrip())

    def fin_du_journal(self) -> str:
        lignes = [ligne.strip() for ligne in list(self.journal) if ligne.strip()][-3:]
        return " | ".join(lignes) or "aucun message"

    def arreter(self) -> None:
        if self.processus is None or self.processus.poll() is not None:
            return
        self.processus.terminate()
        try:
            self.processus.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.processus.kill()
            self.processus.wait()

    def _requete(self, methode: str, chemin: str, corps: dict | None, avec_cle: bool, delai: float) -> Any:
        entetes = {"Content-Type": "application/json"}
        if avec_cle:
            entetes["Authorization"] = f"Bearer {self.cle_api}"
        requete = urllib.request.Request(f"http://{HOTE}:{self.port}{chemin}", method=methode, headers=entetes,
                                         data=None if corps is None else json.dumps(corps).encode("utf-8"))
        with _ouvreur().open(requete, timeout=delai) as reponse:
            return json.loads(reponse.read().decode("utf-8", errors="replace"))

    def appeler(self, chemin: str, corps: dict | None = None, delai: float = 60) -> Any:
        try:
            return self._requete("POST" if corps is not None else "GET", chemin, corps, True, delai)
        except urllib.error.HTTPError as erreur:
            raise ErreurMesure(f"llama-server a répondu {erreur.code} à {chemin}.") from None
        except (urllib.error.URLError, OSError, TimeoutError, ValueError) as erreur:
            mort = self.processus is not None and self.processus.poll() is not None
            raise ErreurMesure(f"Pas de réponse de llama-server à {chemin} ({erreur})."
                               + (f" Il s'est arrêté : {self.fin_du_journal()}" if mort else "")) from None

    def _est_pret(self) -> bool:
        try:
            return self._requete("GET", "/health", None, False, 5).get("status") == "ok"
        except (urllib.error.URLError, OSError, TimeoutError, ValueError, AttributeError):
            return False  # 503 pendant le chargement du modèle, ou le serveur ne répond pas encore

    def attendre_pret(self, delai: float) -> None:
        fin = time.monotonic() + delai
        while True:
            if self.processus.poll() is not None:
                raise ErreurMesure(f"llama-server s'est arrêté (code {self.processus.returncode}) avant d'être prêt : "
                                   f"{self.fin_du_journal()}")
            if self._est_pret():
                return
            if time.monotonic() >= fin:
                raise ErreurMesure(f"llama-server n'était pas prêt après {delai:g} s : il a été arrêté.")
            time.sleep(SONDAGE)


def _lire_timings(reponse: Any, options: OptionsServeur) -> dict[str, float]:
    timings = reponse.get("timings") if isinstance(reponse, dict) else None
    if not isinstance(timings, dict):
        raise ErreurMesure("La réponse de llama-server n'a pas de champ « timings » : version du moteur non gérée ?")
    for cle in ("prompt_n", "prompt_per_second", "predicted_n", "predicted_per_second"):
        valeur = timings.get(cle)
        if isinstance(valeur, bool) or not isinstance(valeur, (int, float)) or not math.isfinite(valeur):
            raise ErreurMesure(f"Le champ timings.{cle} de llama-server manque ou n'est pas un nombre.")
    if timings["predicted_n"] != options.generation:
        raise ErreurMesure(f"llama-server a généré {timings['predicted_n']} jetons au lieu des {options.generation} demandés : "
                           "la vitesse ne serait pas comparable.")
    if timings["prompt_per_second"] <= 0 or timings["predicted_per_second"] <= 0:
        raise ErreurMesure("llama-server a rendu une vitesse nulle ou négative.")
    return timings


def _stats(echantillons: list[float]) -> tuple[float, float]:
    return statistics.fmean(echantillons), statistics.stdev(echantillons) if len(echantillons) > 1 else 0.0


def _mesurer_une_valeur(moteur: Moteur, modele: Path, fichier: gguf.Fichier, cle: str, natif: int | None, totaux: int,
                        valeur: int, options: OptionsServeur, progression: Callable[[str], None]) -> Resultat:
    occupation = mesures.occupation_gpu()  # relevé avant de charger le modèle : après, il dirait notre propre charge
    port, cle_api = _port_libre(), secrets.token_urlsafe(24)
    commande = construire_commande(moteur, modele, port, cle_api, options, cle, valeur)
    progression(f"{valeur} experts actifs : démarrage de llama-server...")
    with Serveur(commande, port, cle_api) as serveur:
        try:
            serveur.attendre_pret(options.delai_demarrage)
        except ErreurMesure as erreur:
            raise mesures.avec_indice_types_inconnus(erreur, fichier) from None
        props = serveur.appeler("/props")
        base = serveur.appeler("/tokenize", {"content": TEXTE_BASE}).get("tokens")
        if not base or not all(isinstance(j, int) for j in base):
            raise ErreurMesure("llama-server n'a pas rendu de jetons pour /tokenize.")
        prompt = (base * math.ceil(options.prompt / len(base)))[:options.prompt]
        corps = {"prompt": prompt, "n_predict": options.generation, "temperature": 0, "seed": SEED, "cache_prompt": False,
                 "ignore_eos": True, "stream": False}
        for i in range(options.echauffement):
            progression(f"{valeur} experts actifs : échauffement {i + 1}/{options.echauffement}")
            serveur.appeler("/completion", corps, options.delai_requete)
        lecture, generation, jetons_lus = [], [], set()
        for i in range(options.repetitions):
            progression(f"{valeur} experts actifs : mesure {i + 1}/{options.repetitions}")
            timings = _lire_timings(serveur.appeler("/completion", corps, options.delai_requete), options)
            lecture.append(float(timings["prompt_per_second"]))
            generation.append(float(timings["predicted_per_second"]))
            jetons_lus.add(int(timings["prompt_n"]))
    if len(jetons_lus) != 1:
        raise ErreurMesure(f"Le nombre de jetons de prompt lus a changé d'une requête à l'autre : {sorted(jetons_lus)}.")

    info = props.get("build_info") if isinstance(props, dict) else None
    trouve = re.match(r"b(\d+)-([0-9a-f]+)$", info) if isinstance(info, str) else None
    configuration = Configuration(
        outil="serveur",
        modele=mesures.decrire_modele(fichier),
        moteur={"build": int(trouve.group(1)) if trouve else None, "commit": trouve.group(2) if trouve else None,
                "build_info": info, "processeur": platform.processor() or None, "carte_graphique": mesures.nom_gpu(),
                "systeme": f"{platform.system()} {platform.release()}"},
        reglages={"experts_actifs": valeur, "experts_actifs_fichier": natif, "experts_totaux": totaux, "cle_experts": cle,
                  "n_gpu_layers": options.couches_gpu if options.couches_gpu is not None else "defaut",
                  "n_cpu_moe": options.experts_cpu or 0, "contexte": options.contexte, "prompt_jetons": options.prompt,
                  "generation_jetons": options.generation, "repetitions": options.repetitions,
                  "echauffement": options.echauffement, "flash_attn": options.flash_attn or "defaut",
                  "threads": options.threads or "defaut", "temperature": 0, "graine": SEED, "cache_prompt": False,
                  "ignore_eos": True, "emplacements": 1})
    mesures_obtenues = []
    for nom, echantillons in ((f"pp{next(iter(jetons_lus))}", lecture), (f"tg{options.generation}", generation)):
        moyenne, ecart = _stats(echantillons)
        mesures_obtenues.append(Mesure(nom, moyenne, ecart, tuple(echantillons)))

    avertissements = mesures.avertissement_gpu(occupation, options.couches_gpu != 0)
    if options.repetitions < 3:
        avertissements.append(f"{options.repetitions} répétition(s) seulement : l'écart-type est peu significatif.")
    if natif is not None and valeur > natif:
        avertissements.append(f"{valeur} experts actifs dépassent les {natif} du fichier : le routeur n'a pas été entraîné "
                              "pour cela, la qualité n'est pas garantie.")
    elif natif is not None and valeur < natif:
        avertissements.append(f"{valeur} experts actifs au lieu de {natif} : la vitesse monte, la qualité baisse "
                              "probablement (elle n'est pas mesurée ici).")
    return Resultat(configuration, tuple(mesures_obtenues), datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    occupation, tuple(avertissements))


def mesurer_experts(moteur: Moteur, modele, valeurs, options: OptionsServeur | None = None,
                    progression: Callable[[str], None] | None = None) -> list[Resultat]:
    """Une mesure par nombre d'experts actifs. Tout est vérifié avant de lancer le moindre serveur."""
    options = options or OptionsServeur()
    options.verifier()
    modele = Path(modele)
    if not modele.is_file():
        raise ErreurMesure(f"Le modèle {modele} n'existe pas.")
    try:
        fichier = gguf.lire(modele)
    except gguf.ErreurGGUF as erreur:
        raise ErreurMesure(f"Ce modèle n'est pas un GGUF lisible : {erreur}") from erreur
    cle, natif, totaux = cles_experts(fichier)
    liste = verifier_valeurs(valeurs, totaux)
    if moteur.commande_serveur is None:
        raise ErreurMesure(f"Ce moteur n'a pas de llama-server (dossier {moteur.dossier}).")
    rapporter = progression or (lambda _message: None)
    return [_mesurer_une_valeur(moteur, modele, fichier, cle, natif, totaux, valeur, options, rapporter)
            for valeur in liste]
