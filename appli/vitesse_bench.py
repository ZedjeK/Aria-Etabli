"""Le pilote de vitesse avec llama-bench : lecture du prompt (pp) et génération (tg), en jetons par seconde.

llama-bench répète chaque test et rend une moyenne, un écart-type et tous les échantillons. Il ne mesure ni la tokenisation
ni l'échantillonnage. Il ne sait pas régler le nombre d'experts actifs (pas d'--override-kv) : pour cela, voir le pilote par
llama-server. Les chiffres des deux outils ne se comparent pas entre eux, d'où le champ « outil » de la configuration.

Le moteur est lancé avec une liste d'arguments, jamais par un shell, avec un délai maximal. Rien n'est écrit sur le disque.
"""

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from appli import gguf, mesures
from appli.mesures import Configuration, ErreurMesure, Mesure, Resultat
from appli.moteur import Moteur

FLASH_ATTN = ("on", "off", "auto")

# Champs du JSON de llama-bench qui changent d'un test à l'autre ou qui ne sont pas des réglages
_PAR_TEST = {"n_prompt", "n_gen", "test_time", "avg_ns", "stddev_ns", "avg_ts", "stddev_ts", "samples_ns", "samples_ts"}
_MODELE = {"model_filename", "model_type", "model_size", "model_n_params"}  # le modèle est décrit par l'inspecteur
_MOTEUR = {"build_commit": "commit", "build_number": "build", "backends": "backends", "cpu_info": "processeur",
           "gpu_info": "carte_graphique"}


@dataclass(frozen=True)
class OptionsBench:
    couches_gpu: int | None = None  # -ngl ; None : le défaut du moteur (toutes les couches). 0 : tout sur le processeur
    experts_cpu: int | None = None  # -ncmoe : experts des N premières couches gardés sur le processeur
    profondeur: int = 0  # -d : jetons déjà dans le contexte avant le test
    prompt: int = 512  # -p : jetons de prompt à lire (0 : pas de test de lecture)
    generation: int = 128  # -n : jetons à générer (0 : pas de test de génération)
    repetitions: int = 5  # -r
    flash_attn: str | None = None  # -fa : « on », « off » ou « auto »
    threads: int | None = None  # -t
    delai_max: int = 3600  # secondes avant d'arrêter le moteur

    def verifier(self) -> None:
        def borne(nom: str, valeur: Any, bas: int, haut: int) -> None:
            if valeur is None:
                return
            if isinstance(valeur, bool) or not isinstance(valeur, int) or not bas <= valeur <= haut:
                raise ErreurMesure(f"{nom} doit être un entier entre {bas} et {haut} (reçu : {valeur!r}).")

        borne("Les couches sur la carte", self.couches_gpu, -1, 10_000)
        borne("Les experts sur le processeur", self.experts_cpu, 0, 10_000)
        borne("La profondeur de contexte", self.profondeur, 0, 10_000_000)
        borne("Le nombre de jetons de prompt", self.prompt, 0, 10_000_000)
        borne("Le nombre de jetons à générer", self.generation, 0, 10_000_000)
        borne("Le nombre de répétitions", self.repetitions, 1, 1000)
        borne("Le nombre de threads", self.threads, 1, 1024)
        borne("Le délai maximal", self.delai_max, 10, 86_400)
        if self.prompt == 0 and self.generation == 0:
            raise ErreurMesure("Rien à mesurer : le prompt et la génération sont tous deux à 0.")
        if self.flash_attn is not None and self.flash_attn not in FLASH_ATTN:
            raise ErreurMesure(f"La flash attention doit être « on », « off » ou « auto » (reçu : {self.flash_attn!r}).")


def construire_commande(moteur: Moteur, modele: Path, options: OptionsBench) -> list[str]:
    if moteur.commande_bench is None:
        raise ErreurMesure(f"Ce moteur n'a pas de llama-bench (dossier {moteur.dossier}).")
    commande = [*moteur.commande_bench, "-m", str(modele), "-o", "json", "-r", str(options.repetitions),
                "-p", str(options.prompt), "-n", str(options.generation), "-d", str(options.profondeur)]
    for drapeau, valeur in (("-ngl", options.couches_gpu), ("-ncmoe", options.experts_cpu),
                            ("-fa", options.flash_attn), ("-t", options.threads)):
        if valeur is not None:
            commande += [drapeau, str(valeur)]
    return commande


def _lancer(commande: list[str], delai: int) -> str:
    try:
        processus = subprocess.run(commande, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                   timeout=delai, check=False, stdin=subprocess.DEVNULL,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired:
        raise ErreurMesure(f"llama-bench n'a pas fini en {delai} s : il a été arrêté.") from None
    except OSError as erreur:
        raise ErreurMesure(f"Impossible de lancer llama-bench : {erreur.strerror or erreur}") from erreur
    if processus.returncode != 0:
        fin = " | ".join(ligne.strip() for ligne in processus.stderr.strip().splitlines()[-3:]) or "aucun message"
        raise ErreurMesure(f"llama-bench s'est arrêté avec le code {processus.returncode} : {fin}")
    return processus.stdout


def _lire_json(sortie: str) -> list[dict[str, Any]]:
    """Le tableau JSON de llama-bench, même si le moteur écrit d'autres lignes avant ou après."""
    decodeur = json.JSONDecoder()
    for trouve in re.finditer(r"\[\s*\{", sortie):
        try:
            donnees, _ = decodeur.raw_decode(sortie, trouve.start())
        except ValueError:
            continue
        if isinstance(donnees, list) and donnees and all(isinstance(e, dict) for e in donnees):
            return donnees
    raise ErreurMesure("llama-bench n'a pas rendu de résultat lisible (pas de tableau JSON dans sa sortie).")


def _nom_test(enregistrement: dict[str, Any]) -> str:
    prompt, generation = enregistrement.get("n_prompt", 0), enregistrement.get("n_gen", 0)
    profondeur = enregistrement.get("n_depth", 0)
    if prompt and not generation:
        nom = f"pp{prompt}"
    elif generation and not prompt:
        nom = f"tg{generation}"
    else:
        nom = f"pp{prompt}+tg{generation}"
    return nom + (f"@d{profondeur}" if profondeur else "")


def _mesure(enregistrement: dict[str, Any]) -> Mesure:
    for cle in ("avg_ts", "stddev_ts", "samples_ts"):
        if cle not in enregistrement:
            raise ErreurMesure(f"La sortie de llama-bench n'a pas le champ {cle} : version du moteur non gérée ?")
    return Mesure(_nom_test(enregistrement), float(enregistrement["avg_ts"]), float(enregistrement["stddev_ts"]),
                  tuple(float(v) for v in enregistrement["samples_ts"]))


def mesurer(moteur: Moteur, modele, options: OptionsBench | None = None) -> Resultat:
    """Mesure la vitesse d'un modèle GGUF avec llama-bench. Lève ErreurMesure si la mesure n'a pas pu être faite."""
    options = options or OptionsBench()
    options.verifier()
    modele = Path(modele)
    if not modele.is_file():
        raise ErreurMesure(f"Le modèle {modele} n'existe pas.")
    try:
        fichier = gguf.lire(modele)
    except gguf.ErreurGGUF as erreur:
        raise ErreurMesure(f"Ce modèle n'est pas un GGUF lisible : {erreur}") from erreur

    commande = construire_commande(moteur, modele, options)
    occupation = mesures.occupation_gpu()  # relevé juste avant, pas pendant : il dirait la charge de la mesure elle-même
    try:
        enregistrements = _lire_json(_lancer(commande, options.delai_max))
    except ErreurMesure as erreur:
        inconnus = sorted({t.type_nom for t in fichier.tenseurs if t.type not in gguf.TYPES_TENSEURS})
        if inconnus:
            raise ErreurMesure(f"{str(erreur).rstrip('.')}. Indice : ce modèle contient des tenseurs de type {', '.join(inconnus)}, que le "
                               "lecteur d'Etabli ne connaît pas ; le moteur essayé est peut-être trop ancien, ou n'est pas "
                               "le fork de llama.cpp qui gère ce format.") from None
        raise

    premier = enregistrements[0]
    configuration = Configuration(
        outil="bench",
        modele=mesures.decrire_modele(fichier),
        moteur={nom: premier.get(cle) for cle, nom in _MOTEUR.items()},
        reglages={cle: valeur for cle, valeur in premier.items() if cle not in _PAR_TEST | _MODELE | _MOTEUR.keys()},
    )
    avertissements = mesures.avertissement_gpu(occupation, options.couches_gpu != 0)
    if options.repetitions < 3:
        avertissements.append(f"{options.repetitions} répétition(s) seulement : l'écart-type est peu significatif.")
    return Resultat(configuration, tuple(_mesure(e) for e in enregistrements),
                    str(premier.get("test_time") or datetime.now(timezone.utc).isoformat(timespec="seconds")),
                    occupation, tuple(avertissements))
