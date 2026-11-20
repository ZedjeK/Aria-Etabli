"""La commande « mesurer » : vitesse d'un modèle GGUF avec un moteur llama.cpp.

    python -m appli mesurer bench MODELE.gguf --moteur DOSSIER [options]

Ne touche ni au dossier de données ni au modèle. Écrit un fichier seulement avec --sortie, et jamais par-dessus un fichier
qui existe déjà.
"""

import json
import sys
from pathlib import Path
from typing import Any, Callable

from appli import moteur as moteurs
from appli import vitesse_bench
from appli.mesures import ErreurMesure, Resultat

AIDE = """Usage : python -m appli mesurer bench MODELE.gguf --moteur DOSSIER [options]

Mesure la vitesse d'un modèle avec llama-bench : lecture du prompt (pp) et génération (tg), en jetons par seconde, avec
l'écart-type et la configuration complète (modèle, moteur, matériel, réglages). Ne modifie ni le modèle ni ton dossier de
données.

  --moteur DOSSIER      dossier qui contient llama-bench (obligatoire)
  --ngl N               couches sur la carte graphique (0 : tout sur le processeur ; défaut : celui du moteur)
  --ncmoe N             experts des N premières couches gardés sur le processeur
  --profondeur D        jetons déjà dans le contexte avant le test (défaut 0)
  --prompt P            jetons de prompt à lire (défaut 512, 0 : pas de test de lecture)
  --generation G        jetons à générer (défaut 128, 0 : pas de test de génération)
  -r, --repetitions N   répétitions de chaque test (défaut 5)
  --flash-attn on|off|auto
  --threads N
  --delai S             secondes avant d'arrêter le moteur (défaut 3600)
  --json                sortie en JSON plutôt qu'en texte
  --sortie FICHIER      écrit aussi le résultat JSON dans ce fichier (refuse un fichier qui existe déjà)
  --aide                ce message"""

_OPTIONS: dict[str, tuple[str, Callable[[str], Any]]] = {
    "--moteur": ("moteur", str), "--ngl": ("couches_gpu", int), "--ncmoe": ("experts_cpu", int),
    "--profondeur": ("profondeur", int), "--prompt": ("prompt", int), "--generation": ("generation", int),
    "--repetitions": ("repetitions", int), "-r": ("repetitions", int), "--flash-attn": ("flash_attn", str),
    "--threads": ("threads", int), "--delai": ("delai_max", int), "--sortie": ("sortie", str),
}
_DRAPEAUX = {"--json", "--aide"}


class _Usage(Exception):
    pass


def _lire_arguments(arguments: list[str]) -> tuple[list[str], dict[str, Any], set[str]]:
    positionnels: list[str] = []
    valeurs: dict[str, Any] = {}
    drapeaux: set[str] = set()
    i = 0
    while i < len(arguments):
        jeton = arguments[i]
        if jeton in _DRAPEAUX:
            drapeaux.add(jeton)
        elif jeton in _OPTIONS:
            nom, convertir = _OPTIONS[jeton]
            if i + 1 >= len(arguments):
                raise _Usage(f"{jeton} attend une valeur.")
            i += 1
            try:
                valeurs[nom] = convertir(arguments[i])
            except ValueError:
                raise _Usage(f"{jeton} attend un nombre entier (reçu : {arguments[i]!r}).") from None
        elif jeton.startswith("-"):
            raise _Usage(f"Option inconnue : {jeton}")
        else:
            positionnels.append(jeton)
        i += 1
    return positionnels, valeurs, drapeaux


def _entier(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _octets(n: int) -> str:
    for unite, taille in (("Gio", 1 << 30), ("Mio", 1 << 20), ("Kio", 1 << 10)):
        if n >= taille:
            return f"{n / taille:.2f} {unite}".replace(".", ",")
    return f"{n} o"


def _decimal(x: float) -> str:
    return f"{x:.2f}".replace(".", ",")


def texte(resultat: Resultat) -> str:
    c = resultat.configuration
    m, mo, r = c.modele, c.moteur, c.reglages
    lignes = [
        f"Modèle    : {m['fichier']} ({m['architecture'] or 'architecture non indiquée'}, "
        f"{m['quantification'] or 'quantification non indiquée'}, {_octets(m['taille_octets'])})",
        f"Moteur    : build {mo.get('build')} (commit {mo.get('commit')}), {mo.get('backends')}",
        f"Matériel  : {mo.get('processeur')} ; {mo.get('carte_graphique')}",
        "Réglages  : " + ", ".join(f"{cle} = {valeur}" for cle, valeur in r.items()
                                    if cle in ("n_gpu_layers", "n_cpu_moe", "n_depth", "flash_attn", "n_threads",
                                               "n_batch", "n_ubatch", "type_k", "type_v")),
        f"Outil     : llama-bench, {len(resultat.mesures[0].echantillons)} répétition(s) par test",
        f"Date      : {resultat.date}",
        "",
        f"  {'test':<14}{'jetons/s':>12}{'écart-type':>12}{'échantillons':>14}",
    ]
    for mesure in resultat.mesures:
        lignes.append(f"  {mesure.test:<14}{_decimal(mesure.jetons_par_seconde):>12}{_decimal(mesure.ecart_type):>12}"
                      f"{len(mesure.echantillons):>14}")
    if resultat.avertissements:
        lignes += ["", "À noter :"] + [f"  - {a}" for a in resultat.avertissements]
    return "\n".join(lignes)


def _bench(arguments: list[str]) -> int:
    positionnels, valeurs, drapeaux = _lire_arguments(arguments)
    if "--aide" in drapeaux:
        print(AIDE)
        return 0
    if len(positionnels) != 1:
        raise _Usage("Donne un seul fichier modèle.")
    if "moteur" not in valeurs:
        raise _Usage("--moteur DOSSIER est obligatoire.")
    dossier, sortie = valeurs.pop("moteur"), valeurs.pop("sortie", None)
    try:
        moteur = moteurs.reconnaitre(dossier)
        resultat = vitesse_bench.mesurer(moteur, positionnels[0], vitesse_bench.OptionsBench(**valeurs))
        donnees = json.dumps(resultat.en_dict(), ensure_ascii=True, indent=2)
        if sortie is not None:
            try:
                with open(Path(sortie), "x", encoding="utf-8") as fichier:
                    fichier.write(donnees + "\n")
            except FileExistsError:
                raise ErreurMesure(f"{sortie} existe déjà : choisis un autre nom, rien n'est écrasé.") from None
            except OSError as erreur:
                raise ErreurMesure(f"Impossible d'écrire {sortie} : {erreur.strerror or erreur}") from erreur
    except (moteurs.ErreurMoteur, ErreurMesure) as erreur:
        print(f"Mesure impossible : {erreur}", file=sys.stderr)
        return 1
    print(donnees if "--json" in drapeaux else texte(resultat))
    return 0


def main(arguments: list[str]) -> int:
    """La commande « mesurer ». 0 : fait ; 1 : mesure impossible ; 2 : mauvais usage."""
    if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if arguments == ["--aide"]:
        print(AIDE)
        return 0
    try:
        if not arguments:
            raise _Usage("Donne une sous-commande : bench.")
        if arguments[0] == "bench":
            return _bench(arguments[1:])
        raise _Usage(f"Sous-commande inconnue : {arguments[0]} (disponible : bench).")
    except _Usage as erreur:
        print(erreur, file=sys.stderr)
        print(AIDE, file=sys.stderr)
        return 2
