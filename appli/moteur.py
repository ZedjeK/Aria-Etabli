"""Un moteur llama.cpp : le dossier qui contient llama-bench et llama-server.

Etabli ne lit pas les dossiers des autres applis : le dossier du moteur est donné explicitement. Un moteur est reconnu à ses
exécutables, rien de plus ; sa version, elle, est lue dans ce que le moteur dit de lui-même au moment d'une mesure (le JSON de
llama-bench donne le build et le commit), car tous les moteurs ne répondent pas à --version (le fork Prism, par exemple).
"""

import sys
from dataclasses import dataclass
from pathlib import Path

SUFFIXE = ".exe" if sys.platform == "win32" else ""


class ErreurMoteur(Exception):
    """Le dossier donné n'est pas un moteur llama.cpp utilisable : le message dit pourquoi, en français."""


@dataclass(frozen=True)
class Moteur:
    dossier: Path
    # Début de la ligne de commande, sans les arguments : (chemin de l'exécutable,). Les tests y mettent un faux moteur.
    commande_bench: tuple[str, ...] | None
    commande_serveur: tuple[str, ...] | None


def reconnaitre(dossier) -> Moteur:
    dossier = Path(dossier)
    if not dossier.is_dir():
        raise ErreurMoteur(f"{dossier} n'est pas un dossier : donne celui qui contient llama-bench et llama-server.")
    bench = dossier / f"llama-bench{SUFFIXE}"
    serveur = dossier / f"llama-server{SUFFIXE}"
    if not bench.is_file() and not serveur.is_file():
        raise ErreurMoteur(f"Le dossier {dossier} ne contient ni {bench.name} ni {serveur.name} : "
                           "ce n'est pas un moteur llama.cpp.")
    return Moteur(dossier,
                  (str(bench),) if bench.is_file() else None,
                  (str(serveur),) if serveur.is_file() else None)
