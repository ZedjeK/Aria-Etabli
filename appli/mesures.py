"""Ce qu'une mesure est : sa configuration complète, ses résultats, et la règle qui interdit de comparer l'incomparable.

Une vitesse en jetons par seconde ne veut rien dire seule : elle dépend de la quantification, des couches sur la carte, de
la profondeur de contexte, du moteur et de son build, du matériel. La Configuration garde tout cela avec chaque mesure, et
`verifier_comparables` refuse de mettre côte à côte deux mesures dont la configuration diffère autrement que par la variable
qu'on fait varier.
"""

import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

from appli import gguf, inspecteur

SEUIL_GPU_OCCUPE = 10  # % d'utilisation de la carte, avant la mesure, au-delà duquel un avertissement est ajouté
ABSENT = "(absent)"


class ErreurMesure(Exception):
    """La mesure n'a pas pu être faite, ou deux mesures ne sont pas comparables : le message dit pourquoi, en français."""


@dataclass(frozen=True)
class Configuration:
    """Tout ce qui doit être identique pour que deux vitesses se comparent. Trois blocs, tous en valeurs simples."""
    outil: str  # « bench » (llama-bench) ou « serveur » (llama-server) : deux protocoles, deux familles de chiffres
    modele: dict[str, Any]  # fichier, taille, architecture, quantification, types de tenseurs
    moteur: dict[str, Any]  # build, commit, backends, processeur, carte graphique
    reglages: dict[str, Any]  # couches sur la carte, profondeur de contexte, threads, cache, flash attention...

    def plat(self) -> dict[str, Any]:
        resultat: dict[str, Any] = {"outil": self.outil}
        for bloc, valeurs in (("modele", self.modele), ("moteur", self.moteur), ("reglages", self.reglages)):
            resultat.update({f"{bloc}.{cle}": valeur for cle, valeur in valeurs.items()})
        return resultat

    def en_dict(self) -> dict[str, Any]:
        return {"outil": self.outil, "modele": dict(self.modele), "moteur": dict(self.moteur),
                "reglages": dict(self.reglages)}


@dataclass(frozen=True)
class Mesure:
    test: str  # « pp512 » (lecture du prompt), « tg128 » (génération), suffixe « @d4096 » avec une profondeur de contexte
    jetons_par_seconde: float
    ecart_type: float
    echantillons: tuple[float, ...]

    def en_dict(self) -> dict[str, Any]:
        return {"test": self.test, "jetons_par_seconde": self.jetons_par_seconde, "ecart_type": self.ecart_type,
                "echantillons": list(self.echantillons)}


@dataclass(frozen=True)
class Resultat:
    configuration: Configuration
    mesures: tuple[Mesure, ...]
    date: str  # UTC, ISO 8601
    occupation_gpu_avant: int | None  # % lu une fois avant la mesure ; None si nvidia-smi est absent
    avertissements: tuple[str, ...]

    def en_dict(self) -> dict[str, Any]:
        return {"date": self.date, "configuration": self.configuration.en_dict(),
                "mesures": [m.en_dict() for m in self.mesures], "occupation_gpu_avant": self.occupation_gpu_avant,
                "avertissements": list(self.avertissements)}


def differences(a: Configuration, b: Configuration) -> dict[str, tuple[Any, Any]]:
    """Les clés (« reglages.n_gpu_layers », « modele.quantification »...) dont la valeur diffère entre a et b."""
    plat_a, plat_b = a.plat(), b.plat()
    return {cle: (plat_a.get(cle, ABSENT), plat_b.get(cle, ABSENT))
            for cle in sorted(plat_a.keys() | plat_b.keys()) if plat_a.get(cle, ABSENT) != plat_b.get(cle, ABSENT)}


def verifier_comparables(configurations: list[Configuration], variables: tuple[str, ...] = ()) -> None:
    """Lève ErreurMesure si les configurations diffèrent par autre chose que les `variables` qu'on fait varier."""
    if not configurations:
        return
    reference = configurations[0]
    for autre in configurations[1:]:
        refusees = {cle: valeurs for cle, valeurs in differences(reference, autre).items() if cle not in variables}
        if refusees:
            detail = " ; ".join(f"{cle} : {a!r} contre {b!r}" for cle, (a, b) in refusees.items())
            raise ErreurMesure(f"Ces mesures ne sont pas comparables, leurs configurations diffèrent : {detail}.")


def decrire_modele(fichier: gguf.Fichier) -> dict[str, Any]:
    """Ce que la configuration retient du modèle, d'après l'inspecteur (rien n'est deviné)."""
    resume = inspecteur.synthese(fichier)
    quantification = resume["general"]["type_de_fichier"]
    return {"fichier": fichier.chemin.name, "taille_octets": fichier.taille,
            "architecture": resume["general"]["architecture"],
            "quantification": quantification["nom"] if quantification else None,
            "types_de_tenseurs": [ligne["type"] for ligne in resume["tenseurs"]["par_type"]],
            "parametres": resume["tenseurs"]["parametres"]}


def occupation_gpu() -> int | None:
    """Utilisation de la carte graphique, en %, lue une seule fois. None si nvidia-smi est absent ou ne répond pas."""
    outil = shutil.which("nvidia-smi")
    if not outil:
        return None
    try:
        sortie = subprocess.run([outil, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=10, check=False).stdout
        return int(sortie.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def avertissement_gpu(occupation: int | None, utilise_la_carte: bool) -> list[str]:
    if occupation is not None and utilise_la_carte and occupation >= SEUIL_GPU_OCCUPE:
        return [f"La carte graphique était utilisée à {occupation} % avant la mesure (un seul relevé) : une autre appli "
                "s'en sert, les vitesses sont probablement faussées."]
    return []
