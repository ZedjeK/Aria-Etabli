"""Les jeux de test : ce qu'Etabli sait télécharger, où il le range, et comment il vérifie ce qu'il a reçu.

    python -m appli jeux                          liste les jeux connus et ce qui est déjà là (aucune connexion)
    python -m appli jeux telecharger gsm8k        montre ce qui serait téléchargé, sans rien faire
    python -m appli jeux telecharger gsm8k --accord   le télécharge, une seule fois

Rien ne se télécharge sans --accord. Le téléchargement n'utilise qu'un hôte, api.github.com (celui des mises à jour), en HTTPS, et
ne suit aucune redirection vers un autre hôte. Chaque fichier est identifié ici par sa taille, son SHA-256 et le commit exact du dépôt
d'origine : un fichier qui ne correspond pas n'est jamais gardé, et un fichier déjà rangé n'est utilisé que s'il correspond encore.
Les fichiers vont dans `jeux\\<jeu>\\` du dossier de données, créé seulement au premier téléchargement.
"""

import hashlib
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from appli import donnees, identite

DOSSIER_JEUX = "jeux"
HOTES_AUTORISES = ("api.github.com",)
DELAI_RESEAU = 60  # secondes


class ErreurJeu(Exception):
    """Un jeu de test n'est pas disponible ou n'a pas pu être téléchargé : le message dit pourquoi, en français."""


@dataclass(frozen=True)
class FichierJeu:
    nom: str
    taille: int  # octets, exacts
    sha256: str  # minuscules
    url: str


@dataclass(frozen=True)
class Jeu:
    nom: str
    titre: str
    licence: str
    origine: str
    fichiers: tuple[FichierJeu, ...]


# GSM8K : dépôt d'origine d'OpenAI (MIT, archivé), figé au commit du 19 novembre 2021 qui est son dernier.
_COMMIT_GSM8K = "3101c7d5072418e28b9008a6636bde82a006892c"
_URL_GSM8K = ("https://api.github.com/repos/openai/grade-school-math/contents/grade_school_math/data/{}?ref="
              + _COMMIT_GSM8K)
GSM8K = Jeu(
    nom="gsm8k",
    titre="GSM8K : 8 500 problèmes de mathématiques de primaire (1 319 en test)",
    licence="MIT, © 2021 OpenAI",
    origine="https://github.com/openai/grade-school-math",
    fichiers=(
        FichierJeu("test.jsonl", 749_738, "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
                   _URL_GSM8K.format("test.jsonl")),
        FichierJeu("train.jsonl", 4_166_206, "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465",
                   _URL_GSM8K.format("train.jsonl")),
    ),
)
JEUX = {GSM8K.nom: GSM8K}


def dossier_du_jeu(jeu: Jeu, racine: Path | None = None) -> Path:
    return (Path(racine) if racine is not None else identite.dossier_donnees()) / DOSSIER_JEUX / jeu.nom


def chemin_du_fichier(jeu: Jeu, nom: str, racine: Path | None = None) -> Path:
    return dossier_du_jeu(jeu, racine) / nom


def _empreinte(chemin: Path) -> str:
    resume = hashlib.sha256()
    with open(chemin, "rb") as fichier:
        for bloc in iter(lambda: fichier.read(1 << 20), b""):
            resume.update(bloc)
    return resume.hexdigest()


def fichier_conforme(chemin: Path, fichier: FichierJeu) -> bool:
    """Vrai si le fichier existe, a la bonne taille et le bon SHA-256."""
    try:
        return chemin.is_file() and chemin.stat().st_size == fichier.taille and _empreinte(chemin) == fichier.sha256
    except OSError:
        return False


def etat(jeu: Jeu, racine: Path | None = None) -> dict[str, str]:
    """« présent » (conforme), « corrompu » (là mais différent de ce qui est attendu) ou « absent », par fichier."""
    resultat = {}
    for fichier in jeu.fichiers:
        chemin = chemin_du_fichier(jeu, fichier.nom, racine)
        resultat[fichier.nom] = ("présent" if fichier_conforme(chemin, fichier)
                                 else "corrompu" if chemin.exists() else "absent")
    return resultat


def a_telecharger(jeu: Jeu, racine: Path | None = None) -> list[FichierJeu]:
    statuts = etat(jeu, racine)
    return [fichier for fichier in jeu.fichiers if statuts[fichier.nom] != "présent"]


def _valider_url(url: str) -> None:
    analyse = urllib.parse.urlsplit(url)
    if analyse.scheme != "https" or analyse.hostname not in HOTES_AUTORISES:
        raise ErreurJeu(f"Adresse refusée : seuls les téléchargements en HTTPS depuis {', '.join(HOTES_AUTORISES)} sont "
                        f"permis ({url}).")


class _RedirectionRestreinte(urllib.request.HTTPRedirectHandler):
    """Une redirection vers un autre hôte (ou en clair) est refusée : la liste des connexions ne doit pas s'élargir."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _valider_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _lire_url(url: str, taille_max: int) -> bytes:
    """Télécharge `url` (au plus `taille_max` octets). Seule fonction du module qui touche le réseau."""
    _valider_url(url)
    requete = urllib.request.Request(url, headers={"Accept": "application/vnd.github.raw+json",
                                                   "User-Agent": f"{identite.EXE}/{identite.version()}"})
    ouvreur = urllib.request.build_opener(_RedirectionRestreinte())
    try:
        with ouvreur.open(requete, timeout=DELAI_RESEAU) as reponse:
            donnees_recues = reponse.read(taille_max + 1)
    except urllib.error.HTTPError as erreur:
        raise ErreurJeu(f"GitHub a répondu {erreur.code} ({erreur.reason}) : réessaie plus tard"
                        + (" (limite de requêtes atteinte)" if erreur.code in (403, 429) else "") + ".") from None
    except (urllib.error.URLError, OSError, TimeoutError) as erreur:
        raise ErreurJeu(f"Impossible de joindre api.github.com ({erreur}).") from None
    if len(donnees_recues) > taille_max:
        raise ErreurJeu(f"Le fichier reçu est plus gros que prévu (plus de {taille_max} octets) : il est refusé.")
    return donnees_recues


def telecharger(jeu: Jeu, racine: Path | None = None, *, accord: bool = False,
                lire: Callable[[str, int], bytes] | None = None,
                progression: Callable[[str], None] | None = None) -> list[Path]:
    """Télécharge ce qui manque ou ne correspond plus, avec accord seulement. Renvoie les fichiers écrits."""
    if not accord:
        raise ErreurJeu("Téléchargement non autorisé : il faut ton accord explicite (--accord).")
    rapporter = progression or (lambda _message: None)
    lire = lire or _lire_url  # résolu ici et non dans la signature : un test peut ainsi remplacer le réseau
    ecrits: list[Path] = []
    for fichier in a_telecharger(jeu, racine):
        chemin = chemin_du_fichier(jeu, fichier.nom, racine)
        rapporter(f"Téléchargement de {fichier.nom} ({fichier.taille} octets)...")
        contenu = lire(fichier.url, fichier.taille)
        if len(contenu) != fichier.taille:
            raise ErreurJeu(f"{fichier.nom} : {len(contenu)} octets reçus au lieu de {fichier.taille} : refusé.")
        if hashlib.sha256(contenu).hexdigest() != fichier.sha256:
            raise ErreurJeu(f"{fichier.nom} : l'empreinte SHA-256 ne correspond pas à celle attendue : refusé, rien n'est "
                            "gardé.")
        chemin.parent.mkdir(parents=True, exist_ok=True)
        partiel = chemin.with_name(chemin.name + ".partiel")
        try:
            partiel.write_bytes(contenu)
            os.replace(partiel, chemin)  # le fichier n'apparaît qu'entier et vérifié
        except OSError as erreur:
            raise ErreurJeu(f"Impossible d'écrire {chemin} : {erreur.strerror or erreur}") from erreur
        finally:
            partiel.unlink(missing_ok=True)
        ecrits.append(chemin)
    if ecrits:
        donnees.noter("jeux", f"Jeu {jeu.nom} téléchargé depuis api.github.com : {', '.join(c.name for c in ecrits)}.",
                      racine)
    return ecrits


# ----- La commande -----

AIDE = """Usage : python -m appli jeux
        python -m appli jeux telecharger JEU [--accord]

Les jeux de test connus d'Etabli, et leur téléchargement.

  (sans argument)         liste les jeux et dit ce qui est déjà téléchargé ; aucune connexion
  telecharger JEU         montre ce qui serait téléchargé (source, taille, empreinte, destination) sans rien faire
  telecharger JEU --accord   télécharge ce qui manque, depuis api.github.com seulement, avec vérification du SHA-256
  --aide                  ce message"""


def _octets(n: int) -> str:
    for unite, taille in (("Mio", 1 << 20), ("Kio", 1 << 10)):
        if n >= taille:
            return f"{n / taille:.2f} {unite}".replace(".", ",")
    return f"{n} o"


def _lister() -> int:
    for jeu in JEUX.values():
        print(f"{jeu.nom} : {jeu.titre}")
        print(f"  licence : {jeu.licence} ; origine : {jeu.origine}")
        statuts = etat(jeu)
        for fichier in jeu.fichiers:
            print(f"  {fichier.nom:<12} {_octets(fichier.taille):>10}  {statuts[fichier.nom]}")
        print(f"  dossier : {dossier_du_jeu(jeu)}")
    return 0


def _telecharger(arguments: list[str]) -> int:
    noms = [a for a in arguments if not a.startswith("--")]
    options = {a for a in arguments if a.startswith("--")}
    if len(noms) != 1 or options - {"--accord"}:
        raise _Usage("Donne un seul jeu, avec au plus l'option --accord.")
    if noms[0] not in JEUX:
        raise _Usage(f"Jeu inconnu : {noms[0]} (connus : {', '.join(JEUX)}).")
    jeu = JEUX[noms[0]]
    manquants = a_telecharger(jeu)
    if not manquants:
        print(f"{jeu.nom} est déjà là, complet et conforme : rien à télécharger.")
        return 0
    if "--accord" not in options:
        print(f"Voici ce qui serait téléchargé pour {jeu.nom} ({jeu.licence}) :")
        for fichier in manquants:
            print(f"  {fichier.nom} : {_octets(fichier.taille)}, SHA-256 {fichier.sha256}")
        print(f"  source : api.github.com (dépôt {jeu.origine}, commit figé), en HTTPS")
        print(f"  destination : {dossier_du_jeu(jeu)}")
        print("Rien n'a été fait. Relance avec --accord pour télécharger.")
        return 0
    try:
        ecrits = telecharger(jeu, accord=True, progression=lambda message: print(message, file=sys.stderr, flush=True))
    except ErreurJeu as erreur:
        print(f"Téléchargement impossible : {erreur}", file=sys.stderr)
        return 1
    print(f"{jeu.nom} : {len(ecrits)} fichier(s) téléchargé(s) et vérifié(s) dans {dossier_du_jeu(jeu)}.")
    return 0


class _Usage(Exception):
    pass


def main(arguments: list[str]) -> int:
    """La commande « jeux ». 0 : fait ; 1 : téléchargement impossible ; 2 : mauvais usage."""
    if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    try:
        if arguments in ([], ["--aide"]):
            if arguments:
                print(AIDE)
                return 0
            return _lister()
        if arguments[0] == "telecharger":
            return _telecharger(arguments[1:])
        raise _Usage(f"Sous-commande inconnue : {arguments[0]}.")
    except _Usage as erreur:
        print(erreur, file=sys.stderr)
        print(AIDE, file=sys.stderr)
        return 2
