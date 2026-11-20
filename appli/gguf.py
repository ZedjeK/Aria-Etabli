"""Lecteur GGUF en lecture seule, sans dépendance.

Un fichier GGUF commence par un en-tête (la signature « GGUF », la version, le nombre de tenseurs, le nombre de clés), puis
les clés de métadonnées, puis la description de chaque tenseur (nom, dimensions, type, décalage). Les poids viennent après,
alignés. Ce module lit seulement ce début : il ne charge jamais les poids, et un fichier de 20 Go s'ouvre en une fraction
de seconde (le fichier est projeté en mémoire en lecture seule : seules les pages de l'en-tête sont touchées).

Le fichier est traité comme non fiable : signature, version, comptes et longueurs sont vérifiés avant d'être suivis, et
toute anomalie lève ErreurGGUF avec un message en clair. Rien n'est écrit, nulle part.

Un type de tenseur absent de la table (une quantification propre à un fork de llama.cpp, par exemple) ne fait pas échouer
la lecture : son nom est « inconnu(N) » et sa taille se déduit de l'écart avec le tenseur suivant.
"""

import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAGIE = b"GGUF"
VERSIONS = (2, 3)
ALIGNEMENT_DEFAUT = 32
MAX_DIMENSIONS = 8
LONGUEUR_MAX_NOM = 65_535
LONGUEUR_MAX_TEXTE = 16 * 1024 * 1024
APERCU = 8  # éléments gardés d'un tableau : le reste est compté, pas lu
MAX_ENTREES = 1_000_000  # clés ou tenseurs : un modèle réel en compte quelques dizaines de milliers au plus

# Taille minimale, en octets, d'une clé et d'une description de tenseur : borne les comptes annoncés par un fichier
_OCTETS_MIN_CLE = 12
_OCTETS_MIN_TENSEUR = 24

# Types de valeurs des métadonnées : numéro -> (code struct, octets)
_SCALAIRES = {0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4), 5: ("i", 4), 6: ("f", 4), 7: ("?", 1),
              10: ("Q", 8), 11: ("q", 8), 12: ("d", 8)}
_CHAINE = 8
_TABLEAU = 9
NOMS_TYPES_VALEUR = {0: "UINT8", 1: "INT8", 2: "UINT16", 3: "INT16", 4: "UINT32", 5: "INT32", 6: "FLOAT32", 7: "BOOL",
                     8: "STRING", 9: "ARRAY", 10: "UINT64", 11: "INT64", 12: "FLOAT64"}

# Types de tenseurs : numéro -> (nom, éléments par bloc, octets par bloc). Table du paquet officiel gguf 0.19.0 ; un test la
# recoupe avec lui.
TYPES_TENSEURS = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 40),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
    39: ("MXFP4", 32, 17),
    40: ("NVFP4", 64, 36),
    41: ("Q1_0", 128, 18),
}


class ErreurGGUF(Exception):
    """Le fichier n'est pas lisible comme un GGUF : le message dit pourquoi, en français."""


@dataclass(frozen=True)
class Tableau:
    """Un tableau de métadonnées : son type d'élément, sa taille, et les premiers éléments seulement."""
    type_element: int
    nombre: int
    apercu: tuple

    @property
    def nom_type(self) -> str:
        return NOMS_TYPES_VALEUR.get(self.type_element, f"inconnu({self.type_element})")


@dataclass(frozen=True)
class Tenseur:
    nom: str
    dimensions: tuple[int, ...]
    type: int
    decalage: int  # relatif au début des données
    elements: int
    octets: int
    octets_exacts: bool  # faux quand le type est inconnu : la taille vient de l'écart avec le tenseur suivant

    @property
    def type_nom(self) -> str:
        return nom_du_type(self.type)


@dataclass(frozen=True)
class Fichier:
    chemin: Path
    taille: int
    version: int
    alignement: int
    debut_donnees: int
    metadonnees: dict[str, Any]
    tenseurs: tuple[Tenseur, ...]


def nom_du_type(numero: int) -> str:
    connu = TYPES_TENSEURS.get(numero)
    return connu[0] if connu else f"inconnu({numero})"


class _Curseur:
    """Lecture séquentielle dans le tampon, avec une vérification avant chaque pas."""

    def __init__(self, tampon, taille: int):
        self.tampon = tampon
        self.taille = taille
        self.position = 0

    @property
    def restant(self) -> int:
        return self.taille - self.position

    def _verifier(self, octets: int, quoi: str) -> None:
        if octets < 0 or octets > self.restant:
            raise ErreurGGUF(f"Le fichier s'arrête au milieu de {quoi} : il est tronqué ou corrompu.")

    def nombre(self, code: str, quoi: str = "l'en-tête"):
        octets = struct.calcsize("<" + code)
        self._verifier(octets, quoi)
        valeur = struct.unpack_from("<" + code, self.tampon, self.position)[0]
        self.position += octets
        return valeur

    def texte(self, maximum: int, quoi: str, garder: bool = True) -> str:
        longueur = self.nombre("Q", quoi)
        if longueur > maximum:
            raise ErreurGGUF(f"Un texte de {quoi} annonce {longueur} octets, au-delà de la limite de {maximum} : "
                             "fichier corrompu ?")
        self._verifier(longueur, quoi)
        debut = self.position
        self.position += longueur
        if not garder:
            return ""
        return bytes(self.tampon[debut:debut + longueur]).decode("utf-8", errors="replace")

    def verifier_compte(self, compte: int, octets_minimum: int, quoi: str, maximum: int | None = None) -> None:
        if compte * octets_minimum > self.restant or (maximum is not None and compte > maximum):
            raise ErreurGGUF(f"Le fichier annonce {compte} {quoi}, plus qu'il ne peut en contenir : corrompu ?")


def _lire_tableau(curseur: _Curseur, cle: str) -> Tableau:
    type_element = curseur.nombre("I", f"la clé {cle}")
    nombre = curseur.nombre("Q", f"la clé {cle}")
    if type_element == _TABLEAU:
        raise ErreurGGUF(f"La clé {cle} est un tableau de tableaux : ce cas n'est pas géré.")
    if type_element in _SCALAIRES:
        code, octets = _SCALAIRES[type_element]
        curseur.verifier_compte(nombre, octets, f"éléments pour la clé {cle}")
        apercu = []
        for i in range(min(nombre, APERCU)):
            apercu.append(struct.unpack_from("<" + code, curseur.tampon, curseur.position + i * octets)[0])
        curseur.position += nombre * octets
        return Tableau(type_element, nombre, tuple(apercu))
    if type_element == _CHAINE:
        curseur.verifier_compte(nombre, 8, f"éléments pour la clé {cle}")
        apercu = []
        for i in range(nombre):
            garder = i < APERCU
            texte = curseur.texte(LONGUEUR_MAX_TEXTE, f"la clé {cle}", garder=garder)
            if garder:
                apercu.append(texte)
        return Tableau(type_element, nombre, tuple(apercu))
    raise ErreurGGUF(f"La clé {cle} est un tableau d'un type inconnu ({type_element}).")


def _lire_valeur(curseur: _Curseur, type_valeur: int, cle: str):
    if type_valeur in _SCALAIRES:
        return curseur.nombre(_SCALAIRES[type_valeur][0], f"la clé {cle}")
    if type_valeur == _CHAINE:
        return curseur.texte(LONGUEUR_MAX_TEXTE, f"la clé {cle}")
    if type_valeur == _TABLEAU:
        return _lire_tableau(curseur, cle)
    raise ErreurGGUF(f"La clé {cle} a un type de valeur inconnu ({type_valeur}).")


def _produit(dimensions: tuple[int, ...]) -> int:
    resultat = 1
    for d in dimensions:
        resultat *= d
    return resultat


def _analyser(tampon, taille: int, chemin: Path) -> Fichier:
    curseur = _Curseur(tampon, taille)
    if bytes(tampon[:4]) != MAGIE:
        raise ErreurGGUF("Ce n'est pas un fichier GGUF : la signature « GGUF » manque au début.")
    curseur.position = 4
    version = curseur.nombre("I")
    if version not in VERSIONS:
        if struct.unpack(">I", struct.pack("<I", version))[0] in VERSIONS:
            raise ErreurGGUF("Ce fichier GGUF est écrit en gros-boutiste (big-endian) : ce cas n'est pas géré.")
        raise ErreurGGUF(f"Version GGUF {version} non gérée (versions lues : 2 et 3).")
    nombre_tenseurs = curseur.nombre("Q")
    nombre_cles = curseur.nombre("Q")
    curseur.verifier_compte(nombre_cles, _OCTETS_MIN_CLE, "clés", MAX_ENTREES)
    curseur.verifier_compte(nombre_tenseurs, _OCTETS_MIN_TENSEUR, "tenseurs", MAX_ENTREES)

    metadonnees: dict[str, Any] = {}
    for _ in range(nombre_cles):
        cle = curseur.texte(LONGUEUR_MAX_NOM, "une clé")
        if cle in metadonnees:
            raise ErreurGGUF(f"La clé {cle} apparaît deux fois.")
        type_valeur = curseur.nombre("I", f"la clé {cle}")
        metadonnees[cle] = _lire_valeur(curseur, type_valeur, cle)

    alignement = metadonnees.get("general.alignment", ALIGNEMENT_DEFAUT)
    if isinstance(alignement, bool) or not isinstance(alignement, int) or alignement <= 0 or alignement & (alignement - 1):
        raise ErreurGGUF(f"La clé general.alignment vaut {alignement!r} : une puissance de deux est attendue.")

    brutes = []
    noms = set()
    for _ in range(nombre_tenseurs):
        nom = curseur.texte(LONGUEUR_MAX_NOM, "un nom de tenseur")
        if nom in noms:
            raise ErreurGGUF(f"Le tenseur {nom} apparaît deux fois.")
        noms.add(nom)
        nb_dimensions = curseur.nombre("I", f"le tenseur {nom}")
        if nb_dimensions > MAX_DIMENSIONS:
            raise ErreurGGUF(f"Le tenseur {nom} annonce {nb_dimensions} dimensions : fichier corrompu ?")
        dimensions = tuple(curseur.nombre("Q", f"le tenseur {nom}") for _ in range(nb_dimensions))
        type_tenseur = curseur.nombre("I", f"le tenseur {nom}")
        decalage = curseur.nombre("Q", f"le tenseur {nom}")
        brutes.append((nom, dimensions, type_tenseur, decalage))

    debut_donnees = -(-curseur.position // alignement) * alignement
    if brutes and debut_donnees > taille:
        raise ErreurGGUF("Le fichier s'arrête avant ses données : il est tronqué.")

    # Écart jusqu'au tenseur suivant (ou jusqu'à la fin du fichier) : donne la taille des types inconnus
    ordre = sorted(range(len(brutes)), key=lambda i: brutes[i][3])
    ecarts = {}
    for rang, i in enumerate(ordre):
        suivant = brutes[ordre[rang + 1]][3] if rang + 1 < len(ordre) else taille - debut_donnees
        ecarts[i] = suivant - brutes[i][3]

    tenseurs = []
    for i, (nom, dimensions, type_tenseur, decalage) in enumerate(brutes):
        if debut_donnees + decalage > taille:
            raise ErreurGGUF(f"Le fichier est tronqué : le tenseur {nom} commence après la fin du fichier.")
        elements = _produit(dimensions)
        connu = TYPES_TENSEURS.get(type_tenseur)
        octets, exacts = ecarts[i], False
        if connu and elements % connu[1] == 0:
            octets, exacts = elements // connu[1] * connu[2], True
            if octets > ecarts[i]:
                raise ErreurGGUF(f"Le fichier est tronqué ou corrompu : le tenseur {nom} demande {octets} octets, "
                                 f"il n'en reste que {max(ecarts[i], 0)}.")
        elif octets < 0:
            raise ErreurGGUF(f"Les décalages des tenseurs sont incohérents autour de {nom} : fichier corrompu ?")
        tenseurs.append(Tenseur(nom, dimensions, type_tenseur, decalage, elements, octets, exacts))
    return Fichier(chemin, taille, version, alignement, debut_donnees, metadonnees, tuple(tenseurs))


def lire(chemin) -> Fichier:
    """Lit l'en-tête d'un fichier GGUF. Lève ErreurGGUF si le fichier n'est pas lisible."""
    chemin = Path(chemin)
    if chemin.is_dir():
        raise ErreurGGUF(f"{chemin} est un dossier, pas un fichier GGUF.")
    try:
        if chemin.stat().st_size < 24:
            raise ErreurGGUF("Ce fichier est trop court pour être un GGUF.")
        with open(chemin, "rb") as fichier, mmap.mmap(fichier.fileno(), 0, access=mmap.ACCESS_READ) as tampon:
            return _analyser(tampon, len(tampon), chemin)
    except OSError as erreur:
        raise ErreurGGUF(f"Impossible de lire {chemin} : {erreur.strerror or erreur}") from erreur
