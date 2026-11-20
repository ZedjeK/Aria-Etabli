"""Écrit de petits fichiers GGUF pour les tests, octet par octet, sans passer par le lecteur qu'on teste.

Une métadonnée est un triplet (clé, type de valeur, valeur) ; un tenseur est (nom, dimensions, type, octets des poids). Pour un
tableau, la valeur est (type des éléments, liste). Une valeur de type `bytes` est écrite telle quelle (pour fabriquer des
fichiers volontairement faux).
"""

import struct
from pathlib import Path

U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
_CODES = {U8: "B", I8: "b", U16: "H", I16: "h", U32: "I", I32: "i", F32: "f", BOOL: "?", U64: "Q", I64: "q", F64: "d"}

# Types de tenseurs utilisés par les tests, avec leurs tailles écrites ici en dur (pas lues dans appli.gguf)
TENSEUR_F32 = 0
TENSEUR_Q8_0 = 8
TENSEUR_INCONNU = 142


def _texte(texte: str) -> bytes:
    octets = texte.encode("utf-8")
    return struct.pack("<Q", len(octets)) + octets


def _valeur(type_valeur: int, valeur) -> bytes:
    if isinstance(valeur, bytes):
        return valeur
    if type_valeur in _CODES:
        return struct.pack("<" + _CODES[type_valeur], valeur)
    if type_valeur == STRING:
        return _texte(valeur)
    if type_valeur == ARRAY:
        type_element, elements = valeur
        return struct.pack("<IQ", type_element, len(elements)) + b"".join(_valeur(type_element, e) for e in elements)
    raise ValueError(f"type de valeur {type_valeur} inconnu de l'écrivain de test")


def _aligner(n: int, alignement: int) -> int:
    return -(-n // alignement) * alignement


def ecrire(chemin: Path, metadonnees=(), tenseurs=(), version: int = 3, alignement: int = 32) -> Path:
    cles = b"".join(_texte(cle) + struct.pack("<I", type_valeur) + _valeur(type_valeur, valeur)
                    for cle, type_valeur, valeur in metadonnees)
    infos = b""
    poids = []
    decalage = 0
    for nom, dimensions, type_tenseur, donnees in tenseurs:
        infos += (_texte(nom) + struct.pack("<I", len(dimensions)) + b"".join(struct.pack("<Q", d) for d in dimensions)
                  + struct.pack("<IQ", type_tenseur, decalage))
        poids.append(donnees)
        decalage = _aligner(decalage + len(donnees), alignement)
    debut = b"GGUF" + struct.pack("<IQQ", version, len(tenseurs), len(metadonnees)) + cles + infos
    contenu = debut + b"\0" * (_aligner(len(debut), alignement) - len(debut))
    for donnees in poids:
        contenu += donnees + b"\0" * (_aligner(len(donnees), alignement) - len(donnees))
    chemin.write_bytes(contenu)
    return chemin
