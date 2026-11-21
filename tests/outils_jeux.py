"""Un petit jeu de test de substitution, avec de vraies empreintes, pour essayer le téléchargement et le chargement."""

import hashlib
import json

from appli.jeux import FichierJeu, Jeu

URL = "https://api.github.com/repos/exemple/jeu/contents/data/{}?ref=abc"


def ligne(question: str, nombre: str) -> str:
    return json.dumps({"question": question, "answer": f"On calcule {question}. #### {nombre}"})


# 6 lignes d'entraînement (5 servent d'exemples) et 4 de test
CONTENUS = {
    "train.jsonl": "\n".join(ligne(f"exemple {i} ?", str(i)) for i in range(1, 7)).encode("utf-8") + b"\n",
    "test.jsonl": "\n".join(ligne(f"test {i} ?", str(10 * i)) for i in range(1, 5)).encode("utf-8") + b"\n",
}


def fichier_jeu(nom: str, contenu: bytes) -> FichierJeu:
    return FichierJeu(nom, len(contenu), hashlib.sha256(contenu).hexdigest(), URL.format(nom))


JEU_ESSAI = Jeu("essai", "Jeu d'essai", "MIT, pour les tests", "https://exemple.test/jeu",
                tuple(fichier_jeu(nom, contenu) for nom, contenu in CONTENUS.items()))


class FauxReseau:
    """Remplace appli.jeux._lire_url : sert les contenus de CONTENUS (ou ceux qu'on lui impose) et compte les appels."""

    def __init__(self, contenus: dict[str, bytes] | None = None):
        self.contenus = dict(CONTENUS if contenus is None else contenus)
        self.appels: list[tuple[str, int]] = []

    def __call__(self, url: str, taille_max: int) -> bytes:
        self.appels.append((url, taille_max))
        nom = url.split("?")[0].rsplit("/", 1)[-1]
        return self.contenus[nom]
