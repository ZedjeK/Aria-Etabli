"""Le protocole d'évaluation de la qualité : GSM8K, 5 exemples en préfixe, complétion brute.

C'est le protocole de la tâche `gsm8k` de lm-evaluation-harness 0.4.13 (format « Question: … Answer: », arrêt sur « Question: »,
filtres `strict-match` et `flexible-extract`, exact_match après suppression des virgules, du « $ », de tout ce qui précède « #### »
et du point final), reproduit ici sans ses dépendances. Une différence assumée : lm-eval tire ses 5 exemples au hasard dans le jeu
d'entraînement, Etabli prend toujours les 5 premiers, pour que la mesure se rejoue à l'identique. Les scores d'Etabli ne se
comparent donc pas aux scores publiés par lm-eval.

Ce module ne fait ni réseau ni appel au serveur : il construit les prompts, lit les réponses, les note et donne l'incertitude.
"""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from appli import jeux

NB_EXEMPLES = 5
ARRETS = ("Question:", "</s>", "<|im_end|>")
MAX_JETONS = 256  # défaut de lm-eval pour une génération
TEMPERATURE = 0
GRAINE = 1234
INVALIDE = "[invalid]"
QUESTIONS_PAR_DEFAUT = 200
Z_95 = 1.959963984540054  # quantile de la loi normale pour un intervalle de confiance à 95 %

_STRICT = re.compile(r"#### (\-?[0-9\.\,]+)")
_FLEXIBLE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
_A_IGNORER = (r",", r"\$", r"(?s).*#### ", r"\.$")  # appliqués à la réponse extraite comme à la bonne réponse

PROTOCOLE = {
    "jeu": "gsm8k", "tache": "gsm8k de lm-evaluation-harness 0.4.13, reproduite",
    "mode": "complétion brute, sans gabarit de chat, sans réflexion",
    "exemples": {"nombre": NB_EXEMPLES, "choix": "les 5 premiers de train.jsonl"},
    "format": "Question: {question}\\nAnswer:", "arrets": list(ARRETS), "jetons_max": MAX_JETONS,
    "temperature": TEMPERATURE, "graine": GRAINE, "echantillon": "les N premières questions de test.jsonl, dans l'ordre du fichier",
    "filtres": ["strict-match", "flexible-extract"], "metrique": "exact_match, intervalle de Wilson à 95 %",
}


class ErreurQualite(Exception):
    """L'évaluation ne peut pas se faire : le message dit pourquoi, en français."""


@dataclass(frozen=True)
class Question:
    question: str
    reponse: str  # le raisonnement complet du jeu, qui se termine par « #### <nombre> »


def lire_jsonl(chemin: Path) -> list[Question]:
    """Lit un fichier du jeu GSM8K (une ligne JSON par question). Refuse toute ligne qui n'a pas la forme attendue."""
    resultat = []
    try:
        lignes = chemin.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as erreur:
        raise ErreurQualite(f"Impossible de lire {chemin} : {erreur}") from erreur
    for numero, ligne in enumerate(lignes, start=1):
        try:
            donnees = json.loads(ligne)
            question, reponse = donnees["question"], donnees["answer"]
        except (ValueError, KeyError, TypeError):
            raise ErreurQualite(f"{chemin.name}, ligne {numero} : ce n'est pas une question avec « question » et « answer ».") from None
        if not isinstance(question, str) or not isinstance(reponse, str) or "#### " not in reponse:
            raise ErreurQualite(f"{chemin.name}, ligne {numero} : la réponse n'est pas de la forme « … #### nombre ».")
        resultat.append(Question(question, reponse))
    return resultat


def charger_gsm8k(nombre: int, racine: Path | None = None,
                  jeu: jeux.Jeu = jeux.GSM8K) -> tuple[list[Question], list[Question]]:
    """(les 5 exemples, les `nombre` premières questions de test), après avoir vérifié l'empreinte des deux fichiers."""
    if isinstance(nombre, bool) or not isinstance(nombre, int) or not 1 <= nombre <= 1319:
        raise ErreurQualite(f"Le nombre de questions doit être un entier entre 1 et 1319 (reçu : {nombre!r}).")
    statuts = jeux.etat(jeu, racine)
    if any(statut != "présent" for statut in statuts.values()):
        raise ErreurQualite("Le jeu GSM8K n'est pas téléchargé, ou ne correspond plus à l'empreinte attendue. "
                            "Montre ce qui serait téléchargé avec : python -m appli jeux telecharger gsm8k")
    exemples = lire_jsonl(jeux.chemin_du_fichier(jeu, "train.jsonl", racine))[:NB_EXEMPLES]
    questions = lire_jsonl(jeux.chemin_du_fichier(jeu, "test.jsonl", racine))[:nombre]
    if len(exemples) < NB_EXEMPLES or len(questions) < nombre:
        raise ErreurQualite("Le jeu GSM8K est plus court que prévu : fichier incomplet.")
    return exemples, questions


def construire_prompt(exemples: list[Question], question: str) -> str:
    """Les exemples (question, « Answer: », raisonnement complet) séparés par une ligne vide, puis la question posée."""
    contexte = "\n\n".join(f"Question: {e.question}\nAnswer: {e.reponse}" for e in exemples)
    return f"{contexte}\n\nQuestion: {question}\nAnswer:"


def extraire_strict(texte: str) -> str:
    """Le nombre qui suit le premier « #### », ou « [invalid] »."""
    trouves = _STRICT.findall(texte)
    return trouves[0].strip() if trouves else INVALIDE


def extraire_flexible(texte: str) -> str:
    """Le dernier nombre du texte, ou « [invalid] »."""
    trouves = _FLEXIBLE.findall(texte)
    if not trouves:
        return INVALIDE
    retenus = [element for element in trouves[-1] if element]
    return retenus[0].strip() if retenus else INVALIDE


def _normaliser(texte: str) -> str:
    for motif in _A_IGNORER:
        texte = re.sub(motif, "", texte)
    return texte.lower()


def noter(texte: str, reponse_attendue: str) -> dict[str, object]:
    """Note une génération par les deux filtres. `reponse_attendue` est le raisonnement complet du jeu."""
    attendu = _normaliser(reponse_attendue)
    strict, flexible = extraire_strict(texte), extraire_flexible(texte)
    return {"strict": _normaliser(strict) == attendu, "flexible": _normaliser(flexible) == attendu,
            "extrait_strict": strict, "extrait_flexible": flexible, "attendu": attendu}


def intervalle_wilson(bonnes: int, total: int, z: float = Z_95) -> tuple[float, float]:
    """Intervalle de confiance de Wilson (à 95 % par défaut) d'une proportion. Plus fiable qu'un « ± » naïf aux petits effectifs."""
    if total <= 0 or not 0 <= bonnes <= total:
        raise ErreurQualite(f"Une proportion de {bonnes} sur {total} n'existe pas.")
    p = bonnes / total
    denominateur = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominateur
    demi = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominateur
    return max(0.0, centre - demi), min(1.0, centre + demi)


def score(notes: list[dict[str, object]], filtre: str) -> dict[str, object]:
    """Bonnes réponses, total, proportion et intervalle de Wilson, pour « strict » ou « flexible »."""
    if filtre not in ("strict", "flexible"):
        raise ErreurQualite(f"Filtre inconnu : {filtre!r}.")
    total = len(notes)
    bonnes = sum(1 for n in notes if n[filtre])
    bas, haut = intervalle_wilson(bonnes, total)
    return {"bonnes": bonnes, "total": total, "proportion": bonnes / total, "wilson_bas": bas, "wilson_haut": haut}
