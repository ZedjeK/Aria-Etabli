"""Aides communes aux tests de mesure : un faux moteur et un petit modèle GGUF."""

import sys
from pathlib import Path

from appli.moteur import Moteur
from tests.outils_gguf import STRING, TENSEUR_F32, U32, ecrire

FAUX_BENCH = Path(__file__).parent / "faux_bench.py"


def moteur_faux(dossier: Path) -> Moteur:
    return Moteur(dossier, (sys.executable, str(FAUX_BENCH)), None)


def modele_gguf(dossier: Path, nom: str = "essai.gguf") -> Path:
    return ecrire(dossier / nom, [("general.architecture", STRING, "llama"), ("general.file_type", U32, 1)],
                  [("blk.0.attn_q.weight", (4, 4), TENSEUR_F32, bytes(64))])
