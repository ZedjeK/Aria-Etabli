"""Aides communes aux tests de mesure : faux moteur (llama-bench et llama-server) et petits modèles GGUF."""

import sys
from pathlib import Path

from appli.moteur import Moteur
from tests.outils_gguf import STRING, TENSEUR_F32, U32, ecrire

FAUX_BENCH = Path(__file__).parent / "faux_bench.py"
FAUX_SERVEUR = Path(__file__).parent / "faux_serveur.py"


def moteur_faux(dossier: Path) -> Moteur:
    return Moteur(dossier, (sys.executable, str(FAUX_BENCH)), (sys.executable, str(FAUX_SERVEUR)))


def modele_gguf(dossier: Path, nom: str = "essai.gguf") -> Path:
    """Un modèle dense : pas de clé d'experts."""
    return ecrire(dossier / nom, [("general.architecture", STRING, "llama"), ("general.file_type", U32, 1)],
                  [("blk.0.attn_q.weight", (4, 4), TENSEUR_F32, bytes(64))])


def modele_moe(dossier: Path, nom: str = "moe.gguf", totaux: int = 16, actifs: int = 4) -> Path:
    """Un modèle MoE : 16 experts, 4 actifs par défaut, préfixe d'architecture « qwen35moe »."""
    return ecrire(dossier / nom, [
        ("general.architecture", STRING, "qwen35moe"), ("general.file_type", U32, 15),
        ("qwen35moe.block_count", U32, 1), ("qwen35moe.expert_count", U32, totaux),
        ("qwen35moe.expert_used_count", U32, actifs)],
        [("blk.0.ffn_gate_exps.weight", (4, 4), TENSEUR_F32, bytes(64))])
