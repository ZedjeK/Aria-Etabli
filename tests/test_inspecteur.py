"""appli/inspecteur.py : la synthèse dit ce que le fichier dit, rien de deviné, et la commande ne touche à rien."""

import json
import pathlib
import sys

import pytest

from appli import __main__ as lancement
from appli import gguf, inspecteur
from tests.outils_gguf import ARRAY, STRING, TENSEUR_F32, TENSEUR_INCONNU, U32, ecrire

GABARIT = "{% for m in messages %}{{ m.content }} ✓ 😀{% endfor %}"


def _moe(tmp_path, blocs_annonces=5):
    metadonnees = [
        ("general.architecture", STRING, "qwen35moe"), ("general.name", STRING, "Essai MoE"), ("general.file_type", U32, 15),
        ("qwen35moe.block_count", U32, blocs_annonces), ("qwen35moe.context_length", U32, 4096),
        ("qwen35moe.embedding_length", U32, 64), ("qwen35moe.expert_count", U32, 128),
        ("qwen35moe.expert_used_count", U32, 8), ("qwen35moe.expert_shared_count", U32, 1),
        ("tokenizer.ggml.model", STRING, "gpt2"), ("tokenizer.ggml.tokens", ARRAY, (STRING, [f"t{i}" for i in range(10)])),
        ("tokenizer.chat_template", STRING, GABARIT),
    ]
    quatre = bytes(16)
    tenseurs = [("token_embd.weight", (4, 2), TENSEUR_F32, bytes(32))]
    for bloc in (0, 1, 2, 4):
        tenseurs += [(f"blk.{bloc}.attn_q.weight", (2, 2), TENSEUR_F32, quatre),
                     (f"blk.{bloc}.ffn_gate_exps.weight", (2, 2), TENSEUR_F32, quatre)]
    tenseurs += [("blk.3.ssm_a", (2, 2), TENSEUR_F32, quatre), ("output_norm.weight", (2,), TENSEUR_F32, bytes(8))]
    return ecrire(tmp_path / "moe.gguf", metadonnees, tenseurs)


def test_la_synthese_donne_les_experts_avec_leurs_cles(tmp_path):
    resume = inspecteur.synthese(gguf.lire(_moe(tmp_path)))
    assert resume["general"]["architecture"] == "qwen35moe"
    assert resume["general"]["prefixe_des_cles"] == "qwen35moe."
    assert resume["general"]["type_de_fichier"] == {"id": 15, "nom": "Q4_K_M"}
    assert resume["architecture"]["blocs"] == {"valeur": 5, "cle": "qwen35moe.block_count"}
    assert resume["architecture"]["contexte"] == {"valeur": 4096, "cle": "qwen35moe.context_length"}
    assert resume["experts"] == {"moe": True,
                                 "totaux": {"valeur": 128, "cle": "qwen35moe.expert_count"},
                                 "actifs": {"valeur": 8, "cle": "qwen35moe.expert_used_count"},
                                 "partages": {"valeur": 1, "cle": "qwen35moe.expert_shared_count"}}
    assert resume["tokenizer"]["modele"] == "gpt2"
    assert resume["tokenizer"]["taille_vocabulaire"] == 10
    assert resume["tokenizer"]["template_de_chat"] == {"present": True, "caracteres": len(GABARIT), "autres_modeles": []}
    assert resume["avertissements"] == []


def test_le_prefixe_vient_du_fichier_jamais_d_une_supposition(tmp_path):
    metadonnees = [("general.architecture", STRING, "zzz"), ("zzz.expert_count", U32, 16),
                   ("qwen3moe.expert_used_count", U32, 99)]  # une clé qui ressemble, mais d'une autre architecture
    resume = inspecteur.synthese(gguf.lire(ecrire(tmp_path / "m.gguf", metadonnees)))
    assert resume["experts"]["totaux"] == {"valeur": 16, "cle": "zzz.expert_count"}
    assert resume["experts"]["actifs"] == {"valeur": None, "cle": None}


def test_sans_architecture_la_synthese_le_dit(tmp_path):
    resume = inspecteur.synthese(gguf.lire(ecrire(tmp_path / "m.gguf", [("qwen3moe.expert_count", U32, 8)])))
    assert resume["general"]["architecture"] is None
    assert resume["experts"]["moe"] is False
    assert any("general.architecture" in a for a in resume["avertissements"])


def test_un_modele_dense_n_a_pas_d_experts(tmp_path):
    chemin = ecrire(tmp_path / "dense.gguf", [("general.architecture", STRING, "llama"), ("llama.block_count", U32, 2)])
    resume = inspecteur.synthese(gguf.lire(chemin))
    assert resume["experts"]["moe"] is False
    assert "aucun (modèle dense" in inspecteur.texte(resume, None)


def test_les_blocs_se_regroupent_par_ensemble_de_tenseurs(tmp_path):
    blocs = inspecteur.synthese(gguf.lire(_moe(tmp_path)))["blocs"]
    assert blocs["nombre_dans_les_tenseurs"] == 5
    assert [(c["blocs"], c["nombre"], c["tenseurs"]) for c in blocs["classes"]] == [
        ("0-2, 4", 4, ["attn_q", "ffn_gate_exps"]),
        ("3", 1, ["ssm_a"]),
    ]
    assert blocs["hors_blocs"] == ["output_norm", "token_embd"]


def test_un_nombre_de_blocs_different_de_la_cle_est_signale(tmp_path):
    resume = inspecteur.synthese(gguf.lire(_moe(tmp_path, blocs_annonces=4)))
    assert any("annonce 4 blocs" in a and "en décrivent 5" in a for a in resume["avertissements"])


def test_les_tenseurs_sont_totalises_par_type(tmp_path):
    tenseurs = inspecteur.synthese(gguf.lire(_moe(tmp_path)))["tenseurs"]
    assert tenseurs["nombre"] == 11
    assert tenseurs["parametres"] == 8 + 4 * 4 * 2 + 4 + 2  # embedding, 4 blocs x 2 tenseurs de 4, ssm_a, output_norm
    assert tenseurs["octets_exacts"] is True
    assert tenseurs["bits_par_poids"] == 32.0
    assert [ligne["type"] for ligne in tenseurs["par_type"]] == ["F32"]


def test_un_type_inconnu_est_signale_avec_ses_bits_par_poids(tmp_path):
    chemin = ecrire(tmp_path / "m.gguf", [("general.architecture", STRING, "llama")],
                    [("w", (256,), TENSEUR_INCONNU, bytes(64)), ("x", (2,), TENSEUR_F32, bytes(8))])
    resume = inspecteur.synthese(gguf.lire(chemin))
    assert resume["tenseurs"]["octets_exacts"] is False
    assert any("type inconnu" in a for a in resume["avertissements"])
    inconnu = next(ligne for ligne in resume["tenseurs"]["par_type"] if ligne["id"] == TENSEUR_INCONNU)
    assert inconnu["type"] == "inconnu(142)" and inconnu["exacts"] is False
    assert "~" in inspecteur.texte(resume, None)  # une taille déduite s'affiche avec un tilde


def test_le_texte_montre_l_essentiel_et_finit_par_le_template(tmp_path):
    fichier = gguf.lire(_moe(tmp_path))
    sortie = inspecteur.texte(inspecteur.synthese(fichier), GABARIT)
    for attendu in ("Architecture   : qwen35moe   (préfixe des clés : qwen35moe.)", "Q4_K_M",
                    "128  (qwen35moe.expert_count)", "8  (qwen35moe.expert_used_count)", "10 tokens",
                    "blocs 0-2, 4 (4)", "hors blocs : output_norm, token_embd"):
        assert attendu in sortie
    assert sortie.endswith("  " + GABARIT)
    assert "Tous les tenseurs" not in sortie


def test_l_option_tenseurs_ajoute_la_liste_complete(tmp_path):
    resume = inspecteur.synthese(gguf.lire(_moe(tmp_path)), avec_tenseurs=True)
    assert len(resume["liste_tenseurs"]) == 11
    assert "blk.3.ssm_a  F32  [2 x 2]" in inspecteur.texte(resume, None)


def test_un_template_absent_est_dit_absent(tmp_path):
    chemin = ecrire(tmp_path / "m.gguf", [("general.architecture", STRING, "llama")])
    sortie = inspecteur.texte(inspecteur.synthese(gguf.lire(chemin)), None)
    assert "Template       : absent" in sortie and sortie.endswith("  (absent)")


# ----- La commande -----

def test_la_commande_en_texte(tmp_path, capsys):
    assert inspecteur.main([str(_moe(tmp_path))]) == 0
    sortie = capsys.readouterr().out
    assert "Essai MoE" in sortie and GABARIT in sortie


def test_la_commande_en_json_est_du_json_valide_avec_le_template_complet(tmp_path, capsys):
    assert inspecteur.main([str(_moe(tmp_path)), "--json", "--tenseurs"]) == 0
    donnees = json.loads(capsys.readouterr().out)
    assert donnees["experts"]["actifs"]["valeur"] == 8
    assert donnees["metadonnees"]["tokenizer.chat_template"] == GABARIT
    assert donnees["metadonnees"]["tokenizer.ggml.tokens"] == {
        "tableau": "STRING", "nombre": 10, "apercu": [f"t{i}" for i in range(8)]}
    assert len(donnees["liste_tenseurs"]) == 11


def test_les_codes_de_retour(tmp_path, capsys):
    assert inspecteur.main(["--aide"]) == 0
    assert "Usage" in capsys.readouterr().out
    assert inspecteur.main([]) == 2
    assert inspecteur.main([str(_moe(tmp_path)), "--bizarre"]) == 2
    assert "Option inconnue : --bizarre" in capsys.readouterr().err
    assert inspecteur.main([str(tmp_path / "absent.gguf")]) == 1
    assert "Impossible d'inspecter ce fichier" in capsys.readouterr().err
    faux = tmp_path / "faux.gguf"
    faux.write_bytes(b"pas du tout un gguf, mais assez long pour passer la taille minimale")
    assert inspecteur.main([str(faux)]) == 1
    assert "signature" in capsys.readouterr().err


def test_la_commande_ne_touche_a_rien(tmp_path, monkeypatch, capsys):
    maison = tmp_path / "maison"
    maison.mkdir()
    monkeypatch.setattr(pathlib.Path, "home", lambda: maison)
    chemin = _moe(tmp_path)
    avant = chemin.read_bytes()
    assert inspecteur.main([str(chemin), "--tenseurs"]) == 0
    capsys.readouterr()
    assert chemin.read_bytes() == avant
    assert list(maison.iterdir()) == []  # pas de dossier de données créé


def test_la_commande_se_lance_avant_le_serveur_et_le_dossier_de_donnees(tmp_path, monkeypatch, capsys):
    def interdit(*_args, **_kwargs):
        raise AssertionError("le dossier de données ne doit pas être touché par « inspecter »")

    monkeypatch.setattr(lancement.donnees, "assurer", interdit)
    monkeypatch.setattr(sys, "argv", ["Aria-Etabli", "inspecter", str(_moe(tmp_path))])
    assert lancement.main() == 0
    assert "Essai MoE" in capsys.readouterr().out


def test_la_table_des_types_de_fichier_est_celle_du_paquet_officiel():
    officiel = pytest.importorskip("gguf")
    for numero, nom in inspecteur.TYPES_FICHIER.items():
        assert officiel.LlamaFileType(numero).name == nom
