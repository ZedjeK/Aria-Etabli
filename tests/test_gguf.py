"""appli/gguf.py : le lecteur lit juste, refuse net ce qui est faux, et n'écrit jamais."""

import os
import struct

import pytest

from appli import gguf
from tests.outils_gguf import (ARRAY, BOOL, F32, F64, I8, I16, I32, I64, STRING, TENSEUR_F32, TENSEUR_INCONNU, TENSEUR_Q8_0,
                               U8, U16, U32, U64, ecrire)


def test_lit_tous_les_types_de_valeurs(tmp_path):
    metadonnees = [
        ("a.u8", U8, 200), ("a.i8", I8, -5), ("a.u16", U16, 65000), ("a.i16", I16, -300), ("a.u32", U32, 4_000_000_000),
        ("a.i32", I32, -7), ("a.f32", F32, 1.5), ("a.bool", BOOL, True), ("a.texte", STRING, "héllo ✓"),
        ("a.u64", U64, 2 ** 63 + 5), ("a.i64", I64, -2 ** 40), ("a.f64", F64, 0.1),
    ]
    fichier = gguf.lire(ecrire(tmp_path / "m.gguf", metadonnees))
    assert fichier.metadonnees == {cle: valeur for cle, _, valeur in metadonnees}
    assert fichier.version == 3
    assert fichier.tenseurs == ()


def test_les_tableaux_gardent_un_apercu_et_le_vrai_nombre(tmp_path):
    metadonnees = [
        ("t.nombres", ARRAY, (U32, list(range(20)))),
        ("t.textes", ARRAY, (STRING, [f"jeton{i}" for i in range(12)])),
        ("t.vide", ARRAY, (U32, [])),
        ("t.flottants", ARRAY, (F32, [0.5, 1.5])),
    ]
    meta = gguf.lire(ecrire(tmp_path / "m.gguf", metadonnees)).metadonnees
    assert meta["t.nombres"] == gguf.Tableau(U32, 20, tuple(range(8)))
    assert meta["t.textes"].nombre == 12
    assert meta["t.textes"].apercu == tuple(f"jeton{i}" for i in range(8))
    assert meta["t.textes"].nom_type == "STRING"
    assert meta["t.vide"] == gguf.Tableau(U32, 0, ())
    assert meta["t.flottants"].apercu == (0.5, 1.5)


def test_tailles_exactes_pour_les_types_connus(tmp_path):
    tenseurs = [("blk.0.attn_q.weight", (4, 3), TENSEUR_F32, bytes(48)),
                ("token_embd.weight", (64, 2), TENSEUR_Q8_0, bytes(136))]
    fichier = gguf.lire(ecrire(tmp_path / "m.gguf", [("general.architecture", STRING, "llama")], tenseurs))
    premier, second = fichier.tenseurs
    assert (premier.nom, premier.dimensions, premier.type_nom, premier.elements) == ("blk.0.attn_q.weight", (4, 3), "F32", 12)
    assert (premier.octets, premier.octets_exacts, premier.decalage) == (48, True, 0)
    assert (second.type_nom, second.elements, second.octets, second.octets_exacts) == ("Q8_0", 128, 136, True)
    assert second.decalage == 64  # 48 octets de poids, alignés sur 32
    assert fichier.debut_donnees % 32 == 0
    assert fichier.taille == fichier.debut_donnees + 64 + 160


def test_un_type_inconnu_ne_fait_pas_echouer_et_sa_taille_se_deduit(tmp_path):
    tenseurs = [("w", (10,), TENSEUR_INCONNU, bytes(100)), ("x", (2,), TENSEUR_F32, bytes(8))]
    inconnu, connu = gguf.lire(ecrire(tmp_path / "m.gguf", [], tenseurs)).tenseurs
    assert inconnu.type_nom == "inconnu(142)"
    assert (inconnu.octets, inconnu.octets_exacts) == (128, False)  # 100 octets, alignés sur 32
    assert (connu.octets, connu.octets_exacts) == (8, True)


def test_un_type_inconnu_en_dernier_tenseur_se_deduit_de_la_fin_du_fichier(tmp_path):
    dernier = gguf.lire(ecrire(tmp_path / "m.gguf", [], [("w", (10,), TENSEUR_INCONNU, bytes(100))])).tenseurs[0]
    assert not dernier.octets_exacts
    assert 100 <= dernier.octets < 100 + 32


def test_alignement_personnalise(tmp_path):
    tenseurs = [("a", (2,), TENSEUR_F32, bytes(8)), ("b", (2,), TENSEUR_F32, bytes(8))]
    fichier = gguf.lire(ecrire(tmp_path / "m.gguf", [("general.alignment", U32, 64)], tenseurs, alignement=64))
    assert fichier.alignement == 64
    assert fichier.debut_donnees % 64 == 0
    assert [t.decalage for t in fichier.tenseurs] == [0, 64]


def test_la_version_2_se_lit(tmp_path):
    assert gguf.lire(ecrire(tmp_path / "m.gguf", [("k", U32, 1)], version=2)).version == 2


def test_un_fichier_sans_tenseur_se_lit(tmp_path):
    fichier = gguf.lire(ecrire(tmp_path / "m.gguf", [("k", STRING, "v")]))
    assert fichier.metadonnees == {"k": "v"} and fichier.tenseurs == ()


@pytest.mark.parametrize("contenu, extrait", [
    (b"GGML" + bytes(100), "signature"),
    (b"GGUF" + struct.pack("<I", 9) + bytes(100), "Version GGUF 9"),
    (b"GGUF" + struct.pack(">I", 3) + bytes(100), "gros-boutiste"),
    (b"GGUF", "trop court"),
    (b"", "trop court"),
])
def test_un_fichier_qui_n_est_pas_un_gguf_est_refuse_en_clair(tmp_path, contenu, extrait):
    chemin = tmp_path / "faux.gguf"
    chemin.write_bytes(contenu)
    with pytest.raises(gguf.ErreurGGUF, match=extrait):
        gguf.lire(chemin)


def test_un_dossier_ou_un_fichier_absent_sont_refuses(tmp_path):
    with pytest.raises(gguf.ErreurGGUF, match="dossier"):
        gguf.lire(tmp_path)
    with pytest.raises(gguf.ErreurGGUF, match="Impossible de lire"):
        gguf.lire(tmp_path / "absent.gguf")


@pytest.mark.parametrize("coupe", [30, 45, 70])
def test_un_fichier_coupe_dans_l_en_tete_est_refuse(tmp_path, coupe):
    complet = ecrire(tmp_path / "complet.gguf", [("general.architecture", STRING, "llama"), ("k", U32, 5)],
                     [("w", (4,), TENSEUR_F32, bytes(16))]).read_bytes()
    coupe_chemin = tmp_path / "coupe.gguf"
    coupe_chemin.write_bytes(complet[:coupe])
    with pytest.raises(gguf.ErreurGGUF):
        gguf.lire(coupe_chemin)


def test_un_telechargement_incomplet_est_repere_dans_les_poids(tmp_path):
    complet = ecrire(tmp_path / "complet.gguf", [], [("w", (64, 2), TENSEUR_Q8_0, bytes(136))]).read_bytes()
    incomplet = tmp_path / "incomplet.gguf"
    incomplet.write_bytes(complet[:-40])
    with pytest.raises(gguf.ErreurGGUF, match="tronqué"):
        gguf.lire(incomplet)


def test_un_fichier_coupe_juste_apres_la_description_des_tenseurs_est_refuse(tmp_path):
    complet = ecrire(tmp_path / "complet.gguf", [], [("w", (4,), TENSEUR_F32, bytes(16))])
    contenu = complet.read_bytes()
    complet.write_bytes(contenu[: len(contenu) - 32])  # tout le bloc de poids manque
    with pytest.raises(gguf.ErreurGGUF):
        gguf.lire(complet)


def test_des_comptes_absurdes_sont_refuses_sans_boucler(tmp_path):
    chemin = tmp_path / "absurde.gguf"
    chemin.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 10 ** 12, 0) + bytes(64))
    with pytest.raises(gguf.ErreurGGUF, match="tenseurs"):
        gguf.lire(chemin)
    chemin.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 10 ** 12) + bytes(64))
    with pytest.raises(gguf.ErreurGGUF, match="clés"):
        gguf.lire(chemin)


def test_la_borne_absolue_sur_les_cles_est_appliquee(tmp_path, monkeypatch):
    monkeypatch.setattr(gguf, "MAX_ENTREES", 3)
    chemin = ecrire(tmp_path / "m.gguf", [(f"k{i}", U32, i) for i in range(5)])
    with pytest.raises(gguf.ErreurGGUF, match="clés"):
        gguf.lire(chemin)


def test_un_texte_d_une_longueur_absurde_est_refuse(tmp_path):
    contenu = b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 2 ** 40) + bytes(64)
    chemin = tmp_path / "m.gguf"
    chemin.write_bytes(contenu)
    with pytest.raises(gguf.ErreurGGUF, match="limite"):
        gguf.lire(chemin)


def test_trop_de_dimensions_est_refuse(tmp_path):
    chemin = ecrire(tmp_path / "m.gguf", [], [("w", tuple([1] * 99), TENSEUR_F32, bytes(4))])
    with pytest.raises(gguf.ErreurGGUF, match="dimensions"):
        gguf.lire(chemin)


def test_une_cle_ou_un_tenseur_en_double_est_refuse(tmp_path):
    with pytest.raises(gguf.ErreurGGUF, match="deux fois"):
        gguf.lire(ecrire(tmp_path / "a.gguf", [("k", U32, 1), ("k", U32, 2)]))
    with pytest.raises(gguf.ErreurGGUF, match="deux fois"):
        gguf.lire(ecrire(tmp_path / "b.gguf", [], [("w", (1,), TENSEUR_F32, bytes(4)), ("w", (1,), TENSEUR_F32, bytes(4))]))


@pytest.mark.parametrize("alignement", [0, 33, -32])
def test_un_alignement_invalide_est_refuse(tmp_path, alignement):
    with pytest.raises(gguf.ErreurGGUF, match="alignment"):
        gguf.lire(ecrire(tmp_path / "m.gguf", [("general.alignment", I32, alignement)]))


def test_un_type_de_valeur_inconnu_ou_un_tableau_de_tableaux_sont_refuses(tmp_path):
    with pytest.raises(gguf.ErreurGGUF, match="type de valeur inconnu"):
        gguf.lire(ecrire(tmp_path / "a.gguf", [("k", 13, b"\0\0\0\0")]))
    with pytest.raises(gguf.ErreurGGUF, match="tableau de tableaux"):
        gguf.lire(ecrire(tmp_path / "b.gguf", [("k", ARRAY, (ARRAY, []))]))


def test_la_lecture_ne_modifie_pas_le_fichier(tmp_path):
    chemin = ecrire(tmp_path / "m.gguf", [("k", STRING, "v")], [("w", (4,), TENSEUR_F32, bytes(16))])
    avant, mtime = chemin.read_bytes(), os.stat(chemin).st_mtime_ns
    gguf.lire(chemin)
    assert chemin.read_bytes() == avant
    assert os.stat(chemin).st_mtime_ns == mtime
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.gguf"]


# ----- Recoupement avec le lecteur officiel (paquet gguf, présent pour les tests seulement) -----

def test_les_tables_de_types_sont_celles_du_paquet_officiel():
    officiel = pytest.importorskip("gguf")
    for numero, (nom, elements, octets) in gguf.TYPES_TENSEURS.items():
        type_officiel = officiel.GGMLQuantizationType(numero)
        assert nom == type_officiel.name
        assert (elements, octets) == officiel.GGML_QUANT_SIZES[type_officiel]


def test_le_lecteur_donne_les_memes_choses_que_le_paquet_officiel(tmp_path):
    officiel = pytest.importorskip("gguf")
    np = pytest.importorskip("numpy")
    chemin = tmp_path / "officiel.gguf"
    ecrivain = officiel.GGUFWriter(str(chemin), "llama")
    ecrivain.add_name("Essai officiel")
    ecrivain.add_block_count(2)
    ecrivain.add_uint32("llama.expert_count", 4)
    ecrivain.add_float32("essai.flottant", 0.5)
    ecrivain.add_bool("essai.booleen", True)
    ecrivain.add_array("tokenizer.ggml.tokens", [f"t{i}" for i in range(12)])
    ecrivain.add_array("essai.entiers", list(range(20)))
    ecrivain.add_tensor("t.f32", np.zeros((3, 4), dtype=np.float32))
    ecrivain.add_tensor("t.f16", np.zeros((8,), dtype=np.float16))
    quantifie = officiel.quants.quantize(np.zeros((2, 64), dtype=np.float32), officiel.GGMLQuantizationType.Q8_0)
    ecrivain.add_tensor("t.q8", quantifie, raw_dtype=officiel.GGMLQuantizationType.Q8_0)
    ecrivain.write_header_to_file()
    ecrivain.write_kv_data_to_file()
    ecrivain.write_tensors_to_file()
    ecrivain.close()

    mien = gguf.lire(chemin)
    lecteur = officiel.GGUFReader(str(chemin))

    cles_officielles = [nom for nom in lecteur.fields if not nom.startswith("GGUF.")]
    assert list(mien.metadonnees) == cles_officielles
    for nom in cles_officielles:
        valeur_officielle, valeur = lecteur.fields[nom].contents(), mien.metadonnees[nom]
        if isinstance(valeur, gguf.Tableau):
            assert valeur.nombre == len(valeur_officielle)
            assert list(valeur.apercu) == list(valeur_officielle[: len(valeur.apercu)])
        else:
            assert valeur == valeur_officielle

    assert len(mien.tenseurs) == len(lecteur.tensors) == 3
    for notre, leur in zip(mien.tenseurs, lecteur.tensors):
        assert notre.nom == leur.name
        assert notre.type == int(leur.tensor_type)
        assert notre.elements == leur.n_elements
        assert notre.octets == leur.n_bytes and notre.octets_exacts
        assert mien.debut_donnees + notre.decalage == leur.data_offset
