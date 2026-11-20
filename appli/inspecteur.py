"""L'inspecteur : ce qu'il y a dans un fichier GGUF, en texte ou en JSON.

    python -m appli inspecter fichier.gguf [--tenseurs] [--json]

Rien n'est écrit, ni dans le dossier de données ni ailleurs : le fichier n'est lu que par appli.gguf, en lecture seule.

Les clés propres à une architecture (nombre de blocs, d'experts, contexte...) portent le nom de l'architecture en préfixe, par
exemple « qwen35.block_count » : ce préfixe est lu dans general.architecture, jamais deviné, et la synthèse donne toujours le
nom complet de la clé d'où vient chaque valeur.
"""

import json
import math
import re
import sys
from typing import Any

from appli import gguf

# Types de fichier (general.file_type) : table du paquet officiel gguf 0.19.0 ; un test la recoupe avec lui
TYPES_FICHIER = {
    0: "ALL_F32", 1: "MOSTLY_F16", 2: "MOSTLY_Q4_0", 3: "MOSTLY_Q4_1", 7: "MOSTLY_Q8_0", 8: "MOSTLY_Q5_0",
    9: "MOSTLY_Q5_1", 10: "MOSTLY_Q2_K", 11: "MOSTLY_Q3_K_S", 12: "MOSTLY_Q3_K_M", 13: "MOSTLY_Q3_K_L",
    14: "MOSTLY_Q4_K_S", 15: "MOSTLY_Q4_K_M", 16: "MOSTLY_Q5_K_S", 17: "MOSTLY_Q5_K_M", 18: "MOSTLY_Q6_K",
    19: "MOSTLY_IQ2_XXS", 20: "MOSTLY_IQ2_XS", 21: "MOSTLY_Q2_K_S", 22: "MOSTLY_IQ3_XS", 23: "MOSTLY_IQ3_XXS",
    24: "MOSTLY_IQ1_S", 25: "MOSTLY_IQ4_NL", 26: "MOSTLY_IQ3_S", 27: "MOSTLY_IQ3_M", 28: "MOSTLY_IQ2_S",
    29: "MOSTLY_IQ2_M", 30: "MOSTLY_IQ4_XS", 31: "MOSTLY_IQ1_M", 32: "MOSTLY_BF16", 36: "MOSTLY_TQ1_0",
    37: "MOSTLY_TQ2_0", 38: "MOSTLY_MXFP4_MOE", 39: "MOSTLY_NVFP4", 40: "MOSTLY_Q1_0", 1024: "GUESSED",
}

_BLOC = re.compile(r"^blk\.(\d+)\.(.+)$")
_SUFFIXES = (".weight", ".bias")
LONGUEUR_VALEUR = 100  # caractères d'une valeur affichée dans la liste des métadonnées (le JSON garde tout)

AIDE = """Usage : python -m appli inspecter FICHIER.GGUF [--tenseurs] [--json]

Montre ce qu'il y a dans un fichier GGUF : architecture, blocs, experts, contexte, tokenizer, template de chat,
quantification, tenseurs. Le fichier n'est ouvert qu'en lecture, rien n'est écrit.

  --tenseurs   ajoute la liste de tous les tenseurs (nom, type, dimensions, taille)
  --json       sortie en JSON plutôt qu'en texte
  --aide       ce message"""


def _sans_suffixe(nom: str) -> str:
    for suffixe in _SUFFIXES:
        if nom.endswith(suffixe):
            return nom[: -len(suffixe)]
    return nom


def _plages(numeros: list[int]) -> str:
    """[0, 1, 2, 4, 5, 7] devient « 0-2, 4-5, 7 »."""
    plages = []
    debut = precedent = None
    for n in sorted(numeros):
        if debut is None:
            debut = precedent = n
        elif n == precedent + 1:
            precedent = n
        else:
            plages.append((debut, precedent))
            debut = precedent = n
    if debut is not None:
        plages.append((debut, precedent))
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in plages)


def _valeur_json(valeur: Any) -> Any:
    if isinstance(valeur, gguf.Tableau):
        return {"tableau": valeur.nom_type, "nombre": valeur.nombre, "apercu": [_valeur_json(v) for v in valeur.apercu]}
    if isinstance(valeur, float) and not math.isfinite(valeur):
        return str(valeur)
    return valeur


def _nombre_entier(valeur: Any) -> int | None:
    return valeur if isinstance(valeur, int) and not isinstance(valeur, bool) else None


def synthese(fichier: gguf.Fichier, avec_tenseurs: bool = False) -> dict[str, Any]:
    meta = fichier.metadonnees
    avertissements: list[str] = []

    architecture = meta.get("general.architecture")
    if not isinstance(architecture, str) or not architecture:
        architecture = None
        avertissements.append("La clé general.architecture est absente : les clés propres à l'architecture sont introuvables.")

    def cle_arch(suffixe: str) -> str | None:
        if architecture is None:
            return None
        cle = f"{architecture}.{suffixe}"
        return cle if cle in meta else None

    def lire_arch(suffixe: str) -> tuple[str | None, int | None]:
        cle = cle_arch(suffixe)
        return cle, _nombre_entier(meta[cle]) if cle else None

    cle_blocs, blocs_annonces = lire_arch("block_count")
    cle_contexte, contexte = lire_arch("context_length")
    cle_embedding, embedding = lire_arch("embedding_length")
    cle_totaux, totaux = lire_arch("expert_count")
    cle_actifs, actifs = lire_arch("expert_used_count")
    cle_partages, partages = lire_arch("expert_shared_count")
    moe = bool(totaux)

    type_fichier = _nombre_entier(meta.get("general.file_type"))
    modele_tokenizer = meta.get("tokenizer.ggml.model")
    liste_tokens = meta.get("tokenizer.ggml.tokens")
    modeles = {cle: valeur for cle, valeur in meta.items() if cle == "tokenizer.chat_template" or cle.startswith("tokenizer.chat_template.")}
    template = meta.get("tokenizer.chat_template")
    autres_templates = sorted(cle.removeprefix("tokenizer.chat_template.") for cle in modeles if cle != "tokenizer.chat_template")

    # Tenseurs : totaux, puis par type
    par_type: dict[int, dict[str, Any]] = {}
    for tenseur in fichier.tenseurs:
        ligne = par_type.setdefault(tenseur.type, {"type": tenseur.type_nom, "id": tenseur.type, "nombre": 0,
                                                   "parametres": 0, "octets": 0, "exacts": True})
        ligne["nombre"] += 1
        ligne["parametres"] += tenseur.elements
        ligne["octets"] += tenseur.octets
        ligne["exacts"] = ligne["exacts"] and tenseur.octets_exacts
    for ligne in par_type.values():
        ligne["bits_par_poids"] = round(ligne["octets"] * 8 / ligne["parametres"], 3) if ligne["parametres"] else None
    total_parametres = sum(t.elements for t in fichier.tenseurs)
    total_octets = sum(t.octets for t in fichier.tenseurs)
    tous_exacts = all(t.octets_exacts for t in fichier.tenseurs)
    inconnus = sum(1 for t in fichier.tenseurs if t.type not in gguf.TYPES_TENSEURS)
    if inconnus:
        avertissements.append(f"{inconnus} tenseur(s) d'un type inconnu de ce lecteur : leur taille est déduite des décalages "
                              "(à l'alignement près), pas calculée.")

    # Blocs : regroupés par ensemble de tenseurs identique
    classes: dict[tuple[str, ...], list[int]] = {}
    hors_blocs: set[str] = set()
    par_bloc: dict[int, set[str]] = {}
    for tenseur in fichier.tenseurs:
        trouve = _BLOC.match(tenseur.nom)
        if trouve:
            par_bloc.setdefault(int(trouve.group(1)), set()).add(_sans_suffixe(trouve.group(2)))
        else:
            hors_blocs.add(_sans_suffixe(tenseur.nom))
    for numero, noms in par_bloc.items():
        classes.setdefault(tuple(sorted(noms)), []).append(numero)
    liste_classes = sorted(({"blocs": _plages(numeros), "nombre": len(numeros), "tenseurs": list(signature)}
                            for signature, numeros in classes.items()),
                           key=lambda c: int(re.match(r"\d+", c["blocs"]).group()))
    if blocs_annonces is not None and par_bloc and blocs_annonces != len(par_bloc):
        avertissements.append(f"La clé {cle_blocs} annonce {blocs_annonces} blocs, les tenseurs en décrivent {len(par_bloc)} : "
                              "des blocs en plus ou en moins (une couche de prédiction multi-jetons, par exemple).")

    resultat: dict[str, Any] = {
        "fichier": {"chemin": str(fichier.chemin), "taille_octets": fichier.taille, "version_gguf": fichier.version,
                    "alignement": fichier.alignement, "debut_donnees": fichier.debut_donnees},
        "general": {
            "nom": meta.get("general.name"),
            "architecture": architecture,
            "prefixe_des_cles": f"{architecture}." if architecture else None,
            "etiquette_de_taille": meta.get("general.size_label"),
            "type_de_fichier": None if type_fichier is None else {
                "id": type_fichier, "nom": TYPES_FICHIER.get(type_fichier, f"inconnu({type_fichier})").removeprefix("MOSTLY_")
                .removeprefix("ALL_")},
        },
        "architecture": {
            "blocs": {"valeur": blocs_annonces, "cle": cle_blocs},
            "contexte": {"valeur": contexte, "cle": cle_contexte},
            "dimension_embedding": {"valeur": embedding, "cle": cle_embedding},
        },
        "experts": {
            "moe": moe,
            "totaux": {"valeur": totaux, "cle": cle_totaux},
            "actifs": {"valeur": actifs, "cle": cle_actifs},
            "partages": {"valeur": partages, "cle": cle_partages},
        },
        "tokenizer": {
            "modele": modele_tokenizer if isinstance(modele_tokenizer, str) else None,
            "taille_vocabulaire": liste_tokens.nombre if isinstance(liste_tokens, gguf.Tableau) else None,
            "template_de_chat": {"present": isinstance(template, str),
                                 "caracteres": len(template) if isinstance(template, str) else 0,
                                 "autres_modeles": autres_templates},
        },
        "tenseurs": {
            "nombre": len(fichier.tenseurs), "parametres": total_parametres, "octets": total_octets,
            "octets_exacts": tous_exacts,
            "bits_par_poids": round(total_octets * 8 / total_parametres, 3) if total_parametres else None,
            "par_type": sorted(par_type.values(), key=lambda ligne: -ligne["octets"]),
        },
        "blocs": {"nombre_dans_les_tenseurs": len(par_bloc), "classes": liste_classes, "hors_blocs": sorted(hors_blocs)},
        "metadonnees": {cle: _valeur_json(valeur) for cle, valeur in meta.items()},
        "avertissements": avertissements,
    }
    if avec_tenseurs:
        resultat["liste_tenseurs"] = [
            {"nom": t.nom, "type": t.type_nom, "dimensions": list(t.dimensions), "octets": t.octets,
             "octets_exacts": t.octets_exacts} for t in fichier.tenseurs]
    return resultat


# ----- Rendu en texte -----

def _entier(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _octets(n: int, exact: bool = True) -> str:
    approx = "" if exact else "~"
    for unite, taille in (("Gio", 1 << 30), ("Mio", 1 << 20), ("Kio", 1 << 10)):
        if n >= taille:
            return f"{approx}{n / taille:.2f} {unite}".replace(".", ",")
    return f"{approx}{n} o"


def _parametres(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f} milliards".replace(".", ",")
    if n >= 1_000_000:
        return f"{n / 1e6:.1f} millions".replace(".", ",")
    return _entier(n)


def _court(valeur: Any) -> str:
    if isinstance(valeur, dict) and "tableau" in valeur:
        debut = ", ".join(repr(v) if isinstance(v, str) else str(v) for v in valeur["apercu"])
        suite = ", ..." if valeur["nombre"] > len(valeur["apercu"]) else ""
        texte = f"[{_entier(valeur['nombre'])} x {valeur['tableau']}] {debut}{suite}"
    else:
        texte = repr(valeur) if isinstance(valeur, str) else str(valeur)
    texte = texte.replace("\n", "\\n").replace("\r", "\\r")
    return texte if len(texte) <= LONGUEUR_VALEUR else texte[: LONGUEUR_VALEUR - 3] + "..."


def _avec_cle(element: dict[str, Any], format_valeur=str) -> str:
    if element["valeur"] is None:
        return "non indiqué dans le fichier"
    return f"{format_valeur(element['valeur'])}  ({element['cle']})"


def texte(resume: dict[str, Any], template: str | None) -> str:
    f, g, a, e, t, ts, b = (resume[k] for k in ("fichier", "general", "architecture", "experts", "tokenizer", "tenseurs", "blocs"))
    lignes: list[str] = []
    ajouter = lignes.append

    ajouter(f"Fichier        : {f['chemin']}")
    ajouter(f"Taille         : {_octets(f['taille_octets'])}   GGUF version {f['version_gguf']}")
    ajouter(f"Nom            : {g['nom'] or 'non indiqué'}")
    if g["architecture"]:
        ajouter(f"Architecture   : {g['architecture']}   (préfixe des clés : {g['prefixe_des_cles']})")
    else:
        ajouter("Architecture   : non indiquée dans le fichier")
    if g["etiquette_de_taille"]:
        ajouter(f"Étiquette      : {g['etiquette_de_taille']}")
    quant = g["type_de_fichier"]
    ajouter("Quantification : " + (f"{quant['nom']}  (libellé de general.file_type ; le détail réel est dans les types ci-dessous)"
                                   if quant else "non indiquée (general.file_type absent)"))
    exact = ts["octets_exacts"]
    ajouter(f"Paramètres     : {_parametres(ts['parametres'])}  ({_entier(ts['parametres'])}, somme des tenseurs)")
    bits = ts["bits_par_poids"]
    ajouter(f"Poids          : {_octets(ts['octets'], exact)}  ({str(bits).replace('.', ',')} bits par poids en moyenne)"
            if bits is not None else f"Poids          : {_octets(ts['octets'], exact)}")
    ajouter(f"Blocs          : {_avec_cle(a['blocs'])}")
    ajouter(f"Contexte       : {_avec_cle(a['contexte'], _entier)}")
    ajouter(f"Embedding      : {_avec_cle(a['dimension_embedding'], _entier)}")
    if e["moe"]:
        ajouter("Experts        : MoE (mélange d'experts)")
        ajouter(f"  totaux       : {_avec_cle(e['totaux'])}")
        ajouter(f"  actifs       : {_avec_cle(e['actifs'])}")
        ajouter(f"  partagés     : {_avec_cle(e['partages'])}")
    else:
        ajouter("Experts        : aucun (modèle dense : pas de clé expert_count dans le fichier)")
    vocabulaire = t["taille_vocabulaire"]
    ajouter(f"Tokenizer      : {t['modele'] or 'non indiqué'}, "
            f"{_entier(vocabulaire) + ' tokens' if vocabulaire is not None else 'vocabulaire absent'}")
    modele = t["template_de_chat"]
    suite = f" ; autres modèles : {', '.join(modele['autres_modeles'])}" if modele["autres_modeles"] else ""
    ajouter("Template       : " + (f"présent, {_entier(modele['caracteres'])} caractères{suite} (texte en fin de sortie)"
                                   if modele["present"] else f"absent{suite}"))

    if resume["avertissements"]:
        ajouter("")
        ajouter("À noter :")
        lignes.extend(f"  - {texte_avertissement}" for texte_avertissement in resume["avertissements"])

    ajouter("")
    ajouter("Tenseurs par type")
    ajouter(f"  {'type':<14}{'nombre':>7}  {'paramètres':>16}  {'taille':>12}  {'bits/poids':>10}")
    for ligne in ts["par_type"]:
        bits_ligne = "" if ligne["bits_par_poids"] is None else f"{ligne['bits_par_poids']:.2f}".replace(".", ",")
        ajouter(f"  {ligne['type']:<14}{ligne['nombre']:>7}  {_entier(ligne['parametres']):>16}  "
                f"{_octets(ligne['octets'], ligne['exacts']):>12}  {bits_ligne:>10}")
    ajouter(f"  {'total':<14}{ts['nombre']:>7}  {_entier(ts['parametres']):>16}  {_octets(ts['octets'], exact):>12}")

    ajouter("")
    ajouter(f"Blocs ({b['nombre_dans_les_tenseurs']} décrits par les tenseurs)")
    if b["classes"]:
        for classe in b["classes"]:
            ajouter(f"  blocs {classe['blocs']} ({classe['nombre']}) : {', '.join(classe['tenseurs'])}")
    else:
        ajouter("  aucun tenseur de la forme blk.N.*")
    if b["hors_blocs"]:
        ajouter(f"  hors blocs : {', '.join(b['hors_blocs'])}")

    ajouter("")
    ajouter(f"Métadonnées ({len(resume['metadonnees'])} clés)")
    largeur = max((len(cle) for cle in resume["metadonnees"]), default=0)
    for cle, valeur in resume["metadonnees"].items():
        ajouter(f"  {cle:<{largeur}} = {_court(valeur)}")

    if "liste_tenseurs" in resume:
        ajouter("")
        ajouter(f"Tous les tenseurs ({len(resume['liste_tenseurs'])})")
        for tenseur in resume["liste_tenseurs"]:
            dimensions = " x ".join(str(d) for d in tenseur["dimensions"]) or "scalaire"
            ajouter(f"  {tenseur['nom']}  {tenseur['type']}  [{dimensions}]  {_octets(tenseur['octets'], tenseur['octets_exacts'])}")

    ajouter("")
    ajouter("Template de chat")
    if template:
        lignes.extend("  " + ligne for ligne in template.splitlines())
    else:
        ajouter("  (absent)")
    return "\n".join(lignes)


def main(arguments: list[str]) -> int:
    """La commande « inspecter ». 0 : fait ; 1 : fichier illisible ; 2 : mauvais usage."""
    if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # un template peut contenir des caractères que la console ne sait pas écrire
    options = {a for a in arguments if a.startswith("--")}
    fichiers = [a for a in arguments if not a.startswith("--")]
    if "--aide" in options:
        print(AIDE)
        return 0
    inconnues = options - {"--tenseurs", "--json"}
    if inconnues or len(fichiers) != 1:
        if inconnues:
            print(f"Option inconnue : {', '.join(sorted(inconnues))}", file=sys.stderr)
        print(AIDE, file=sys.stderr)
        return 2
    try:
        fichier = gguf.lire(fichiers[0])
    except gguf.ErreurGGUF as erreur:
        print(f"Impossible d'inspecter ce fichier : {erreur}", file=sys.stderr)
        return 1
    resume = synthese(fichier, avec_tenseurs="--tenseurs" in options)
    if "--json" in options:
        print(json.dumps(resume, ensure_ascii=True, indent=2))
    else:
        modele = fichier.metadonnees.get("tokenizer.chat_template")
        print(texte(resume, modele if isinstance(modele, str) else None))
    return 0
