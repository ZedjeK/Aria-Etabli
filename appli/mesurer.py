"""La commande « mesurer » : vitesse d'un modèle GGUF avec un moteur llama.cpp.

    python -m appli mesurer bench MODELE.gguf --moteur DOSSIER [options]
    python -m appli mesurer experts MODELE.gguf --moteur DOSSIER --experts 4,6,8 [options]

Ne touche ni au dossier de données ni au modèle. Écrit un fichier seulement avec --sortie, et jamais par-dessus un fichier
qui existe déjà.
"""

import json
import sys
from pathlib import Path
from typing import Any, Callable

from appli import mesures
from appli import moteur as moteurs
from appli import vitesse_bench, vitesse_serveur
from appli.mesures import ErreurMesure, Resultat

AIDE = """Usage : python -m appli mesurer bench   MODELE.gguf --moteur DOSSIER [options]
        python -m appli mesurer experts MODELE.gguf --moteur DOSSIER --experts 4,6,8 [options]

Mesure la vitesse d'un modèle, en jetons par seconde, avec la configuration complète (modèle, moteur, matériel, réglages).
Ne modifie ni le modèle ni ton dossier de données.

bench : avec llama-bench. Lecture du prompt (pp) et génération (tg), avec l'écart-type de chaque test.
experts : avec llama-server. Une mesure par nombre d'experts actifs d'un modèle MoE, sans écrire le modèle : le réglage
          passe par --override-kv au lancement. Un serveur est lancé puis arrêté pour chaque valeur (127.0.0.1 seulement).
          Les vitesses ne disent rien de la qualité.

Options communes
  --moteur DOSSIER      dossier qui contient llama-bench et llama-server (obligatoire)
  --ngl N               couches sur la carte graphique (0 : tout sur le processeur ; défaut : celui du moteur)
  --ncmoe N             experts des N premières couches gardés sur le processeur
  --prompt P            jetons de prompt (défaut 512 ; avec bench, 0 saute le test de lecture)
  --generation G        jetons à générer (défaut 128)
  -r, --repetitions N   répétitions (défaut 5)
  --flash-attn on|off|auto
  --threads N
  --json                sortie en JSON plutôt qu'en texte
  --sortie FICHIER      écrit aussi le résultat JSON dans ce fichier (refuse un fichier qui existe déjà)
  --aide                ce message

Propres à bench
  --profondeur D        jetons déjà dans le contexte avant le test (défaut 0)
  --delai S             secondes avant d'arrêter llama-bench (défaut 3600)

Propres à experts
  --experts 4,6,8       nombres d'experts actifs à mesurer, de 1 au nombre d'experts du fichier (obligatoire)
  --contexte C          taille du contexte du serveur (défaut 4096)
  --echauffement N      requêtes faites avant la mesure et ignorées (défaut 1)
  --delai S             secondes pour que le modèle se charge (défaut 600)"""


class _Usage(Exception):
    pass


def _liste_entiers(texte: str) -> tuple[int, ...]:
    try:
        return tuple(int(element) for element in texte.split(","))
    except ValueError:
        raise _Usage(f"--experts attend des nombres entiers séparés par des virgules, par exemple 4,6,8 "
                     f"(reçu : {texte!r}).") from None


_COMMUNES: dict[str, tuple[str, Callable[[str], Any]]] = {
    "--moteur": ("moteur", str), "--ngl": ("couches_gpu", int), "--ncmoe": ("experts_cpu", int),
    "--prompt": ("prompt", int), "--generation": ("generation", int), "--repetitions": ("repetitions", int),
    "-r": ("repetitions", int), "--flash-attn": ("flash_attn", str), "--threads": ("threads", int),
    "--sortie": ("sortie", str),
}
_OPTIONS_BENCH = {**_COMMUNES, "--profondeur": ("profondeur", int), "--delai": ("delai_max", int)}
_OPTIONS_EXPERTS = {**_COMMUNES, "--experts": ("experts", _liste_entiers), "--contexte": ("contexte", int),
                    "--echauffement": ("echauffement", int), "--delai": ("delai_demarrage", int)}
_DRAPEAUX = {"--json", "--aide"}


def _lire_arguments(arguments: list[str], options: dict[str, tuple[str, Callable[[str], Any]]]
                    ) -> tuple[list[str], dict[str, Any], set[str]]:
    positionnels: list[str] = []
    valeurs: dict[str, Any] = {}
    drapeaux: set[str] = set()
    i = 0
    while i < len(arguments):
        jeton = arguments[i]
        if jeton in _DRAPEAUX:
            drapeaux.add(jeton)
        elif jeton in options:
            nom, convertir = options[jeton]
            if i + 1 >= len(arguments):
                raise _Usage(f"{jeton} attend une valeur.")
            i += 1
            try:
                valeurs[nom] = convertir(arguments[i])
            except ValueError:
                raise _Usage(f"{jeton} attend un nombre entier (reçu : {arguments[i]!r}).") from None
        elif jeton.startswith("-"):
            raise _Usage(f"Option inconnue : {jeton}")
        else:
            positionnels.append(jeton)
        i += 1
    return positionnels, valeurs, drapeaux


def _entier(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _octets(n: int) -> str:
    for unite, taille in (("Gio", 1 << 30), ("Mio", 1 << 20), ("Kio", 1 << 10)):
        if n >= taille:
            return f"{n / taille:.2f} {unite}".replace(".", ",")
    return f"{n} o"


def _decimal(x: float) -> str:
    return f"{x:.2f}".replace(".", ",")


def _ligne_modele(m: dict[str, Any]) -> str:
    return (f"Modèle    : {m['fichier']} ({m['architecture'] or 'architecture non indiquée'}, "
            f"{m['quantification'] or 'quantification non indiquée'}, {_octets(m['taille_octets'])})")


def texte(resultat: Resultat) -> str:
    c = resultat.configuration
    mo, r = c.moteur, c.reglages
    lignes = [
        _ligne_modele(c.modele),
        f"Moteur    : build {mo.get('build')} (commit {mo.get('commit')}), {mo.get('backends')}",
        f"Matériel  : {mo.get('processeur')} ; {mo.get('carte_graphique')}",
        "Réglages  : " + ", ".join(f"{cle} = {valeur}" for cle, valeur in r.items()
                                    if cle in ("n_gpu_layers", "n_cpu_moe", "n_depth", "flash_attn", "n_threads",
                                               "n_batch", "n_ubatch", "type_k", "type_v")),
        f"Outil     : llama-bench, {len(resultat.mesures[0].echantillons)} répétition(s) par test",
        f"Date      : {resultat.date}",
        "",
        f"  {'test':<14}{'jetons/s':>12}{'écart-type':>12}{'échantillons':>14}",
    ]
    for mesure in resultat.mesures:
        lignes.append(f"  {mesure.test:<14}{_decimal(mesure.jetons_par_seconde):>12}{_decimal(mesure.ecart_type):>12}"
                      f"{len(mesure.echantillons):>14}")
    if resultat.avertissements:
        lignes += ["", "À noter :"] + [f"  - {a}" for a in resultat.avertissements]
    return "\n".join(lignes)


def texte_experts(resultats: list[Resultat]) -> str:
    premier = resultats[0].configuration
    mo, r = premier.moteur, premier.reglages
    natif = r.get("experts_actifs_fichier")
    reference = next((res for res in resultats if res.configuration.reglages["experts_actifs"] == natif), None)
    ref_tg = reference.mesures[1].jetons_par_seconde if reference else None
    lignes = [
        _ligne_modele(premier.modele),
        f"Moteur    : build {mo.get('build')} (commit {mo.get('commit')}) ; {mo.get('processeur')} ; "
        f"{mo.get('carte_graphique')}",
        f"Experts   : {r['experts_totaux']} au total, {natif} actifs dans le fichier ({r['cle_experts']})",
        "Réglages  : " + ", ".join(f"{cle} = {r[cle]}" for cle in ("n_gpu_layers", "contexte", "prompt_jetons",
                                                                   "generation_jetons", "flash_attn", "threads")),
        f"Outil     : llama-server (--override-kv), {r['repetitions']} répétition(s) après {r['echauffement']} "
        "d'échauffement",
        f"Date      : {resultats[-1].date}",
        "",
        f"  {'experts':>8}{'pp jetons/s':>14}{'écart':>10}{'tg jetons/s':>14}{'écart':>10}{'tg / fichier':>15}",
    ]
    for res in resultats:
        pp, tg = res.mesures
        valeur = res.configuration.reglages["experts_actifs"]
        if ref_tg is None:
            comparaison = ""
        elif res is reference:
            comparaison = "référence"
        else:
            comparaison = f"{(tg.jetons_par_seconde / ref_tg - 1) * 100:+.1f} %".replace(".", ",")
        lignes.append(f"  {valeur:>8}{_decimal(pp.jetons_par_seconde):>14}{_decimal(pp.ecart_type):>10}"
                      f"{_decimal(tg.jetons_par_seconde):>14}{_decimal(tg.ecart_type):>10}{comparaison:>15}")
    notes = list(dict.fromkeys(a for res in resultats for a in res.avertissements))
    lignes += ["", "À noter :"] + [f"  - {a}" for a in notes]
    lignes.append("  - Ces vitesses ne disent rien de la qualité : un MoE avec moins d'experts que prévu répond plus vite "
                  "mais moins bien. La qualité se mesure séparément.")
    return "\n".join(lignes)


def _ecrire_sortie(sortie: str | None, donnees: str) -> None:
    if sortie is None:
        return
    try:
        with open(Path(sortie), "x", encoding="utf-8") as fichier:
            fichier.write(donnees + "\n")
    except FileExistsError:
        raise ErreurMesure(f"{sortie} existe déjà : choisis un autre nom, rien n'est écrasé.") from None
    except OSError as erreur:
        raise ErreurMesure(f"Impossible d'écrire {sortie} : {erreur.strerror or erreur}") from erreur


def _preparer(arguments: list[str], options: dict) -> tuple[str, dict[str, Any], set[str], str, str | None] | None:
    positionnels, valeurs, drapeaux = _lire_arguments(arguments, options)
    if "--aide" in drapeaux:
        print(AIDE)
        return None
    if len(positionnels) != 1:
        raise _Usage("Donne un seul fichier modèle.")
    if "moteur" not in valeurs:
        raise _Usage("--moteur DOSSIER est obligatoire.")
    return positionnels[0], valeurs, drapeaux, valeurs.pop("moteur"), valeurs.pop("sortie", None)


def _bench(arguments: list[str]) -> int:
    prepare = _preparer(arguments, _OPTIONS_BENCH)
    if prepare is None:
        return 0
    modele, valeurs, drapeaux, dossier, sortie = prepare
    try:
        moteur = moteurs.reconnaitre(dossier)
        resultat = vitesse_bench.mesurer(moteur, modele, vitesse_bench.OptionsBench(**valeurs))
        donnees = json.dumps(resultat.en_dict(), ensure_ascii=True, indent=2)
        _ecrire_sortie(sortie, donnees)
    except (moteurs.ErreurMoteur, ErreurMesure) as erreur:
        print(f"Mesure impossible : {erreur}", file=sys.stderr)
        return 1
    print(donnees if "--json" in drapeaux else texte(resultat))
    return 0


def _experts(arguments: list[str]) -> int:
    prepare = _preparer(arguments, _OPTIONS_EXPERTS)
    if prepare is None:
        return 0
    modele, valeurs, drapeaux, dossier, sortie = prepare
    if "experts" not in valeurs:
        raise _Usage("--experts 4,6,8 est obligatoire.")
    experts = valeurs.pop("experts")
    try:
        moteur = moteurs.reconnaitre(dossier)
        resultats = vitesse_serveur.mesurer_experts(
            moteur, modele, experts, vitesse_serveur.OptionsServeur(**valeurs),
            progression=lambda message: print(message, file=sys.stderr, flush=True))
        # Seul le nombre d'experts actifs doit varier d'une mesure à l'autre : sinon la comparaison ne vaut rien
        mesures.verifier_comparables([res.configuration for res in resultats], variables=("reglages.experts_actifs",))
        donnees = json.dumps({"comparables": True, "resultats": [res.en_dict() for res in resultats]},
                             ensure_ascii=True, indent=2)
        _ecrire_sortie(sortie, donnees)
    except (moteurs.ErreurMoteur, ErreurMesure) as erreur:
        print(f"Mesure impossible : {erreur}", file=sys.stderr)
        return 1
    print(donnees if "--json" in drapeaux else texte_experts(resultats))
    return 0


def main(arguments: list[str]) -> int:
    """La commande « mesurer ». 0 : fait ; 1 : mesure impossible ; 2 : mauvais usage."""
    if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if arguments == ["--aide"]:
        print(AIDE)
        return 0
    try:
        if not arguments:
            raise _Usage("Donne une sous-commande : bench ou experts.")
        if arguments[0] == "bench":
            return _bench(arguments[1:])
        if arguments[0] == "experts":
            return _experts(arguments[1:])
        raise _Usage(f"Sous-commande inconnue : {arguments[0]} (disponibles : bench, experts).")
    except _Usage as erreur:
        print(erreur, file=sys.stderr)
        print(AIDE, file=sys.stderr)
        return 2
