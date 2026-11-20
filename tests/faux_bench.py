"""Faux llama-bench pour les tests : imite la ligne de commande et le JSON du vrai (voir donnees/llama-bench-reel.json).

Le nom du fichier modèle commande le comportement : « plante » (code 3), « lent » (dort 30 s), « pasjson » (n'écrit pas de
JSON), « bruit » (écrit d'autres lignes avant et après le JSON). Si FAUX_ARGS est défini, la ligne de commande reçue y est écrite.
"""

import json
import os
import pathlib
import sys
import time

FLASH = {"on": 1, "off": 0, "auto": -1}


def main(argv: list[str]) -> int:
    if os.environ.get("FAUX_ARGS"):
        pathlib.Path(os.environ["FAUX_ARGS"]).write_text(json.dumps(argv), encoding="utf-8")
    options = dict(zip(argv[::2], argv[1::2]))
    modele = options["-m"]
    nom = pathlib.Path(modele).name
    if "plante" in nom:
        print("load_backend: loaded CPU backend", file=sys.stderr)
        print("error: failed to load model", file=sys.stderr)
        return 3
    if "lent" in nom:
        time.sleep(30)
    if "pasjson" in nom:
        print("bonjour")
        return 0
    base = json.loads((pathlib.Path(__file__).parent / "donnees" / "llama-bench-reel.json").read_text(encoding="utf-8"))[0]
    prompt, generation, profondeur, repetitions = (int(options[cle]) for cle in ("-p", "-n", "-d", "-r"))
    enregistrements = []
    for n_prompt, n_gen, moyenne, ecart in ((prompt, 0, 1000.0, 10.0), (0, generation, 100.0, 1.0)):
        if not n_prompt and not n_gen:
            continue
        e = dict(base)
        e.update(model_filename=modele, model_size=os.path.getsize(modele), n_prompt=n_prompt, n_gen=n_gen,
                 n_depth=profondeur, n_gpu_layers=int(options.get("-ngl", -1)), n_cpu_moe=int(options.get("-ncmoe", 0)),
                 n_threads=int(options.get("-t", 12)), flash_attn=FLASH.get(options.get("-fa", "auto"), -1),
                 avg_ts=moyenne, stddev_ts=ecart,
                 samples_ts=[moyenne + (ecart if i % 2 else -ecart) for i in range(repetitions)])
        enregistrements.append(e)
    if "bruit" in nom:
        print("load_backend: loaded CPU backend [info]")
    print(json.dumps(enregistrements, indent=2))
    if "bruit" in nom:
        print("fin [ok]")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
