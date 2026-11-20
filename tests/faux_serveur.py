"""Faux llama-server pour les tests : mêmes routes, même clé d'API et même format de réponse que le vrai.

Le nom du fichier modèle commande le comportement : « plante » (s'arrête tout de suite, code 3), « lent » (n'est jamais prêt),
« courtjetons » (génère un jeton de moins que demandé), « sanstimings » (réponse sans timings), « pasdejson » (réponse qui
n'est pas du JSON). Si FAUX_ARGS est défini, la ligne de commande reçue y est ajoutée (une ligne JSON par lancement) ; si
FAUX_REQUETES est défini, chaque requête /completion y est notée.

Les vitesses dépendent du nombre d'experts reçu par --override-kv : tg = 300 - 10 x experts, pp = 1000 + 5 x experts, avec un
petit écart d'une requête à l'autre pour que l'écart-type ne soit pas nul.
"""

import http.server
import json
import os
import pathlib
import sys
import threading
import time

FLAGS = {"--no-webui"}


def _lire_options(argv: list[str]) -> dict[str, str]:
    options, i = {}, 0
    while i < len(argv):
        if argv[i] in FLAGS:
            options[argv[i]] = "1"
            i += 1
        else:
            options[argv[i]] = argv[i + 1]
            i += 2
    return options


def main(argv: list[str]) -> int:
    if os.environ.get("FAUX_ARGS"):
        with open(os.environ["FAUX_ARGS"], "a", encoding="utf-8") as fichier:
            fichier.write(json.dumps(argv) + "\n")
    options = _lire_options(argv)
    nom = pathlib.Path(options["-m"]).name
    if "plante" in nom:
        print("load_backend: loaded CPU backend", file=sys.stderr)
        print("error: failed to load model", file=sys.stderr)
        return 3
    cle = options["--api-key"]
    experts = 8
    if "--override-kv" in options:
        experts = int(options["--override-kv"].rsplit(":", 1)[1])
    debut = time.monotonic()
    compteur = {"requetes": 0}

    class Gestionnaire(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def _repondre(self, code: int, corps, brut: bool = False) -> None:
            octets = corps.encode() if brut else json.dumps(corps).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(octets)))
            self.end_headers()
            self.wfile.write(octets)

        def _autorise(self) -> bool:
            if self.headers.get("Authorization") != f"Bearer {cle}":
                self._repondre(401, {"error": {"message": "Invalid API Key"}})
                return False
            return True

        def do_GET(self):
            if self.path == "/health":
                if "lent" in nom or time.monotonic() - debut < 0.3:
                    self._repondre(503, {"error": {"message": "Loading model"}})
                else:
                    self._repondre(200, {"status": "ok"})
            elif self.path == "/props" and self._autorise():
                self._repondre(200, {"build_info": "b12007-fb5997274", "total_slots": 1})
            elif self.path not in ("/health", "/props"):
                self._repondre(404, {})

        def do_POST(self):
            corps = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
            if not self._autorise():
                return
            if self.path == "/tokenize":
                self._repondre(200, {"tokens": list(range(1000, 1010))})
            elif self.path == "/completion":
                compteur["requetes"] += 1
                if os.environ.get("FAUX_REQUETES"):
                    with open(os.environ["FAUX_REQUETES"], "a", encoding="utf-8") as fichier:
                        fichier.write(json.dumps({k: (len(v) if k == "prompt" else v) for k, v in corps.items()}) + "\n")
                if "pasdejson" in nom:
                    self._repondre(200, "oups", brut=True)
                    return
                n = compteur["requetes"]
                demandes = corps["n_predict"]
                reponse = {"content": " texte", "tokens_predicted": demandes, "stop_type": "limit"}
                if "sanstimings" not in nom:
                    reponse["timings"] = {
                        "cache_n": 0, "prompt_n": len(corps["prompt"]), "prompt_ms": 10.0,
                        "prompt_per_second": 1000.0 + 5 * experts + n,
                        "predicted_n": demandes - 1 if "courtjetons" in nom else demandes, "predicted_ms": 100.0,
                        "predicted_per_second": 300.0 - 10 * experts + (n % 3)}
                self._repondre(200, reponse)
            else:
                self._repondre(404, {})

    serveur = http.server.ThreadingHTTPServer((options["--host"], int(options["--port"])), Gestionnaire)
    threading.Thread(target=serveur.serve_forever, daemon=True).start()
    while True:
        time.sleep(0.2)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
