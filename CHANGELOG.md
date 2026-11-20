# Journal des versions

Un bloc par version, le plus récent en haut. Les notes complètes d'une version
sont dans `docs/releases/`.

## Non publié

### Ajouté

- Première version d'Aria Etabli.
- Inspecteur GGUF en ligne de commande (`python -m appli inspecter fichier.gguf`) : architecture, blocs, experts totaux
  et actifs, contexte, tokenizer, template de chat, quantification, tenseurs. Lecture seule, sans dépendance ; ouvre
  aussi les fichiers aux quantifications inconnues du paquet `gguf`, comme les Bonsai.
- Mesure de vitesse avec `llama-bench` (`python -m appli mesurer bench modele.gguf --moteur dossier`) : jetons par seconde
  en lecture du prompt et en génération, écart-type, configuration complète de la mesure, et une règle
  (`verifier_comparables`) qui refuse de comparer des configurations mélangées.
- Mesure de vitesse selon le nombre d'experts actifs d'un modèle MoE (`python -m appli mesurer experts modele.gguf
  --moteur dossier --experts 4,6,8`) : un `llama-server` par valeur, réglé par `--override-kv` sans écrire le modèle,
  sur `127.0.0.1` avec une clé d'API aléatoire, toujours arrêté ensuite. Seule la vitesse est mesurée, pas la qualité.
