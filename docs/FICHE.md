# Fiche projet : Aria Etabli

Ce que cette appli a de propre. À relire avant toute modification, et à tenir à
jour quand l'appli change.

| | |
| --- | --- |
| Nom | Aria Etabli (dépôt et exe : `Aria-Etabli`) |
| Promesse | Ouvre un modèle de langage local, change un réglage, et mesure ce que ça change. |
| Élément orange du logo | une loupe, posée entre les jambages du A |
| Fond animé | des petites cases-experts qui s'allument et s'éteignent, comme le routeur d'un MoE ; à chaque passage, l'emplacement des cases et leur nombre sont tirés au hasard, entre 2 et 5 par défaut |
| Stack | Python + FastAPI |
| Droits administrateur | non |
| Démarrage automatique | non |
| Installation par l'exe | non : exe portable seulement |
| Port local (interface web) | 8350 |
| Données de l'utilisateur | `%USERPROFILE%\.Aria-Etabli` (réglages, recettes, journal des mesures, `jeux\` pour les jeux de test téléchargés) ; secrètes : aucune pour l'instant (le jeton de l'interface est tiré à chaque lancement) |
| Écritures hors du dossier de données | aucune par défaut : ni démarrage automatique, ni raccourci du Bureau, ni installation. Une copie de modèle (plusieurs dizaines de Go) n'est écrite que sur demande, dans le dossier que tu choisis ; les modèles d'origine ne sont jamais modifiés |
| Connexions sortantes | `api.github.com` pour chercher une nouvelle version, et `github.com` avec son hôte de téléchargement pour le zip (mises à jour intégrées) ; `api.github.com` encore pour télécharger un jeu de test (GSM8K, dépôt `openai/grade-school-math` figé à un commit), une seule fois, seulement avec `--accord`, en HTTPS, sans suivre de redirection vers un autre hôte, avec vérification de la taille et du SHA-256. Aucune autre, aucune télémétrie, et plus de `huggingface.co` : le jeu n'est pas pris sur Hugging Face. Le `llama-server` lancé en processus enfant n'écoute que sur `127.0.0.1` |
| Composants tiers embarqués | polices Space Grotesk et IBM Plex Mono. llama.cpp (`llama-bench` et `llama-server`) n'est pas encore embarqué : le moteur est fourni avec `--moteur`, il sera livré à côté de l'exe, version épinglée. Ni le paquet Python `gguf` (lecteur maison, `gguf` ne sert qu'aux tests) ni lm-evaluation-harness (son protocole GSM8K est reproduit, sans ses dépendances) ne sont embarqués. Le jeu GSM8K (MIT, © 2021 OpenAI) est une donnée téléchargée, pas un composant de l'exe. Versions, licences et vérifications à relever dans `LICENCES-TIERS.txt` quand un composant arrive |
| Réussite de la v0.1 | le GGUF du Qwen3.6-35B-A3B s'ouvre et affiche son architecture, ses experts totaux et actifs et son template ; le banc tourne sur 5 valeurs d'experts actifs ou plus et trace une courbe qualité contre vitesse, rejouable à l'identique depuis les recettes, avec la configuration complète enregistrée pour chaque mesure |

Les lignes « à renseigner » se complètent avant la première ligne de code.
