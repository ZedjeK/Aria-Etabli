<p align="center">
  <img src="docs/logo.png" width="160" alt="Logo d'Aria Etabli : un A néon glitché avec une loupe, posée entre les jambages du A orange">
</p>

<h1 align="center">Aria Etabli</h1>

<p align="center">
  Ouvre un modèle de langage local, change un réglage, et mesure ce que ça change.
</p>

## Ce qu'il fait

Ce que l'appli permet, en clair, sans jargon.

## Avant de commencer

Windows 10 ou 11, 64 bits. Droits, ports et matériel nécessaires.

## Installation

1. Télécharge `Aria-Etabli-<version>.zip` depuis la page des versions.
2. Décompresse d'abord le zip en entier : clic droit, « Extraire tout ».
   Choisis un dossier stable (pas Téléchargements).
   Ne lance pas l'exe depuis l'aperçu du zip dans l'Explorateur : Windows n'en
   extrait que l'exe, sans les fichiers livrés à côté de lui.
3. Lance `Aria-Etabli.exe`.

L'exe n'est pas signé : Windows peut afficher « Windows a protégé votre
ordinateur ». Clique sur « Informations complémentaires », puis sur « Exécuter
quand même ».

## Premiers pas

Le premier lancement, étape par étape.

## Au quotidien

### Inspecter un fichier GGUF

Depuis le dossier du dépôt, avec les dépendances de `requirements.txt` installées :

```powershell
python -m appli inspecter C:\modeles\mon-modele.gguf
python -m appli inspecter C:\modeles\mon-modele.gguf --tenseurs --json
```

La commande montre l'architecture, le nombre de blocs, les experts (totaux, actifs, partagés), le contexte, le tokenizer,
le template de chat, la quantification, les tenseurs par type et les blocs regroupés par ensemble de tenseurs. Chaque
valeur est donnée avec le nom complet de la clé d'où elle vient : le préfixe d'une architecture est lu dans le fichier,
jamais deviné. `--tenseurs` ajoute la liste de tous les tenseurs, `--json` change la sortie en JSON.

Le fichier n'est ouvert qu'en lecture et la commande n'écrit rien. Lancée depuis l'exe, elle laisse seulement celui-ci
créer son dossier d'extraction `cache\` dans ton dossier de données, comme à chaque lancement.

### Télécharger un jeu de test

Les mesures de qualité ont besoin d'un jeu de questions. Le seul connu pour l'instant est GSM8K (problèmes de
mathématiques de primaire, 1 319 questions de test), du dépôt `openai/grade-school-math` (MIT, © 2021 OpenAI).

```powershell
python -m appli jeux                              # ce qui est connu et ce qui est déjà là, sans connexion
python -m appli jeux telecharger gsm8k            # montre ce qui serait téléchargé, sans rien faire
python -m appli jeux telecharger gsm8k --accord   # le télécharge, une seule fois (environ 4,7 Mio)
```

Rien ne se télécharge sans `--accord`. Le téléchargement n'utilise que `api.github.com`, en HTTPS, et ne suit aucune
redirection vers un autre hôte. Les fichiers viennent d'un commit précis du dépôt d'origine et sont vérifiés par leur
taille et leur SHA-256 avant d'être rangés dans `%USERPROFILE%\.Aria-Etabli\jeux\gsm8k` : un fichier qui ne correspond
pas n'est jamais gardé, et un fichier déjà là qui a changé est signalé comme corrompu.

### Mesurer la vitesse d'un modèle

Il faut un moteur llama.cpp : le dossier qui contient `llama-bench` (Etabli n'en embarque pas encore).

```powershell
python -m appli mesurer bench C:\modeles\mon-modele.gguf --moteur C:\llama.cpp
python -m appli mesurer bench C:\modeles\mon-modele.gguf --moteur C:\llama.cpp --ngl 20 --profondeur 4096 --json
```

`llama-bench` mesure la lecture du prompt (`pp`) et la génération (`tg`) en jetons par seconde, avec l'écart-type et tous
les échantillons. La commande y ajoute la **configuration complète** : le modèle (architecture, quantification, types de
tenseurs), le moteur (build, commit, backends), le matériel et tous les réglages que le moteur déclare. Deux mesures ne
se comparent que si cette configuration est la même, à la variable qu'on fait varier près : la fonction
`verifier_comparables` de `appli/mesures.py` fait cette vérification. `mesurer experts` l'applique avant d'afficher son
tableau ; `mesurer bench` ne mesure qu'un modèle et ne compare rien. `python -m appli mesurer --aide` donne toutes les
options.

Rien n'est enregistré par la commande : elle affiche son résultat, et `--sortie fichier.json` l'écrit aussi dans un
fichier, sans jamais écraser un fichier existant.

### Mesurer la vitesse selon le nombre d'experts actifs

Pour un modèle MoE, `llama-bench` ne sait pas régler le nombre d'experts actifs. La sous-commande `experts` passe par
`llama-server` : elle lance un serveur par valeur, avec `--override-kv <architecture>.expert_used_count=int:N` (la clé est
lue dans le fichier, le modèle n'est jamais écrit), mesure, puis l'arrête.

```powershell
python -m appli mesurer experts C:\modeles\mon-modele-moe.gguf --moteur C:\llama.cpp --experts 4,6,8,12,16
```

Le résultat est un tableau : pour chaque valeur, la vitesse de lecture du prompt (`pp`) et de génération (`tg`) avec leur
écart-type, comparée à la valeur du fichier quand elle fait partie de la liste. Chaque mesure se fait après un échauffement
(`--echauffement`, 1 par défaut) et se répète (`-r`, 5 par défaut). La commande vérifie avant d'afficher que seul le nombre
d'experts actifs diffère d'une mesure à l'autre.

Le serveur n'écoute que sur `127.0.0.1`, sur un port libre, avec une clé d'API tirée au hasard pour ce seul lancement et
jamais écrite. Les requêtes ne passent par aucun proxy, et le serveur est arrêté dans tous les cas, même sur une erreur.

## Où sont tes données

Tout est dans `%USERPROFILE%\.Aria-Etabli` :

- `reglages.json` : tes réglages ;
- `journal.json` : les 500 dernières opérations ;
- `sauvegardes\` : une copie datée avant chaque modification, et la version précédente de l'appli ;
- `secrets\` : ce que l'appli garde de secret, chiffré pour ton compte Windows (supprime la ligne si elle n'en garde pas) ;
- `jeux\` : les jeux de test téléchargés avec ton accord, créé au premier téléchargement ;
- `cache\` : fichiers temporaires de l'interface.

Supprimer ce dossier efface tes réglages et tes données : l'appli repart à neuf.

## Mises à jour

Au lancement, l'appli regarde s'il existe une nouvelle version : une seule
requête, vers la page des versions de ce dépôt sur GitHub, sans rien qui
t'identifie. Une case des réglages coupe cette recherche.

Rien ne se télécharge ni ne s'installe sans ton clic sur « Installer la mise à
jour ». L'appli télécharge alors le zip de la version, vérifie sa taille et son
empreinte, garde ta version actuelle dans `sauvegardes\versions\`, met la
nouvelle à sa place et se relance.

Pour revenir à la version précédente : quitte l'appli, puis copie les fichiers
de `sauvegardes\versions\<version>\` à la place de ceux de l'appli.

## Sécurité

- **Droits** : l'appli se lance sans droits administrateur.
- **Connexions sortantes** : `api.github.com` pour chercher une nouvelle version et, seulement avec `--accord`, pour
  télécharger un jeu de test ; `github.com` et son hôte de téléchargement pour le zip d'une mise à jour, après ton clic ;
  `127.0.0.1` pour parler au `llama-server` que `mesurer experts` lance, sans rien faire sortir de ta machine. Aucune
  autre, aucune télémétrie.
- **Secrets** : ce que l'appli garde, où, comment c'est protégé, et contre quoi
  cette protection ne joue pas.

Pour signaler une faille : voir [SECURITY.md](SECURITY.md).

## Limites à connaître

Inspecteur GGUF :

- il lit l'en-tête (métadonnées et description des tenseurs), jamais les poids ;
- il ne gère pas la version 1 du format ni les fichiers gros-boutistes (big-endian) ;
- un type de tenseur qu'il ne connaît pas (une quantification propre à un fork de llama.cpp, par exemple) s'affiche
  `inconnu(N)`, et sa taille, déduite des décalages, est marquée d'un tilde (`~`) : elle est exacte à l'alignement près ;
- l'exe n'a pas de console : lance la commande depuis les sources, ou depuis l'exe en redirigeant la sortie vers un
  fichier (`Aria-Etabli.exe inspecter modele.gguf > resume.txt`).

Mesure de vitesse :

- `llama-bench` ne sait pas régler le nombre d'experts actifs d'un modèle MoE : `mesurer bench` ne le fait pas, seule
  `mesurer experts` (par `llama-server`) le fait ;
- les vitesses de `bench` et de `experts` viennent de deux protocoles différents : elles ne se comparent pas entre elles ;
- `mesurer experts` ne mesure que la vitesse, pas la qualité : moins d'experts que prévu à l'entraînement va plus vite
  et répond moins bien, et plus d'experts n'est pas garanti meilleur ;
- le prompt de `mesurer experts` est une phrase répétée jusqu'à la taille demandée : le routage des experts d'un MoE
  dépend du texte, une vraie charge peut donner d'autres vitesses ;
- le moteur est à fournir avec `--moteur` : Etabli ne lit pas les dossiers des autres applis et n'embarque pas encore de moteur ;
- un modèle à quantification propre à un fork de llama.cpp (les Bonsai) ne se mesure qu'avec le moteur de ce fork ;
- l'occupation de la carte graphique n'est relevée qu'une fois avant la mesure, avec `nvidia-smi` s'il est présent : c'est un
  indice, pas une garantie ;
- ce sont des vitesses mesurées sur ta machine, avec ton moteur : elles ne valent pas pour une autre configuration.

Mises à jour : l'empreinte SHA-256 vérifie que le téléchargement est intact, pas
d'où il vient. La provenance repose sur le compte GitHub de l'auteur, sur les
releases immuables (une version publiée ne change plus) et sur l'attestation de
provenance que GitHub enregistre pour chaque zip (`gh attestation verify`, voir
les notes de version). Si le compte de l'auteur était compromis, une version
piégée pourrait quand même être proposée.

Projet personnel, sans audit de sécurité indépendant.

## Provenance des composants

Section à garder seulement si l'appli embarque un composant tiers. Le détail
est dans [LICENCES-TIERS.txt](LICENCES-TIERS.txt).

## Compiler soi-même

Prérequis de la stack, puis :

```powershell
.\build.ps1
```

Comment vérifier que l'exe obtenu est le bon.

## Structure

L'arborescence commentée, en dix lignes.

## Licence

MIT : voir [LICENSE](LICENSE). Les composants embarqués gardent leur propre
licence : voir [LICENCES-TIERS.txt](LICENCES-TIERS.txt).
