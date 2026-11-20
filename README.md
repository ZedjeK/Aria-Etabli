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

## Où sont tes données

Tout est dans `%USERPROFILE%\.Aria-Etabli` :

- `reglages.json` : tes réglages ;
- `journal.json` : les 500 dernières opérations ;
- `sauvegardes\` : une copie datée avant chaque modification, et la version précédente de l'appli ;
- `secrets\` : ce que l'appli garde de secret, chiffré pour ton compte Windows (supprime la ligne si elle n'en garde pas) ;
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
- **Connexions sortantes** : la liste complète, destination par destination.
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
