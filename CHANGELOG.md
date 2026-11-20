# Journal des versions

Un bloc par version, le plus récent en haut. Les notes complètes d'une version
sont dans `docs/releases/`.

## Non publié

### Ajouté

- Première version d'Aria Etabli.
- Inspecteur GGUF en ligne de commande (`python -m appli inspecter fichier.gguf`) : architecture, blocs, experts totaux
  et actifs, contexte, tokenizer, template de chat, quantification, tenseurs. Lecture seule, sans dépendance ; ouvre
  aussi les fichiers aux quantifications inconnues du paquet `gguf`, comme les Bonsai.
