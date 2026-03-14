###########
# Imports #
###########

import os
import argparse
import csv
import spacy
from pathlib import Path
from collections import defaultdict
 
##############################################################
# Correspondance between language and spacy pretrained model #
##############################################################
MODELS = {
    "fr":  "fr_core_news_sm",
    "oc":  "xx_ent_wiki_sm",   # no spaCy models for occitan
    "co":  "xx_ent_wiki_sm",   # no spaCy models for corsican
    "gsw": "xx_ent_wiki_sm",   # no spaCy models for alsacian
    "xx":  "xx_ent_wiki_sm",   # fallback
}

DEFAULT_MODEL = "xx_ent_wiki_sm"
 
 
def load_model(lang_code: str):
    """
    Loads the Spacy model corresponding to the given language code
    Uses multilingual model if the correct model is absent
    """ 
    model_name = MODELS.get(lang_code, DEFAULT_MODEL)
    try:
        nlp = spacy.load(model_name)
        print(f"  [OK] Modèle chargé : {model_name}")
    except OSError:
        print(f"  [!] Modèle '{model_name}' introuvable, bascule sur '{DEFAULT_MODEL}'")
        try:
            nlp = spacy.load(DEFAULT_MODEL)
        except OSError:
            raise SystemExit(
                f"\n[ERREUR] Aucun modèle spaCy disponible.\n"
                f"Installez-en un avec :\n"
                f"  python -m spacy download {DEFAULT_MODEL}\n"
                f"  python -m spacy download fr_core_news_sm\n"
            )

    if not nlp.has_pipe("sentencizer") and not nlp.has_pipe("senter") and not nlp.has_pipe("parser"):
        nlp.add_pipe("sentencizer")
        print(f"  [INFO] Sentencizer ajouté au pipeline (modèle sans détection de phrases)")
    
    return nlp
 
 
def detect_lang_from_path(pth: Path) -> str:
    known_codes = set(MODELS.keys())
    
    # Searches language code in the file name
    file_name = pth.stem.lower()
    file_parts = file_name.replace("-", "_").split("_")
    for part in file_parts:
        if part in known_codes:
            return part

    # If the name is not found in the file name, search in parent directory name
    for part in reversed(pth.parts[:-1]):  # exclut le fichier lui-même
        lower_part = part.lower()
        for code in known_codes:
            if code in lower_part.split("_") or lower_part == code:
                return code

    return "xx"
 
 
def analyse_file(pth: Path, nlp) -> dict:
    """
    Reads a txt file and passes it in spacy to check the number of tokens, sentences and lemmas (where possible)
    """
    try:
        text = pth.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fallback if encoding is not utf-8
        text = pth.read_text(encoding="latin-1")
 
    if not text.strip():
        return {
            "file": str(pth),
            "language": "—",
            "nb_tokens": 0,
            "nb_snts": 0,
            "nb_distinct_lemmas": 0,
            "status": "fichier vide",
        }
 
    # Partitioning if text has more than a million characters (Spacy limit)
    MAX_CHARS = 900_000
    if len(text) > MAX_CHARS:
        print(f"  [!] Texte tronqué à {MAX_CHARS} caractères pour {pth.name}")
        text = text[:MAX_CHARS]
 
    doc = nlp(text)
 
    nb_tokens = len([t for t in doc if not t.is_space])
    nb_snts = len(list(doc.sents))
    # Distinct lemmas, excluding punctuation
    lemmas = set(
        t.lemma_.lower()
        for t in doc
        if not t.is_punct and not t.is_space and t.lemma_.strip()
    )
    nb_lemmas = len(lemmas)
 
    language = detect_lang_from_path(pth)
 
    return {
        "file": str(pth),
        "language": language,
        "nb_tokens": nb_tokens,
        "nb_snts": nb_snts,
        "nb_distinct_lemmas": nb_lemmas,
        "status": "OK",
    }
 
 
def browse_dir(dir_path: Path) -> list[Path]:
    """
    Returns a list of every txt file in the directory and its sub-directories
    """
    return sorted(dir_path.rglob("*.txt"))
 
 
def show_results(results: list[dict]):
    """Affiche un tableau récapitulatif dans le terminal."""
    print("\n" + "=" * 80)
    print(f"{'FICHIER':<45} {'LANG':>4} {'TOKENS':>8} {'PHRASES':>8} {'LEMMES':>8}")
    print("=" * 80)
 
    totals = defaultdict(int)
    current_dir = None
    dir_totals = defaultdict(int)
 
    for r in results:
        directory = str(Path(r["file"]).parent)

        # If we change directory we show the last directory's totals
        if current_dir is not None and directory != current_dir:
            print(f"  {'→ Sous-total : ' + Path(current_dir).name:<43} {'':>4} "
                  f"{dir_totals['tokens']:>8} {dir_totals['snts']:>8} "
                  f"{dir_totals['lemmas']:>8}")
            print("-" * 80)
            dir_totals = defaultdict(int)

        current_dir = directory

        name = Path(r["file"]).name
        if len(name) > 44:
            name = "…" + name[-43:]
        print(
            f"{name:<45} {r['language']:>4} "
            f"{r['nb_tokens']:>8} {r['nb_snts']:>8} {r['nb_distinct_lemmas']:>8}"
        )
        dir_totals["tokens"]  += r["nb_tokens"]
        dir_totals["snts"] += r["nb_snts"]
        dir_totals["lemmas"]  += r["nb_distinct_lemmas"]
        totals["tokens"] += r["nb_tokens"]
        totals["snts"] += r["nb_snts"]
        totals["lemmas"] += r["nb_distinct_lemmas"]

        # Sous-total du dernier dossier
    if current_dir is not None:
        print(f"  {'→ Sous-total : ' + Path(current_dir).name:<43} {'':>4} "
              f"{dir_totals['tokens']:>8} {dir_totals['snts']:>8} "
              f"{dir_totals['lemmas']:>8}")
 
    print("=" * 80)
    print(
        f"{'TOTAL (' + str(len(results)) + ' fichiers)':<45} {'':>4} "
        f"{totals['tokens']:>8} {totals['snts']:>8} {totals['lemmas']:>8}"
    )
    print("=" * 80)
    print("⚠  Le total des lemmes est une somme par fichier (pas de déduplication globale).")
 
 
def export_csv(results: list[dict], output: Path):
    """Exports results in a CSV file"""
    colonnes = ["file", "language", "nb_tokens", "nb_snts", "nb_distinct_lemmas", "status"]
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=colonnes)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[CSV] Résultats exportés dans : {output}")
 
 
def export_table(results: list[dict], output: Path):
    """
    Exports a table of global statistics for the visited files
    """
    language_stats = defaultdict(lambda: {"tokens": 0, "snts": 0, "lemmas": 0, "files": 0})
    dir_stats = defaultdict(lambda: {"tokens": 0, "snts": 0, "lemmas": 0, "files": 0})
 
    for r in results:
        l = r["language"]
        d = str(Path(r["file"]).parent)
        language_stats[l]["tokens"] += r["nb_tokens"]
        language_stats[l]["snts"] += r["nb_snts"]
        language_stats[l]["lemmas"] += r["nb_distinct_lemmas"]
        language_stats[l]["files"] += 1
        dir_stats[d]["tokens"]   += r["nb_tokens"]
        dir_stats[d]["snts"]  += r["nb_snts"]
        dir_stats[d]["lemmas"]   += r["nb_distinct_lemmas"]
        dir_stats[d]["files"] += 1
    
    lines = []
    # Global statistics
    lines.append("RAPPORT D'ANALYSE DU CORPUS" + "\n" + ("=" * 50))
    lines.append(f"Nombre total de fichiers analysés : {len(results)}" + "\n\n" + "STATISTIQUES PAR LANGUE")
    lines.append("-" * 50)
    lines.append(f"{'Langue':<10} {'Fichiers':>8} {'Tokens':>10} {'Phrases':>10} {'Lemmes':>10}")
    lines.append("-" * 50)
 
    total_t, total_p, total_l, total_f = 0, 0, 0, 0
    for langue, s in sorted(language_stats.items()):
        lines.append(
            f"{langue:<10} {s['files']:>8} {s['tokens']:>10} "
            f"{s['snts']:>10} {s['lemmas']:>10}"
        )
        total_t += s["tokens"]
        total_p += s["snts"]
        total_l += s["lemmas"]
        total_f += s["files"]
    lines.append("-" * 50)
    lines.append(f"{'TOTAL':<10} {total_f:>8} {total_t:>10} {total_p:>10} {total_l:>10}")

    # Directory statistics
    lines.append("")
    lines.append("STATISTIQUES PAR DOSSIER")
    lines.append("-" * 50)
    lines.append(f"{'Dossier':<30} {'Fichiers':>8} {'Tokens':>10} {'Phrases':>10} {'Lemmes':>10}")
    lines.append("-" * 50)
    for dir_path, s in sorted(dir_stats.items()):
        dir_name = Path(dir_path).name
        if len(dir_name) > 29:
            dir_name = "…" + dir_name[-28:]
        lines.append(
            f"{dir_name:<30} {s['files']:>8} {s['tokens']:>10} "
            f"{s['snts']:>10} {s['lemmas']:>10}"
        )
    lines.append("-" * 50)
    lines.append(f"{'TOTAL':<30} {total_f:>8} {total_t:>10} {total_p:>10} {total_l:>10}")

    # File statistics
    lines.append("")
    lines.append("DÉTAIL PAR FICHIER")
    lines.append("-" * 50)
    for r in results:
        lines.append(
            f"{Path(r['file']).name} | langue: {r['language']} | "
            f"tokens: {r['nb_tokens']} | phrases: {r['nb_snts']} | "
            f"lemmes: {r['nb_distinct_lemmas']} | statut: {r['status']}"
        )
 
    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[TXT] Rapport exporté dans : {output}")

########
# MAIN #
########
def main():
    parser = argparse.ArgumentParser(
        description="Analyse de corpus avec spaCy : tokens, phrases, lemmes."
    )
    parser.add_argument(
        "--dossier", "-d",
        default="data/paralleles/",
        help="pth vers le dossier racine contenant les fichiers .txt"
    )
    parser.add_argument(
        "--langue", "-l",
        default=None,
        help=(
            "Force une langue pour tous les fichiers (ex: fr, oc, co, gsw, xx). "
            "Si absent, la langue est déduite depuis les noms de dossiers."
        )
    )
    parser.add_argument(
        "--sortie", "-s",
        default="resultats_corpus",
        help="Préfixe des fichiers de sortie (défaut: resultats_corpus)"
    )
    args = parser.parse_args()
 
    dir_path = Path(args.dossier)
    if not dir_path.is_dir():
        raise SystemExit(f"[ERREUR] Le dossier '{dir_path}' n'existe pas.")
 
    files = browse_dir(dir_path)
    if not files:
        raise SystemExit(f"[ERREUR] Aucun fichier .txt trouvé dans '{dir_path}'.")
 
    print(f"\n{len(files)} fichier(s) .txt trouvé(s) dans '{dir_path}'\n")
 
    # Loads required models
    MODELS_charges = {}
    results = []
 
    for pth in files:
        code = args.langue if args.langue else detect_lang_from_path(pth)
        print(f"Traitement : {pth.relative_to(dir_path)}")
 
        if code not in MODELS_charges:
            MODELS_charges[code] = load_model(code)
 
        nlp = MODELS_charges[code]
        res = analyse_file(pth, nlp)
        res["language"] = code  # If --langue arg is forced then overwrite language in results
        results.append(res)
 
    show_results(results)
 
    # Exports results
    output = Path(".")
    export_csv(results, output / f"{args.sortie}.csv")
    export_table(results, output / f"{args.sortie}.txt")
 
 
if __name__ == "__main__":
    main()
 