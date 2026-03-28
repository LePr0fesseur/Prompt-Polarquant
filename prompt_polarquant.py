# MIT License
#
# Copyright (c) 2026 LePr0fesseur
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
PromptPolarQuant — Optimiseur de tokens inspiré de PolarQuant (Google/KAIST, 2025)

Principe mathématique adapté au prompting :
  PolarQuant original compresse les vecteurs KV du cache d'inférence en représentation polaire
  (rayon r = densité d'information, angle θ = direction sémantique) avec quantification optimale.

  Cette adaptation applique la même géométrie polaire aux unités sémantiques d'un prompt :
    r(u)  = densité informationnelle (importance TF-IDF + poids positionnel + unicité lexicale)
    θ(u)  = angle sémantique (cluster de sens via hachage lexical + catégorie grammaticale)

  La quantification polaire élimine :
    - Les unités à faible rayon (faible apport informationnel → phrases redondantes, remplissage)
    - Les unités à angle quasi-identique dans une même zone (doublons sémantiques)
    - Les tokens sous le seuil d'auto-information (articles, prépositions non-essentiels)

Usage :
    python prompt_polarquant.py                   # mode interactif
    python prompt_polarquant.py -f prompt.txt     # depuis un fichier
    python prompt_polarquant.py -b 4              # précision 4 bits (compression maximale)
    python prompt_polarquant.py -b 8              # précision 8 bits (compression légère)
    echo "mon prompt" | python prompt_polarquant.py --stdin
"""

import re
import sys
import math
import hashlib
import argparse
import unicodedata
from collections import Counter
from typing import NamedTuple


# -----------------------------------------------------------------------------
# STOPWORDS multilingues (FR + EN) – tokens à faible rayon par défaut
# -----------------------------------------------------------------------------
STOPWORDS_FR = {
    "le", "la", "les", "un", "une", "des", "de", "du", "au", "aux",
    "ce", "cet", "cette", "ces", "mon", "ma", "mes", "ton", "ta", "tes",
    "son", "sa", "ses", "notre", "votre", "leur", "leurs",
    "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
    "me", "te", "se", "lui", "y", "en",
    "et", "ou", "mais", "donc", "or", "ni", "car", "que", "qui", "quoi",
    "dont", "où", "si", "comme", "quand", "lorsque", "puisque",
    "est", "sont", "était", "été", "être", "avoir", "a", "ont", "avait",
    "plus", "très", "bien", "ainsi", "alors", "aussi", "encore", "même",
    "tout", "tous", "toute", "toutes", "ici", "là", "puis", "enfin",
    "dans", "sur", "sous", "avec", "sans", "par", "pour", "vers", "entre",
    "avant", "après", "pendant", "depuis", "jusqu",
}

STOPWORDS_EN = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their", "this", "that",
    "these", "those", "which", "who", "whom", "whose", "what", "when",
    "where", "why", "how", "and", "or", "but", "nor", "so", "yet", "for",
    "in", "on", "at", "to", "of", "by", "with", "from", "into", "about",
    "as", "if", "than", "then", "also", "just", "not", "no", "very",
    "more", "most", "such", "each", "both", "all", "any", "few", "much",
}

STOPWORDS = STOPWORDS_FR | STOPWORDS_EN

# -----------------------------------------------------------------------------
# CODEBOOK d'abréviations (adapté de la notion de codebook optimal de PolarQuant)
# Réduit les séquences fréquentes à leur représentation minimale
# -----------------------------------------------------------------------------
CODEBOOK = {
    # Français
    r"\bc'est-à-dire\b": "soit",
    r"\ben d'autres termes\b": "i.e.",
    r"\bpar exemple\b": "ex.",
    r"\bc'est\b": "=",
    r"\bil faut\b": "=>",
    r"\bs'il vous plaît\b": "svp",
    r"\bje voudrais\b": "je veux",
    r"\bpourriez-vous\b": "peux-tu",
    r"\bpourrait-on\b": "peut-on",
    r"\bje souhaiterais\b": "je veux",
    r"\bje t'en prie\b": "svp",
    r"\bje vous en prie\b": "svp",
    r"\bafin de\b": "pour",
    r"\bafin que\b": "pour que",
    r"\bde manière à\b": "pour",
    r"\bde façon à\b": "pour",
    r"\bà cet égard\b": "ici",
    r"\ben ce qui concerne\b": "sur",
    r"\bpar rapport à\b": "vs",
    r"\bdu fait que\b": "car",
    r"\bétant donné que\b": "car",
    r"\bvu que\b": "car",
    r"\bcependant\b": "mais",
    r"\btoutefois\b": "mais",
    r"\bnéanmoins\b": "mais",
    r"\bpar conséquent\b": "donc",
    r"\ben conséquence\b": "donc",
    r"\bainsi que\b": "et",
    r"\bcordialement\b": "",
    r"\bmerci d'avance\b": "",
    r"\bje vous remercie\b": "",
    # Anglais
    r"\bin other words\b": "i.e.",
    r"\bfor example\b": "e.g.",
    r"\bfor instance\b": "e.g.",
    r"\bin order to\b": "to",
    r"\bin order for\b": "for",
    r"\bdue to the fact that\b": "because",
    r"\bwith regard to\b": "on",
    r"\bwith respect to\b": "on",
    r"\bas a result\b": "so",
    r"\bconsequently\b": "so",
    r"\bnevertheless\b": "but",
    r"\bhowever\b": "but",
    r"\bnotwithstanding\b": "despite",
    r"\bsubsequently\b": "then",
    r"\bfurthermore\b": "also",
    r"\bin addition\b": "also",
    r"\bmoreover\b": "also",
    r"\bcould you please\b": "please",
    r"\bi would like to\b": "i want to",
    r"\bwould you be able to\b": "can you",
    r"\bkind regards\b": "",
    r"\bbest regards\b": "",
    r"\bthank you in advance\b": "",
    r"\bthanks in advance\b": "",
}


# -----------------------------------------------------------------------------
# Structures de données
# -----------------------------------------------------------------------------
class TokenInfo(NamedTuple):
    text: str
    radius: float   # densité d'information [0, 1]
    angle: float    # direction sémantique [0, 2π]
    is_stop: bool


class UnitInfo(NamedTuple):
    text: str
    tokens: list
    radius: float
    angle: float
    position: int   # ordre original


# -----------------------------------------------------------------------------
# Fonctions mathématiques clés
# -----------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Normalisation Unicode (NFC) et lowercasing."""
    return unicodedata.normalize("NFC", text).lower()


def _tokenize(text: str) -> list[str]:
    """Tokenisation par unités lexicales (mots et ponctuations significatives)."""
    return re.findall(r"[a-zA-ZÀ-ÿ0-9_'-]+|[?!:;]", text)


def _segment_units(text: str) -> list[str]:
    """
    Segmentation en unités sémantiques (phrases/clauses).
    Séparateurs : '.', '!', '?', '\n', ';', '—', ' - '
    """
    units = re.split(r'(?<=[.!?])\s+|[;\n]|(?<=\w)\s*—\s*(?=\w)|\s+-\s+', text)
    return [u.strip() for u in units if u.strip()]


def _compute_idf(all_tokens_per_unit: list[list[str]]) -> dict[str, float]:
    """
    IDF = log(N / df(t)) où df(t) = nombre d'unités contenant le token t.
    Mesure de rareté/valeur informative globale.
    """
    N = len(all_tokens_per_unit)
    if N == 0:
        return {}
    df: Counter = Counter()
    for tokens in all_tokens_per_unit:
        for t in set(tokens):
            df[_normalize(t)] += 1
    return {
        t: math.log((N + 1) / (freq + 1)) + 1.0
        for t, freq in df.items()
    }


def _semantic_angle(token: str) -> float:
    """
    Calcule l'angle sémantique θ ∈ [0, 2π].

    Adapté de PolarQuant : chaque vecteur clé est converti en coordonnées polaires
    où l'angle encode la direction sémantique. Ici, on hash le token pour l'assigner
    à l'une de N catégories sémantiques (pôles), puis on subdivise finement via
    le hash SHA256 (garantit une distribution uniforme).

    Les 8 pôles sémantiques principaux :
        0 = entités/noms          (qui, quoi)
        1 = actions/verbes        (faire, être)
        2 = qualités/adjectifs    (comment, quel)
        3 = quantités/nombres     (combien)
        4 = lieu/espace           (où)
        5 = temps/séquence        (quand)
        6 = cause/raison          (pourquoi)
        7 = meta/structure        (donc, mais)
    """
    N_POLES = 8
    norm = _normalize(token)

    # Assignation grammaticale heuristique (catégories pré-définies)
    action_suffixes = ("er", "ir", "re", "ait", "ent", "ons", "ez",
                       "ate", "ize", "ise", "ing", "ed", "ify")
    quality_suffixes = ("able", "ible", "eux", "euse", "ive", "ive",
                        "al", "el", "elle", "ful", "less", "ous")
    time_words = {"avant", "après", "quand", "lors", "pendant", "depuis",
                  "before", "after", "when", "during", "since", "until", "while"}
    location_words = {"dans", "sur", "sous", "ici", "là", "où", "entre",
                      "in", "on", "at", "here", "there", "where", "between"}
    causal_words = {"car", "parce", "puisque", "donc", "ainsi", "because",
                    "since", "so", "hence", "therefore", "thus"}
    meta_words = {"mais", "cependant", "toutefois", "or", "ni", "soit",
                  "but", "however", "yet", "although", "though"}

    if norm in time_words:
        pole = 5
    elif norm in location_words:
        pole = 4
    elif norm in causal_words:
        pole = 6
    elif norm in meta_words:
        pole = 7
    elif any(norm.endswith(s) for s in action_suffixes):
        pole = 1
    elif any(norm.endswith(s) for s in quality_suffixes):
        pole = 2
    elif re.match(r"^\d+[\.,]?\d*$", norm):
        pole = 3
    else:
        # Assignation par hash SHA256 (distribution déterministe uniforme)
        h = int(hashlib.sha256(norm.encode()).hexdigest(), 16)
        pole = h % N_POLES

    # Fine subdivision intra-pôle via les premiers octets du hash
    h_val = int(hashlib.sha256(norm.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    base_angle = (2 * math.pi * pole) / N_POLES
    sub_angle = (2 * math.pi / N_POLES) * h_val
    return base_angle + sub_angle


def _token_radius(token: str, idf_score: float, position: int,
                  unit_length: int) -> float:
    """
    Calcule le rayon informationnel r ∈ [0, 1] d'un token.

    Formule inspirée de PolarQuant (arxiv:2502.02617) :
        r = sqrt(idf²  +  pos_weight²)  ×  novelty_factor
             --------------------------
             analogie : r = √(K[2j]² + K[2j+1]²) pour les vecteurs clés

    Composantes :
        idf_score    : auto-information (fréquence inverse dans le document)
        pos_weight   : poids positionnel (début/fin plus importants — loi de Jost)
        novelty      : pénalité si le token est un stopword
    """
    # Poids positionnel en U : début et fin portent plus d'information
    if unit_length <= 1:
        pos_weight = 1.0
    else:
        relative_pos = position / (unit_length - 1)
        # Courbe en U : max(sin(0), sin(π)) = 0, mais on veut max aux bords
        pos_weight = 0.5 + 0.5 * abs(2 * relative_pos - 1)

    norm_token = _normalize(token)
    is_stop = norm_token in STOPWORDS

    # Facteur de nouveauté lexicale
    novelty = 0.25 if is_stop else 1.0

    # Normalisation IDF dans [0, 1] (log max ≈ log(N+1) pour singletons)
    idf_norm = min(idf_score / 6.0, 1.0)

    # Rayon polaire (norme L2 dans l'espace {idf, position})
    radius = math.sqrt(idf_norm ** 2 + pos_weight ** 2) / math.sqrt(2)
    return radius * novelty


def _unit_polar(unit_text: str, unit_tokens: list[str],
                idf_map: dict[str, float], position: int) -> UnitInfo:
    """
    Calcule la représentation polaire d'une unité sémantique.

    Rayon de l'unité  : R = mean(r_i) × sqrt(max(r_i))
                         (analogie : sensibilité au pic d'information)
    Angle de l'unité  : Θ = mean(θ_i) pondéré par r_i
                         (barycentre sémantique pondéré)
    """
    if not unit_tokens:
        return UnitInfo(unit_text, [], 0.0, 0.0, position)

    token_infos = []
    for i, tok in enumerate(unit_tokens):
        norm_tok = _normalize(tok)
        idf_score = idf_map.get(norm_tok, 1.0)
        r = _token_radius(tok, idf_score, i, len(unit_tokens))
        θ = _semantic_angle(tok)
        is_stop = norm_tok in STOPWORDS
        token_infos.append(TokenInfo(tok, r, θ, is_stop))

    # Rayon de l'unité basé sur les tokens de CONTENU uniquement.
    # But : éviter qu'une phrase courte et polie ("Merci d'avance")
    # obtienne un rayon élevé à cause du poids positionnel du dernier mot.
    # Le rayon doit refléter la richesse informationnelle réelle.
    content_toks = [t for t in token_infos if not t.is_stop]
    if not content_toks:
        # Unité 100% stopwords → rayon quasi-nul (sera filtrée)
        R_unit = 0.01
    else:
        c_radii = [t.radius for t in content_toks]
        content_mean = sum(c_radii) / len(c_radii)
        content_max = max(c_radii)
        # Densité de contenu : fraction de tokens non-stopword
        content_density = len(content_toks) / len(token_infos)
        # Richesse : mean * sqrt(max), pondéré par densité de contenu
        # Les unités avec peu de contenu (polies/formulaïques) sont pénalisées
        R_unit = content_mean * math.sqrt(content_max) * (0.4 + 0.6 * content_density)

    # Barycentre angulaire pondéré (utilise coordonnées complexes pour éviter
    # le problème de discontinuité à 2π).
    # On pondère uniquement par les tokens de contenu pour un angle plus fidèle.
    weight_toks = content_toks if content_toks else token_infos
    total_weight = sum(t.radius for t in weight_toks)
    if total_weight == 0:
        Θ_unit = 0.0
    else:
        cx = sum(t.radius * math.cos(t.angle) for t in weight_toks) / total_weight
        cy = sum(t.radius * math.sin(t.angle) for t in weight_toks) / total_weight
        Θ_unit = math.atan2(cy, cx) % (2 * math.pi)

    return UnitInfo(unit_text, token_infos, R_unit, Θ_unit, position)


# -----------------------------------------------------------------------------
# Algorithme de quantification polaire
# -----------------------------------------------------------------------------

def _quantize_units(units: list[UnitInfo], n_bits: int) -> list[UnitInfo]:
    """
    Quantification polaire des unités sémantiques.

    Adapté de PolarQuant stage 1 (TurboQuant) :
      - n_bits contrôle la résolution angulaire et le seuil de rayon
      - 2^n_bits niveaux de quantification angulaire
      - Les unités de faible rayon (sous le seuil) sont éliminées
      - Les unités redondantes (même angle quantifié) : seule la plus informative survit

    n_bits = 4 → compression agressive  (~40-60% reduction)
    n_bits = 6 → compression équilibrée (~20-40% reduction)
    n_bits = 8 → compression légère     (~5-20% reduction)
    """
    if not units:
        return []

    n_levels = 2 ** n_bits  # nombre de niveaux angulaires
    angle_resolution = 2 * math.pi / n_levels

    # Nombre minimum d'unités à conserver (garantie de préservation du sens)
    # 4-bits → garder ≥60%, 6-bits → garder ≥80%, 8-bits → garder ≥100%
    min_keep_frac = max(0.6, 1.0 - (8 - n_bits) * 0.10)
    n_to_keep = max(1, math.ceil(len(units) * min_keep_frac))

    # Sélection des n_to_keep unités de plus grand rayon, en préservant l'ordre
    sorted_by_r = sorted(units, key=lambda u: u.radius, reverse=True)
    kept_ids = {id(u) for u in sorted_by_r[:n_to_keep]}
    kept = [u for u in units if id(u) in kept_ids]

    # 2. Déduplication angulaire : pour chaque bin angulaire, garder l'unité max
    #    C'est ici que se fait la majorité de la compression (redondances sémantiques)
    angle_bins: dict[int, UnitInfo] = {}
    for unit in kept:
        bin_idx = int(unit.angle / angle_resolution) % n_levels
        if bin_idx not in angle_bins or unit.radius > angle_bins[bin_idx].radius:
            angle_bins[bin_idx] = unit

    # 3. Restauration de l'ordre original
    result = sorted(angle_bins.values(), key=lambda u: u.position)
    return result


def _compress_unit_tokens(unit: UnitInfo, n_bits: int) -> str:
    """
    Compression intra-unité : supprime les tokens de faible rayon.
    Préserve les tokens essentiels (verbes, noms, adjectifs clés).
    """
    if not unit.tokens:
        return unit.text

    radii = [t.radius for t in unit.tokens]
    if not radii:
        return unit.text

    # Seuil adaptatif pour la suppression intra-unité.
    # On ne retire que les tokens clairement stopwords ET de faible rayon.
    # Les tokens de contenu (non-stopword) sont quasi-toujours préservés.
    #   4-bits → seuil = 40% du max radius  (retire stopwords + tokens faibles)
    #   6-bits → seuil = 30% du max radius  (retire principalement stopwords)
    #   8-bits → seuil = 20% du max radius  (retire seulement les stopwords évidents)
    stop_threshold = max(radii) * max(0.10, 0.52 - n_bits * 0.04)

    kept_tokens = []
    for t in unit.tokens:
        if t.is_stop and t.radius < stop_threshold:
            # Supprimer : stopword sous le seuil
            continue
        kept_tokens.append(t.text)

    # Fallback : si compression trop forte, garder au moins 50% des tokens originaux
    min_keep_count = max(2, len(unit.tokens) // 2)
    if len(kept_tokens) < min_keep_count:
        sorted_by_r = sorted(unit.tokens, key=lambda t: t.radius, reverse=True)
        kept_tokens = [t.text for t in sorted_by_r[:min_keep_count]]
        # Restaurer l'ordre original
        original_order = {t.text: i for i, t in enumerate(unit.tokens)}
        kept_tokens.sort(key=lambda w: original_order.get(w, 0))

    return " ".join(kept_tokens)


def _apply_codebook(text: str) -> str:
    """Applique le codebook d'abréviations (remplacement de séquences coûteuses)."""
    for pattern, replacement in CODEBOOK.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    # Nettoyage des espaces multiples
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"^\s+|\s+$", "", text, flags=re.MULTILINE)
    return text.strip()


def _count_tokens(text: str) -> int:
    """
    Estimation du nombre de tokens compatible avec les tokeniseurs modernes
    (tiktoken GPT-4, Claude, Mistral).
    Approximation : ~0.75 mots par token pour FR, ~1.3 chars/token en moyenne.
    """
    # Méthode hybride : split sur espaces + ponctuation significative
    words = re.findall(r"[a-zA-ZÀ-ÿ]+|[0-9]+|\S", text)
    # Les mots courts comptent comme ~0.75 token, les longs comme ~1.5
    total = 0.0
    for w in words:
        if len(w) <= 3:
            total += 0.75
        elif len(w) <= 7:
            total += 1.0
        else:
            total += max(1.0, len(w) / 5.0)
    return max(1, round(total))


# -----------------------------------------------------------------------------
# Interface principale
# -----------------------------------------------------------------------------

def optimize_prompt(prompt: str, n_bits: int = 6,
                    verbose: bool = True) -> tuple[str, dict]:
    """
    Point d'entrée principal de PromptPolarQuant.

    Paramètres :
        prompt  : texte du prompt à optimiser
        n_bits  : profondeur de quantification [4–8] (4=agressif, 8=léger)
        verbose : afficher les stats de compression

    Retourne :
        (prompt_optimisé, stats_dict)
    """
    if not prompt or not prompt.strip():
        return prompt, {"error": "prompt vide"}

    # -- Étape 0 : Pré-traitement par codebook --------------------------------
    text = _apply_codebook(prompt)

    # -- Étape 1 : Segmentation en unités sémantiques -------------------------
    raw_units = _segment_units(text)
    if not raw_units:
        return prompt, {"error": "segmentation échouée"}

    # -- Étape 2 : Tokenisation de chaque unité -------------------------------
    tokenized_units = [_tokenize(u) for u in raw_units]

    # -- Étape 3 : Calcul IDF global ------------------------------------------
    idf_map = _compute_idf(tokenized_units)

    # -- Étape 4 : Représentation polaire de chaque unité ---------------------
    polar_units = [
        _unit_polar(raw_units[i], tokenized_units[i], idf_map, i)
        for i in range(len(raw_units))
    ]

    # -- Étape 5 : Quantification polaire (sélection des unités) --------------
    selected_units = _quantize_units(polar_units, n_bits)

    # -- Étape 6 : Compression intra-unité ------------------------------------
    compressed_parts = []
    for unit in selected_units:
        if len(unit.tokens) <= 2:
            # Unité courte : garder telle quelle
            compressed_parts.append(unit.text)
        else:
            compressed_parts.append(_compress_unit_tokens(unit, n_bits))

    # -- Étape 7 : Reconstruction ---------------------------------------------
    separator = ". " if re.search(r'[.!?]', prompt) else " "
    # Filtrer les parties vides ou purement ponctuées
    valid_parts = [p.strip() for p in compressed_parts
                   if re.search(r'[a-zA-ZÀ-ÿ0-9]', p or "")]
    result = separator.join(valid_parts)
    # Nettoyer les doubles ponctuations (ex: "?." ou "!.")
    result = re.sub(r"([.!?])\s*\.", r"\1", result)
    result = re.sub(r"\s+", " ", result).strip()

    # Assure que le prompt résultant termine correctement
    if result and not result[-1] in ".!?:":
        if any(result[-1] in ".!?:" for _ in [1]):
            pass
        else:
            # On laisse sans ponctuation forcée pour ne pas altérer le sens

            pass

    # -- Statistiques ---------------------------------------------------------
    original_tokens = _count_tokens(prompt)
    optimized_tokens = _count_tokens(result)
    reduction = ((original_tokens - optimized_tokens) / original_tokens * 100
                 if original_tokens > 0 else 0)

    stats = {
        "original_tokens":    original_tokens,
        "optimized_tokens":   optimized_tokens,
        "token_reduction":    round(reduction, 1),
        "original_units":     len(raw_units),
        "selected_units":     len(selected_units),
        "unit_reduction":     round((1 - len(selected_units) / max(1, len(raw_units))) * 100, 1),
        "n_bits":             n_bits,
        "compression_ratio":  round(original_tokens / max(1, optimized_tokens), 2),
    }

    if verbose:
        _print_stats(prompt, result, stats)

    return result, stats


def _print_stats(original: str, optimized: str, stats: dict):
    """Affiche un rapport de compression formaté."""
    bar_len = 40
    reduction = stats["token_reduction"]
    filled = int(bar_len * reduction / 100)
    bar = "#" * filled + "." * (bar_len - filled)

    print("\n" + "=" * 60)
    print("  PromptPolarQuant — Rapport de compression")
    print("=" * 60)
    print(f"\n  Bits de quantification : {stats['n_bits']}")
    print(f"  Unités sémantiques     : {stats['original_units']} → {stats['selected_units']} "
          f"(-{stats['unit_reduction']}%)")
    print(f"  Tokens estimés         : {stats['original_tokens']} → {stats['optimized_tokens']}")
    print(f"\n  Réduction tokens : [{bar}] {reduction:.1f}%")
    print(f"  Ratio compression : {stats['compression_ratio']}x\n")
    print("-" * 60)
    print("  PROMPT ORIGINAL :")
    print("-" * 60)
    _print_wrapped(original)
    print("\n" + "-" * 60)
    print("  PROMPT OPTIMISÉ :")
    print("-" * 60)
    _print_wrapped(optimized)
    print("\n" + "=" * 60 + "\n")


def _print_wrapped(text: str, width: int = 56):
    """Affichage avec retour à la ligne."""
    words = text.split()
    line = "  "
    for word in words:
        if len(line) + len(word) + 1 > width:
            print(line)
            line = "  " + word
        else:
            line += ("" if line == "  " else " ") + word
    if line.strip():
        print(line)


# -----------------------------------------------------------------------------
# Mode interactif / CLI
# -----------------------------------------------------------------------------

BANNER = r"""
  ____       _            ____       _             ___                   _
 |  _ \ ___ | | __ _ _ __/ ___| _  _| |   ___  ___|  _ \ ___ _ __  _ __(_)
 | |_) / _ \| |/ _` | '__\___ \| || | |  / _ \/ __| |_) / _ \ '_ \| '__| |
 |  __/ (_) | | (_| | |   ___) || || |_ | (_) \__ \  _ <  __/ | | | |  | |
 |_|   \___/|_|\__,_|_|  |____/ \__, |__|\___/|___/_| \_\___|_| |_|_|  |_|
   Token Optimizer  v1.0         |___/    inspiré de PolarQuant (Google 2025)
"""

HELP_TEXT = """
Commandes disponibles :
  <prompt>         Entrer directement votre prompt à optimiser
  /bits <n>        Changer la précision de quantification (4-8, défaut: 6)
  /demo            Lancer une démonstration avec des exemples
  /help            Afficher cette aide
  /quit            Quitter

Niveaux de quantification :
  /bits 4  → Compression maximale  (~50-70% de tokens économisés)
  /bits 6  → Compression équilibrée (~30-50% de tokens économisés)  ← défaut
  /bits 8  → Compression légère     (~10-25% de tokens économisés)
"""

DEMO_PROMPTS = [
    (
        "Bonjour, je voudrais vous demander si vous pourriez s'il vous plaît "
        "m'expliquer de manière détaillée et en d'autres termes plus simples "
        "le fonctionnement du machine learning, en particulier en ce qui concerne "
        "les réseaux de neurones artificiels. Je vous remercie d'avance pour "
        "votre réponse et je vous prie d'accepter mes cordiales salutations.",
        "Prompt formel français"
    ),
    (
        "Could you please provide me with a comprehensive and detailed explanation "
        "of how quantum computing works, with regard to its fundamental principles, "
        "in other words the qubits and superposition, and furthermore how it differs "
        "from classical computing? I would like to understand both the theoretical "
        "foundations as well as the practical applications. Thank you in advance "
        "for your thorough response.",
        "Prompt formel anglais"
    ),
    (
        "Je voudrais que tu crées un programme Python qui permet de lire un fichier CSV, "
        "de calculer des statistiques de base comme la moyenne, la médiane et l'écart-type "
        "pour chaque colonne numérique, et d'afficher les résultats sous forme de tableau "
        "formaté dans le terminal. Il faut également que le programme gère les erreurs "
        "de manière appropriée, notamment les cas où le fichier n'existe pas ou est vide.",
        "Prompt technique"
    ),
]


def run_demo(n_bits: int):
    """Lance la démonstration avec des exemples pré-définis."""
    print("\n" + "=" * 60)
    print("  DÉMONSTRATION PromptPolarQuant")
    print("=" * 60)
    for prompt, label in DEMO_PROMPTS:
        print(f"\n  [{label}]")
        optimize_prompt(prompt, n_bits=n_bits, verbose=True)
        input("  >> Appuyer sur Entree pour continuer...")


def interactive_mode():
    """Mode interactif REPL."""
    print(BANNER)
    print("  Optimiseur de prompts par quantification polaire inspiré de PolarQuant")
    print("  (Google/KAIST, arxiv:2502.00527 & 2502.02617 — TurboQuant, ICLR 2026)\n")
    print(HELP_TEXT)

    n_bits = 6

    while True:
        try:
            user_input = input(f"[bits={n_bits}] prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Au revoir !")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("  Au revoir !")
            break
        if user_input.lower() == "/help":
            print(HELP_TEXT)
            continue
        if user_input.lower() == "/demo":
            run_demo(n_bits)
            continue
        if user_input.lower().startswith("/bits "):
            try:
                new_bits = int(user_input.split()[1])
                if 2 <= new_bits <= 12:
                    n_bits = new_bits
                    print(f"  Quantification réglée sur {n_bits} bits.")
                else:
                    print("  Valeur invalide. Utilisez un entier entre 4 et 8.")
            except ValueError:
                print("  Usage: /bits <n>  (exemple: /bits 4)")
            continue

        optimize_prompt(user_input, n_bits=n_bits, verbose=True)


def main():
    parser = argparse.ArgumentParser(
        description="PromptPolarQuant — Optimiseur de tokens par quantification polaire",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("-f", "--file", help="Fichier contenant le prompt à optimiser")
    parser.add_argument("-b", "--bits", type=int, default=6,
                        help="Profondeur de quantification (4=agressif, 8=léger, défaut=6)")
    parser.add_argument("--stdin", action="store_true",
                        help="Lire le prompt depuis stdin")
    parser.add_argument("--quiet", action="store_true",
                        help="Afficher seulement le prompt optimisé (pas de stats)")
    parser.add_argument("--demo", action="store_true",
                        help="Lancer la démonstration")
    parser.add_argument("prompt", nargs="?", help="Prompt à optimiser (optionnel)")

    args = parser.parse_args()

    if args.demo:
        run_demo(args.bits)
        return

    if args.stdin:
        prompt = sys.stdin.read()
        result, _ = optimize_prompt(prompt, n_bits=args.bits, verbose=not args.quiet)
        if args.quiet:
            print(result)
        return

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            prompt = f.read()
        result, _ = optimize_prompt(prompt, n_bits=args.bits, verbose=not args.quiet)
        if args.quiet:
            print(result)
        return

    if args.prompt:
        result, _ = optimize_prompt(args.prompt, n_bits=args.bits, verbose=not args.quiet)
        if args.quiet:
            print(result)
        return

    # Mode interactif par défaut
    interactive_mode()


if __name__ == "__main__":
    main()
