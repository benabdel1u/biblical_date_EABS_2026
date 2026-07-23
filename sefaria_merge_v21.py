# sefaria_merge_v21.py
#
# Aligns the Hebrew verbs of the Pentateuch (BHSA) with their Syriac counterparts
# in the Leiden Peshitta (ETCBC / CALAP encoding), verse by verse, and writes one
# row per Hebrew verb. See README_v21.md for inputs, outputs and operation, and
# CHANGELOG_v21.md for the history of the versions.
#
# v21 is a documentation release: no scoring behaviour changed. The calibration
# block below now reports the full manual verification campaign (246 pair types,
# two thirds of the matched occurrences), which supersedes the pilot of 99 pairs on
# which the v20 figures rested.
#
# Design decisions that are load-bearing, and are argued at their point of use:
#   - wayyiqtol is NEUTRAL by default: it earns no tense bonus, and loses none.
#   - The verbal stem is NOT scored (see the note before HEB_TENSE_TO_SYR).
#   - Absence of root evidence is not grounds for rejection (see accept_assignment).
#   - Syriac morphology is parsed BY PREFIX, not by column position.
#   - Hebrew (BHSA) and Syriac (CALAP) tag vocabularies are mapped explicitly.
#   - Book names differ on each side (Genesis / Gen) and go through canonical keys.
#   - The Hebrew header sits on the second row of the sheet.

import json
import re
import unicodedata

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

try:
    from scipy.optimize import linear_sum_assignment
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ===========================================================================
# Configuration
# ===========================================================================

HEBREW_FILE = 'list_verbs_Pentateuch_Hebrew.xlsx'
SYRIAC_FILE = 'list_verbs_Pentateuch_Peshitta.xlsx'
OUTPUT_FILE = 'merged_verbs_results_v21.csv'

# The Hebrew sheet carries a title line above the real header.
HEBREW_HEADER_ROW = 1     # 0-indexed; set to 0 if you ever clean the sheet.

# Optional. A two-column file (lemma, gloss) used to fill in the Syriac glosses,
# which are absent from the Peshitta spreadsheet. Leave as None until you have it.
# Accepts .xlsx or .csv. Joined on `lemma` (the Latin transliteration code).
SYRIAC_GLOSS_FILE = None

# --- wayyiqtol policy ------------------------------------------------------
# The Hebrew wayyiqtol is a prefix conjugation with past reference. The Peshitta
# renders it overwhelmingly with the perfect. Hard-coding wayq -> pf would write a
# translational hypothesis into the scoring function and then reward the matches that
# confirm it; that is question-begging if the rendering of wayyiqtol is itself an object
# of study. wayq is 23.2% of the Hebrew Pentateuch verbs, so this is not a marginal call.
#
#   'neutral'  (default) wayq forms receive NO tense bonus and NO penalty. The tense
#              component is simply abstained on. Person still scores normally.
#   'pf'       wayq is treated as equivalent to Syriac pf, like any other tense pair.
#              Defensible, but must be declared in the write-up.
#   'observe'  same as 'neutral' for scoring, but the Syriac tense actually chosen for
#              each wayq form is recorded in a `wayq_rendered_as` column, so the
#              distribution can be studied after the fact instead of assumed.
WAYQ_POLICY = 'neutral'

SYRIAC_MORPH_SOURCE_COLS = ['verb_tense', 'vt', 'ps', 'vo']
SYRIAC_EXPORT_COL_CANDIDATES = ['word_form']

WEIGHTS = dict(canonical_bonus=1.0, sefaria_bonus=0.6, bert_weight=1.0)
MORPH_PER_FIELD = 0.15

# --- Acceptance thresholds -------------------------------------------------
# CALIBRATED by hand. These are measured, not guessed.
#
# Method. A pair TYPE is one Hebrew lexeme matched with one Syriac lemma, judged once
# whatever its frequency: the couple 'MR / >MR occurs 1,327 times and was judged once.
# 246 such types were annotated by the author against the Masoretic and the Leiden text.
# They account for 11,119 of the 17,145 matched occurrences, i.e. two thirds of the
# corpus. Verdicts and corrections are in the annotation file shipped with the data.
#
# THESE FIGURES REPLACE THOSE OF v20, which rested on a pilot of 99 pairs confined to the
# 0.31-0.57 range. The pilot was optimistic: it gave 91% by type overall, and 89% / 100% /
# 95% on the three bands it covered, against 60% / 51% / 67% once those bands were fully
# enumerated. If you are comparing against an older printout, this is why the numbers moved.
# The occurrence-weighted figure barely changed (98% presumed, 98.9% measured), because the
# corrected types are rare ones.
#
# Precision by score band (`Score` in the Match_Method column):
#
#     band          types   by type   by occurrence   occurrences
#     0.30 - 0.35       9      56%          -              -        <- rejected
#     0.35 - 0.40      20      60%         68%            31
#     0.40 - 0.45      37      51%         79%            97
#     0.45 - 0.55      97      67%         95%           665
#     0.55 - 0.57       6     100%        100%            53
#     0.57 - 0.80      35      91%         95%         1,225
#     0.80 - 1.00       4     100%        100%           363
#     1.00 and above   38     100%        100%         8,685
#     ---------------------------------------------------------
#     overall         246      73.6%       98.9%       11,119
#
# The two columns answer different questions and must not be confused. By type, every
# couple weighs the same, so the figure is dominated by rare couples: 73.6%. By
# occurrence, which is how every table computed from this file counts, the figure is
# 98.9%, and 99.5% in the high-score zone that carries the bulk of the corpus. The
# errors sit almost entirely in couples occurring once or twice; the single largest is
# VM' / HWJ (45 occurrences), where the Peshitta uses the periphrasis NHW' VM' and the
# programme matches the auxiliary.
#
# WHY 0.35. Precision collapses below it: 56% in the 0.30-0.35 band, against 60% and
# above once the threshold is cleared. The earlier threshold of 0.60 discarded 2,618
# correct matches on the Pentateuch. Raising the threshold to 0.40 removes only 19 of
# the 17,145 matched pairs and leaves every published figure unchanged, so the results
# are insensitive to the threshold upwards and protected by the collapse downwards.
#
# TRAP: the acceptance rule requires BOTH conditions. Lowering MIN_BERT_SEMANTIC_ONLY
# while leaving MIN_TOTAL_SCORE at 0.60 changes nothing, because a pair scoring 0.39
# with no morphological bonus totals 0.39. The two must move together.
MIN_BERT_SEMANTIC_ONLY = 0.35
MIN_TOTAL_SCORE = 0.35

# Effect on the full Pentateuch, precision now measured across the whole score range
# rather than presumed above 0.57:
#     threshold   matched   rate     precision   expected false positives
#     0.60         14,544   80.4%       -              -
#     0.40         17,126   94.7%      98.9%        ~189
#     0.35         17,145   94.8%      98.9%        ~189  (1.1% of matches)


# ===========================================================================
# Nomenclature: BHSA (Hebrew) <-> CALAP word_grammar (Syriac)
# ===========================================================================

# Canonical book keys. Both sides are mapped onto these.
BOOK_CANON = {
    # Hebrew side (BHSA Latin names)
    'genesis': 'GEN', 'exodus': 'EXO', 'leviticus': 'LEV',
    'numeri': 'NUM', 'deuteronomium': 'DEU',
    # Syriac side. The rebuilt sheet uses full English names, which agree with the
    # Hebrew for Genesis/Exodus/Leviticus but NOT for Numbers/Deuteronomy (the Hebrew
    # sheet keeps the Latin Numeri/Deuteronomium). The short forms of the earlier
    # Peshitta export are kept so that older files still load.
    'numbers': 'NUM', 'deuteronomy': 'DEU',
    'gen': 'GEN', 'ex': 'EXO', 'lev': 'LEV', 'nm': 'NUM', 'deut': 'DEU',
}


# The Syriac glosses come from a Syriac lexicon and occasionally carry lexicographic
# apparatus that means nothing to a sentence encoder: an infinitival "to " (140 rows),
# or a stem marker ("PA command"). BHSA glosses have neither. Stripping them costs
# nothing and removes noise from the embedding.
_GLOSS_STEM_MARKER = re.compile(
    r'^\s*(?:pe|pa|af|aph|eth|ethpe|ethpa|ettaf|shaph)\.?\s+', re.IGNORECASE)


def normalize_gloss(g):
    if not isinstance(g, str):
        return g
    g = g.strip()
    g = _GLOSS_STEM_MARKER.sub('', g)
    if g.lower().startswith('to '):
        g = g[3:]
    return g.strip()

# CALAP word_grammar, `vt` function:
#   pf perfect | ipf imperfect | imp imperative | inf infinitive | ptc participle
# BHSA `vt` feature:
#   perf | impf | wayq | impv | infa | infc | ptca | ptcp
HEB_TENSE_TO_SYR = {
    'perf': 'pf',
    'impf': 'ipf',
    'impv': 'imp',
    'infa': 'inf',
    'infc': 'inf',
    'ptca': 'ptc',
    'ptcp': 'ptc',
    # 'wayq' is deliberately absent: see WAYQ_POLICY.
}

# CALAP `ps`: first | second | third.  BHSA `ps`: p1 | p2 | p3 | unknown.
HEB_PERSON_TO_SYR = {
    'p1': 'first',
    'p2': 'second',
    'p3': 'third',
    # 'unknown' is deliberately absent: it must never match anything.
}

# NOTE. The verbal stem is NOT scored, and this is a deliberate choice, not an oversight.
# BHSA has qal/nif/piel/pual/hif/hof/hit/... ; CALAP has only pe/pa/af. A mapping
# qal->pe, piel->pa, hif->af is tempting, but whether the Peshitta preserves the stem of
# its Vorlage is itself a research question. Scoring it would presuppose the answer, in
# exactly the way WAYQ_POLICY is designed to avoid.


# ===========================================================================
# Caches
# ===========================================================================

SEFARIA_CACHE_FILE = 'sefaria_cache.json'   # set to None to disable

def _load_sefaria_cache():
    if not SEFARIA_CACHE_FILE:
        return {}
    try:
        with open(SEFARIA_CACHE_FILE, encoding='utf-8') as f:
            c = json.load(f)
        print(f"  Sefaria cache: {len(c)} roots loaded from {SEFARIA_CACHE_FILE}")
        return c
    except (FileNotFoundError, ValueError):
        return {}


def _save_sefaria_cache():
    if not SEFARIA_CACHE_FILE:
        return
    try:
        with open(SEFARIA_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(sefaria_cache, f, ensure_ascii=False)
    except OSError as e:
        print(f"  [warn] could not save the Sefaria cache: {e}")


sefaria_cache = _load_sefaria_cache()
canonical_cache = {}
gloss_embedding_cache = {}


# ===========================================================================
# Transliteration and root aliases
# ===========================================================================

HEBREW_TRANSLIT_MAP = {
    'א': '>', 'ב': 'B', 'ג': 'G', 'ד': 'D', 'ה': 'H', 'ו': 'W', 'ז': 'Z',
    'ח': 'X', 'ט': 'V', 'י': 'J', 'כ': 'K', 'ך': 'K', 'ל': 'L', 'מ': 'M', 'ם': 'M',
    'נ': 'N', 'ן': 'N', 'ס': 'S', 'ע': '<', 'פ': 'P', 'ף': 'P', 'צ': 'Y', 'ץ': 'Y',
    'ק': 'Q', 'ר': 'R', 'שׂ': 'S', 'שׁ': 'C', 'ש': 'C', 'ת': 'T'
}

SYRIAC_TRANSLIT_MAP = {
    'ܐ': '>', 'ܒ': 'B', 'ܓ': 'G', 'ܕ': 'D', 'ܗ': 'H', 'ܘ': 'W', 'ܙ': 'Z',
    'ܚ': 'X', 'ܛ': 'V', 'ܝ': 'J', 'ܟ': 'K', 'ܠ': 'L', 'ܡ': 'M', 'ܢ': 'N',
    'ܣ': 'S', 'ܥ': '<', 'ܦ': 'P', 'ܨ': 'Y', 'ܩ': 'Q', 'ܪ': 'R', 'ܫ': 'C', 'ܬ': 'T',
}

ROOT_ALIASES = {
    'HRH': {'BVN'},
    'NTN': {'CLM'},
    'RDH': {'CLV'},
    'MCL': {'CLV'},
    'CLV': {'CLV', 'MCL', 'RDH'},
    'BW>': {'>TJ'},
    'NWX': {'CBQ'},
    '<SH': {'<BD'},
}

SYRIAC_SCRIPT_RE = re.compile(r'[\u0700-\u074F]')


def transliterate(text, translit_map):
    if not isinstance(text, str):
        return ""
    sorted_keys = sorted(translit_map.keys(), key=len, reverse=True)
    out, i, n = [], 0, len(text)
    while i < n:
        for key in sorted_keys:
            if text.startswith(key, i):
                out.append(translit_map[key])
                i += len(key)
                break
        else:
            i += 1
    return "".join(out)


def transliterate_hebrew(root):
    return transliterate(root, HEBREW_TRANSLIT_MAP)


def transliterate_syriac(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[\u0730-\u074A]', '', text)
    return transliterate(text, SYRIAC_TRANSLIT_MAP)


def syriac_lemma_to_translit(lemma):
    """The Peshitta sheet stores lemmas already in the Latin code, but we detect the
    script per cell rather than assume it (v17 fix, retained)."""
    if not isinstance(lemma, str):
        return ""
    lemma = lemma.strip()
    return transliterate_syriac(lemma) if SYRIAC_SCRIPT_RE.search(lemma) else lemma


# ===========================================================================
# Canonicalization
# ===========================================================================

def expand_aliases_for_root(root):
    aliases = {root}
    if root in ROOT_ALIASES:
        aliases |= set(ROOT_ALIASES[root])
    for k, v in ROOT_ALIASES.items():
        if root in v:
            aliases.add(k)
            aliases |= set(v)
    return aliases


def normalize_initial_aleph_he_variation(form):
    v = {form}
    if form.startswith('>'):
        v.add('H' + form[1:])
    elif form.startswith('H'):
        v.add('>' + form[1:])
    return v


def normalize_s_sh_c_collisions(form):
    return {form, form.translate(str.maketrans({'S': 'C', 'C': 'S'}))}


def normalize_final_matres(form):
    v = {form}
    if form.endswith('>') or form.endswith('H'):
        v.add(form[:-1] + 'J')
    if form.endswith('W') and len(form) > 1:
        v.add(form[:-1])
    return v


def create_canonical_variants(translit_root):
    if not isinstance(translit_root, str) or not translit_root:
        return frozenset()
    if translit_root in canonical_cache:
        return canonical_cache[translit_root]

    variants = {translit_root} | normalize_final_matres(translit_root)
    variants |= {x for v in variants for x in normalize_initial_aleph_he_variation(v)}
    variants |= {x for v in variants for x in normalize_s_sh_c_collisions(v)}
    variants |= {x for v in variants for x in expand_aliases_for_root(v)}

    canonical = frozenset(variants)
    canonical_cache[translit_root] = canonical
    return canonical


# ===========================================================================
# Syriac morphology: parse BY PREFIX
# ===========================================================================

_TAG_RE = re.compile(r'^(vs|vt|ps|vo)=(.+)$')


def parse_syriac_morphology(row):
    """
    The Peshitta export writes tags as `vt=pf`, `ps=third`, `vo=act`, `vs=pe`, but the
    CALAP word_grammar deletes `ps` for participles and infinitives:

        not exist(vbe)              :: -vt, -vs, -ps
        exist(vbe) && exist(nme)    :: -ps

    When a field is deleted the remaining ones shift left, so `vo=act` lands in the column
    headed PS in 3,962 rows (21%). Reading by column position is therefore wrong, and
    "repairing" those rows would be wrong too: the absence of `ps` is meaningful, not
    missing. We read the prefix and ignore the header.
    """
    tags = {}
    for col in SYRIAC_MORPH_SOURCE_COLS:
        val = row.get(col)
        if not isinstance(val, str):
            continue
        m = _TAG_RE.match(val.strip())
        if m:
            tags[m.group(1)] = m.group(2).strip()
    return tags


# ===========================================================================
# Sefaria
# ===========================================================================

def get_sefaria_cognates(hebrew_root):
    if hebrew_root in sefaria_cache:
        return sefaria_cache[hebrew_root]
    url = f"https://www.sefaria.org/api/words/{hebrew_root}"
    candidates = []
    try:
        r = requests.get(url, headers={'User-Agent': 'Sefaria-Merge-Script/1.0'}, timeout=10)
        if r.status_code == 200:
            for entry in r.json():
                if entry.get("parent_lexicon") == "BDB Dictionary" and entry.get("root") is True:
                    for sense in entry.get('content', {}).get('senses', []):
                        for m in re.findall(r'([\u0700-\u074F]+)', sense.get('definition', '')):
                            tr = transliterate_syriac(m)
                            if tr:
                                candidates.extend(create_canonical_variants(tr))
    except requests.exceptions.RequestException:
        candidates = []
    candidates = sorted(set(candidates))
    sefaria_cache[hebrew_root] = candidates
    return candidates


# ===========================================================================
# Semantic similarity
# ===========================================================================

def get_embedding(gloss, model):
    if not isinstance(gloss, str) or not gloss.strip():
        return None
    if gloss not in gloss_embedding_cache:
        gloss_embedding_cache[gloss] = model.encode(gloss, convert_to_tensor=True)
    return gloss_embedding_cache[gloss]


def bert_similarity(heb_row, syr_row, model):
    """Raw LaBSE cosine. Returns 0.0 whenever either gloss is missing, which is the case
    for the entire Peshitta sheet as it currently stands."""
    if model is None:
        return 0.0
    from sentence_transformers import util
    he = get_embedding(heb_row.get('gloss', ''), model)
    se = get_embedding(syr_row.get('gloss', ''), model)
    if he is None or se is None:
        return 0.0
    try:
        return float(util.cos_sim(he, se).item())
    except Exception:
        return 0.0


# ===========================================================================
# Scoring
# ===========================================================================

def heb_syr_morph_similarity(heb_row, syr_row):
    """
    Returns (score_in_{0,1,2}, tense_abstained: bool).

    Hebrew tense/person are BHSA tags; Syriac vt/ps are CALAP tags. They are mapped, not
    string-compared: `impf` != `ipf` and `p3` != `third` as raw strings, so the v16/v17
    equality test silently scored 0 on the whole corpus.
    """
    score = 0
    heb_tense = str(heb_row.get('tense', '') or '').strip().lower()
    heb_person = str(heb_row.get('person', '') or '').strip().lower()
    syr_vt = str(syr_row.get('syr_vt', '') or '').strip().lower()
    syr_ps = str(syr_row.get('syr_ps', '') or '').strip().lower()

    tense_abstained = False
    if heb_tense == 'wayq' and WAYQ_POLICY in ('neutral', 'observe'):
        tense_abstained = True          # abstain: no bonus, no penalty
    else:
        mapped = 'pf' if (heb_tense == 'wayq' and WAYQ_POLICY == 'pf') \
            else HEB_TENSE_TO_SYR.get(heb_tense)
        if mapped and syr_vt and mapped == syr_vt:
            score += 1

    mapped_ps = HEB_PERSON_TO_SYR.get(heb_person)
    if mapped_ps and syr_ps and mapped_ps == syr_ps:
        score += 1

    return score, tense_abstained


def compute_pair_score(heb_row, syr_row, model, local_sefaria_cache):
    heb_can = heb_row.get('canonical_variants', frozenset())
    syr_can = syr_row.get('canonical_variants', frozenset())

    can = WEIGHTS['canonical_bonus'] if heb_can & syr_can else 0.0

    heb_root = heb_row.get('lexeme-v', '')
    if heb_root not in local_sefaria_cache:
        local_sefaria_cache[heb_root] = set(get_sefaria_cognates(heb_root))
    sef = WEIGHTS['sefaria_bonus'] if local_sefaria_cache[heb_root] & syr_can else 0.0

    bert_raw = bert_similarity(heb_row, syr_row, model)
    bert = WEIGHTS['bert_weight'] * bert_raw

    morph_n, tense_abstained = heb_syr_morph_similarity(heb_row, syr_row)
    morph = MORPH_PER_FIELD * morph_n

    total = can + sef + bert + morph
    comps = dict(canonical_bonus=can, sefaria_bonus=sef, bert_sim=bert,
                 bert_raw=bert_raw, morph_bonus=morph, tense_abstained=tense_abstained)
    return total, comps


def accept_assignment(total, comps):
    """
    The Hungarian algorithm always returns a full assignment; it has no notion of
    'no acceptable partner'. Some gate is therefore necessary, or every leftover Hebrew
    verb is forcibly paired with whatever Syriac verb remains.

    But the gate must not be too tight, and v17's was.

    v16's changelog held up Gn 5:22 as the case to fix: once the two yalad/JLD pairs are
    taken, halak "walk" faces only CPR "be beautiful", and the changelog argued that
    Not Found "is more accurate". v17 built this rule to guarantee that outcome.

    THE PREMISE WAS WRONG. Hand annotation marks halak x CPR as a CORRECT
    alignment: the Peshitta renders "Enoch walked with God" by "Enoch pleased God"
    (Gn 5:22, 5:24, 6:9). It is a genuine verbal equivalence, and a theologically
    interesting one - exactly the kind of case this project exists to find. Rejecting it
    was not conservatism, it was data loss.

    The rule below therefore keeps the structural shortcut (root evidence is always enough)
    but no longer treats the ABSENCE of root evidence as grounds for suspicion. A purely
    semantic match now only has to clear a properly calibrated threshold.
    """
    if comps['canonical_bonus'] > 0 or comps['sefaria_bonus'] > 0:
        return True, 'root evidence'
    if comps['bert_raw'] >= MIN_BERT_SEMANTIC_ONLY and total >= MIN_TOTAL_SCORE:
        return True, 'semantic only'
    return False, 'below threshold'


# ===========================================================================
# Hungarian assignment
# ===========================================================================

def _pad_to_square(W, pad_value=0.0):
    W = np.asarray(W, dtype=float)
    n, m = W.shape
    if n == m:
        return W
    if n < m:
        return np.vstack([W, np.full((m - n, m), pad_value)])
    return np.hstack([W, np.full((n, n - m), pad_value)])


def _hungarian_maximize(W):
    W = np.asarray(W, dtype=float)
    if W.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    cost = (np.nanmax(W) if np.isfinite(W).any() else 0.0) - W
    return linear_sum_assignment(cost) if _HAVE_SCIPY else _hungarian_pure_python(cost)


def _hungarian_pure_python(cost):
    """Square Hungarian, minimising `cost`. v17 fixed the two `np.where(...)` calls that
    lacked `[0]` and made this crash on any augmenting path. Validated against SciPy."""
    cost = np.asarray(cost, dtype=float).copy()
    N = cost.shape[0]
    assert cost.shape[0] == cost.shape[1], "square matrix expected"

    cost -= cost.min(axis=1, keepdims=True)
    cost -= cost.min(axis=0, keepdims=True)

    starred = np.zeros_like(cost, dtype=bool)
    primed = np.zeros_like(cost, dtype=bool)
    covered_rows = np.zeros(N, dtype=bool)
    covered_cols = np.zeros(N, dtype=bool)

    for i in range(N):
        for j in range(N):
            if cost[i, j] == 0 and not covered_rows[i] and not covered_cols[j]:
                starred[i, j] = covered_rows[i] = covered_cols[j] = True
                break
    covered_rows[:] = covered_cols[:] = False

    def cover_cols_with_starred():
        covered_cols[:] = starred.any(axis=0)

    cover_cols_with_starred()

    def find_uncovered_zero():
        for i in range(N):
            if not covered_rows[i]:
                for j in range(N):
                    if not covered_cols[j] and cost[i, j] == 0:
                        return i, j
        return None

    while covered_cols.sum() < N:
        pos = find_uncovered_zero()
        while pos is None:
            ur, uc = ~covered_rows, ~covered_cols
            m = np.min(cost[np.ix_(ur, uc)])
            cost[covered_rows, :] += m
            cost[:, uc] -= m
            pos = find_uncovered_zero()

        i, j = pos
        primed[i, j] = True
        star_col = np.where(starred[i])[0]

        if star_col.size == 0:
            path = [(i, j)]
            while True:
                rows = np.where(starred[:, path[-1][1]])[0]
                if rows.size == 0:
                    break
                r = int(rows[0])
                path.append((r, path[-1][1]))
                path.append((r, int(np.where(primed[r])[0][0])))
            for (r, c) in path:
                starred[r, c] = not starred[r, c]
            primed[:] = False
            covered_rows[:] = False
            cover_cols_with_starred()
        else:
            covered_rows[i] = True
            covered_cols[int(star_col[0])] = False

    return np.arange(N, dtype=int), np.argmax(starred, axis=1)


# ===========================================================================
# Matching one verse
# ===========================================================================

def match_verse(heb_verse_df, syr_candidates_df, model, export_cols):
    results = []

    def unmatched(heb_row, method):
        row = heb_row.to_dict()
        row['Syriac_Lemma'] = None
        row['Syriac_Gloss'] = None
        row['Syriac_VT'] = None
        row['Syriac_PS'] = None
        row['Syriac_VS'] = None
        row['Syriac_VO'] = None
        for c in export_cols:
            row[c] = None
        if WAYQ_POLICY == 'observe':
            row['wayq_rendered_as'] = None
        row['Match_Method'] = method
        return row

    if syr_candidates_df is None or syr_candidates_df.empty:
        return [unmatched(r, 'Not Found (no Syriac verb in this verse)')
                for _, r in heb_verse_df.iterrows()]

    heb_idx = list(heb_verse_df.index)
    syr_idx = list(syr_candidates_df.index)
    H, S = heb_verse_df.loc[heb_idx], syr_candidates_df.loc[syr_idx]

    local_sef = {}
    W = np.zeros((len(H), len(S)))
    comps_grid = [[None] * len(S) for _ in range(len(H))]

    for i, (_, h) in enumerate(H.iterrows()):
        for j, (_, s) in enumerate(S.iterrows()):
            W[i, j], comps_grid[i][j] = compute_pair_score(h, s, model, local_sef)

    row_ind, col_ind = _hungarian_maximize(_pad_to_square(W))

    assign = {}
    for i_r, j_c in zip(row_ind, col_ind):
        if i_r < len(H) and j_c < len(S):
            assign[heb_idx[i_r]] = (S.loc[syr_idx[j_c]], float(W[i_r, j_c]), comps_grid[i_r][j_c])

    for idx, h in heb_verse_df.iterrows():
        got = assign.get(idx)
        if got is None:
            results.append(unmatched(h, 'Not Found (no candidate left)'))
            continue

        s, score, c = got
        morph_str = 'abstain' if c['tense_abstained'] else f"{c['morph_bonus']:.2f}"
        breakdown = (f"Score: {score:.2f} | can={c['canonical_bonus']:.2f}, "
                     f"sef={c['sefaria_bonus']:.2f}, bert={c['bert_sim']:.2f}, "
                     f"morph={c['morph_bonus']:.2f}"
                     + (" (tense abstained: wayq)" if c['tense_abstained'] else ""))

        accepted, reason = accept_assignment(score, c)
        if not accepted:
            results.append(unmatched(
                h, f"Not Found ({reason} | {breakdown} | best candidate was {s.get('lemma')})"))
            continue

        row = h.to_dict()
        row['Syriac_Lemma'] = s.get('lemma')
        row['Syriac_Gloss'] = s.get('gloss')
        row['Syriac_VT'] = s.get('syr_vt')
        row['Syriac_PS'] = s.get('syr_ps')
        row['Syriac_VS'] = s.get('syr_vs')
        row['Syriac_VO'] = s.get('syr_vo')
        for col in export_cols:
            row[col] = s.get(col)
        if WAYQ_POLICY == 'observe':
            row['wayq_rendered_as'] = s.get('syr_vt') if str(h.get('tense')).lower() == 'wayq' else None
        row['Match_Method'] = f"Global Opt Match ({breakdown})"
        results.append(row)

    return results


# ===========================================================================
# Loading and preparation
# ===========================================================================

def canonical_book(name):
    return BOOK_CANON.get(str(name).strip().lower())


def load_and_prepare():
    df_heb = pd.read_excel(HEBREW_FILE, header=HEBREW_HEADER_ROW)
    df_syr = pd.read_excel(SYRIAC_FILE)
    df_heb.columns = df_heb.columns.str.lower().str.strip()
    df_syr.columns = df_syr.columns.str.lower().str.strip()

    for name, df in (('Hebrew', df_heb), ('Syriac', df_syr)):
        missing = {'book', 'chapter', 'verse'} - set(df.columns)
        if missing:
            raise ValueError(
                f"{name} sheet is missing {missing}. Columns found: {list(df.columns)[:6]}... "
                f"If the Hebrew sheet has a title line above the header, check HEBREW_HEADER_ROW.")

    # --- Books -------------------------------------------------------------
    for name, df in (('Hebrew', df_heb), ('Syriac', df_syr)):
        df['book_key'] = df['book'].map(canonical_book)
        unknown = sorted(df.loc[df['book_key'].isna(), 'book'].dropna().unique())
        if unknown:
            print(f"  [warn] {name}: book names absent from BOOK_CANON, rows dropped: {unknown}")
        df.drop(df.index[df['book_key'].isna()], inplace=True)

    shared = sorted(set(df_heb['book_key']) & set(df_syr['book_key']))
    print(f"  books shared by both sheets: {shared}")
    if not shared:
        raise ValueError("No book in common after normalisation. Check BOOK_CANON.")

    for df in (df_heb, df_syr):
        df['chapter'] = df['chapter'].astype(int)
        df['verse'] = df['verse'].astype(int)

    # --- Syriac morphology, parsed by prefix -------------------------------
    tags = df_syr.apply(parse_syriac_morphology, axis=1)
    for f in ('vs', 'vt', 'ps', 'vo'):
        df_syr[f'syr_{f}'] = tags.map(lambda d, f=f: d.get(f))
    n_no_ps = int(df_syr['syr_ps'].isna().sum())
    print(f"  Syriac forms without `ps` (participles/infinitives, by design): "
          f"{n_no_ps} / {len(df_syr)}")

    # --- Syriac glosses ----------------------------------------------------
    if 'gloss' not in df_syr.columns:
        df_syr['gloss'] = np.nan
    if SYRIAC_GLOSS_FILE:
        reader = pd.read_csv if str(SYRIAC_GLOSS_FILE).lower().endswith('.csv') else pd.read_excel
        gl = reader(SYRIAC_GLOSS_FILE)
        gl.columns = gl.columns.str.lower().str.strip()
        gl = gl[['lemma', 'gloss']].dropna().drop_duplicates('lemma')
        df_syr = df_syr.drop(columns=['gloss']).merge(gl, on='lemma', how='left')
        print(f"  glosses loaded from {SYRIAC_GLOSS_FILE}: "
              f"{df_syr['gloss'].notna().sum()} / {len(df_syr)} rows covered")

    df_syr['gloss'] = df_syr['gloss'].map(normalize_gloss)
    df_heb['gloss'] = df_heb['gloss'].map(normalize_gloss)

    n_gloss = int(df_syr['gloss'].notna().sum())
    if n_gloss == 0:
        n_lemmas = df_syr['lemma'].nunique()
        print("\n  " + "!" * 68)
        print("  !! NO SYRIAC GLOSS. The LaBSE component is inert: bert = 0 everywhere.")
        print("  !! Matching is therefore ROOT-BASED ONLY (canonical + Sefaria).")
        print("  !! Suppletive pairs are unreachable by construction:")
        print("  !!     ra'ah / XZJ,  halak / >ZL,  natan / JHB,  hayah / HWJ")
        print(f"  !! Set SYRIAC_GLOSS_FILE to fix. Only {n_lemmas} distinct lemmas need a gloss.")
        print("  " + "!" * 68 + "\n")

    # --- Roots and canonical variants --------------------------------------
    df_heb['translit_root'] = df_heb['lexeme-v'].apply(transliterate_hebrew)
    df_syr['translit_root'] = df_syr['lemma'].apply(syriac_lemma_to_translit)
    df_heb['canonical_variants'] = df_heb['translit_root'].apply(create_canonical_variants)
    df_syr['canonical_variants'] = df_syr['translit_root'].apply(create_canonical_variants)

    export_cols = [c for c in SYRIAC_EXPORT_COL_CANDIDATES if c in df_syr.columns]
    return df_heb, df_syr, export_cols


# ===========================================================================
# Main
# ===========================================================================

def main():
    print(f"wayq policy: {WAYQ_POLICY}")
    if not _HAVE_SCIPY:
        print("  [warn] SciPy absent; using the pure-Python Hungarian fallback.")

    df_heb, df_syr, export_cols = load_and_prepare()

    model = None
    if int(df_syr['gloss'].notna().sum()) > 0:
        from sentence_transformers import SentenceTransformer
        print("Loading LaBSE...")
        model = SentenceTransformer('sentence-transformers/LaBSE')
        print("Model loaded.")
    else:
        print("Skipping LaBSE: there is nothing to encode on the Syriac side.")

    heb_grouped = df_heb.groupby(['book_key', 'chapter', 'verse'])
    syr_grouped = df_syr.groupby(['book_key', 'chapter', 'verse'])

    results = []
    for key, heb_verse in tqdm(heb_grouped, desc="Matching verbs"):
        try:
            syr_cands = syr_grouped.get_group(key).copy()
        except KeyError:
            syr_cands = pd.DataFrame()
        results.extend(match_verse(heb_verse, syr_cands, model, export_cols))

    _save_sefaria_cache()

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    mm = out['Match_Method'].astype(str)
    nf = mm.str.startswith('Not Found')
    print(f"\n--- DONE ---  {OUTPUT_FILE}")
    print(f"  matched   : {(~nf).sum()} / {len(out)} ({(~nf).mean():.1%})")
    print(f"  not found : {nf.sum()}")
    print(f"    no Syriac verb in verse : {mm.str.contains('no Syriac verb').sum()}")
    print(f"    below threshold         : {mm.str.contains('below threshold').sum()}")
    print(f"    no candidate left       : {mm.str.contains('no candidate left').sum()}")
    sef_fired = mm.str.contains(r'sef=0\.60')
    can_fired = ~mm.str.contains(r'can=0\.00') & ~nf

    n_root = int(can_fired.sum())
    n_sef = int((sef_fired & ~can_fired & ~nf).sum())
    n_sem = int((~nf).sum()) - n_root - n_sef
    n_ok = int((~nf).sum())

    print("\n  --- TYPOLOGY OF THE EQUIVALENCES ---")
    print(f"    direct root kinship          : {n_root:>6}  {n_root / n_ok:.1%}")
    print(f"    disguised kinship (via BDB)  : {n_sef:>6}  {n_sef / n_ok:.1%}")
    print(f"    lexical replacement          : {n_sem:>6}  {n_sem / n_ok:.1%}")
    print("\n  Sefaria was DECISIVE (recovered a cognate no rule could reach): " + str(n_sef))
    print("\n  Reminder: MIN_BERT_SEMANTIC_ONLY / MIN_TOTAL_SCORE are uncalibrated. "
          "No rate printed above is quotable until they are checked against a manual sample.")


if __name__ == "__main__":
    main()
