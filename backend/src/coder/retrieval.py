"""Hybrid retrieval over the ICD-10-CM and CPT code corpora.

We use two retrievers in parallel:

  1. BM25 (rank-bm25) for keyword/exact matches. This is critical because many
     ICD-10-CM descriptions contain rare medical terms that an embedding model
     may not have seen often (e.g., "Type 1 diabetes mellitus with diabetic
     chronic kidney disease, stage 4").

  2. Dense embedding retrieval (Chroma + BAAI/bge-small-en-v1.5) for semantic
     matches. This catches paraphrases ("high blood pressure" → "Essential
     hypertension") that BM25 misses.

Results are merged with Reciprocal Rank Fusion (RRF), which is a simple,
parameter-light fusion method that consistently outperforms either retriever
alone in published benchmarks on clinical and general-domain retrieval.

Why bge-small-en-v1.5: it's a strong open-source embedding model, small enough
to run on CPU, and Apache-2.0 licensed. For higher quality, swap in
medcpt-query/article from NCBI which is medical-domain-specific.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

from .schema import CodeCandidate, TextSpan

_CHROMA_DIR = Path(__file__).resolve().parents[2] / ".chroma"
_BM25_CACHE: dict[str, tuple[BM25Okapi, list[dict]]] = {}


@lru_cache(maxsize=1)
def _get_spacy_nlp():
    import spacy
    return spacy.load("en_core_web_sm")


_NUMERIC_RE = re.compile(
    r"^[\d\s./:,]+"          # pure numbers / fractions / dates
    r"|^\d+\s*(?:mg|ml|mcg|g|kg|mmhg|mmol|units?|tabs?)$"  # doses/units
    r"|^(?:bid|tid|qid|qd|prn|po|iv|im|sc|sq|hs|ac|pc|stat)$",  # sig abbreviations
    re.I,
)

# Exact-match terms that produce only noise when used as standalone queries.
# "follow-up" is the critical one: it surfaces Z encounter codes (Z39.2, Z08,
# Z36.2) at rank #1 which then dominate the merged candidate set.
_ADMIN_BLOCKLIST = frozenset({
    "follow-up", "follow up", "type",
})

# Tokens that carry no clinical coding signal on their own.
_NONCLINICAL_TOKENS = frozenset({
    "male", "female", "man", "woman", "patient", "presents",
    "year", "old", "aged", "today", "currently", "now",
})

# Common drug name suffixes — single-word queries ending in these are
# medication names with no direct ICD-10 code and produce junk BM25 results.
_DRUG_SUFFIXES = (
    "olol", "pril", "sartan", "statin", "flozin",
    "gliptin", "tidine", "azole", "mycin", "cillin", "cycline", "mab",
)

# Medical abbreviations that the BM25 corpus does not contain (it indexes the
# full ICD descriptions). Expanding them before spaCy NLP ensures the NER and
# noun-chunk extractor produce useful query strings.
#
# "post-MI" is the canonical failure case: the hyphen causes spaCy to emit
# the token "MI" which is then dropped by the len < 4 guard, so I25.2
# (Old myocardial infarction) never enters the candidate set. After expansion,
# spaCy sees "post-myocardial infarction" and extracts "myocardial infarction"
# as a noun chunk.
#
# "EF 45%" is the second failure case: "EF" is 2 chars (dropped) and "45%"
# is filtered as numeric, so echocardiogram-derived heart failure diagnoses
# are invisible to retrieval. Expanding EF → "ejection fraction" lets spaCy
# emit the chunk and BM25 match I50.2x/I50.3x descriptions.
_ABBR_EXPANSIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bMI\b"),   "myocardial infarction"),
    (re.compile(r"\bHF\b"),   "heart failure"),
    (re.compile(r"\bEF\b"),   "ejection fraction"),
    (re.compile(r"\bHTN\b"),  "hypertension"),
    (re.compile(r"\bCAD\b"),  "coronary artery disease"),
    (re.compile(r"\bCKD\b"),  "chronic kidney disease"),
    (re.compile(r"\bDVT\b"),  "deep vein thrombosis"),
    (re.compile(r"\bT2DM\b"), "type 2 diabetes mellitus"),
    (re.compile(r"\bAFib\b"), "atrial fibrillation"),
    (re.compile(r"\bAfib\b"), "atrial fibrillation"),
]


def _expand_abbreviations(text: str) -> str:
    for pattern, replacement in _ABBR_EXPANSIONS:
        text = pattern.sub(replacement, text)
    return text


# Noun-chunk prefixes that indicate the chunk describes a negated or absent
# finding. NegEx catches named entities; this catches the broader class of
# "No X" / "negative X" noun chunks that NegEx misses because en_core_web_sm
# does not recognise clinical terms as named entities.
# These chunks MUST NOT be used as retrieval queries — querying "no ST-segment
# elevation" returns ST-elevation codes at rank 1, which then crowd out the
# true diagnosis in both BM25 and dense retrieval.
_NEGATION_PREFIXES: tuple[str, ...] = (
    "no ",
    "not ",
    "without ",
    "negative ",
    "denies ",
    "ruled out ",
    "no evidence ",
    "absence of ",
    "no history ",
    "no prior ",
    "no family ",
)


def _is_useful_query(text: str) -> bool:
    """Return False for strings that produce only noise as retrieval queries."""
    if len(text) < 4:
        return False
    lower = text.lower().strip()
    if lower in _ADMIN_BLOCKLIST:
        return False
    # Drop negated phrases — querying them retrieves codes for the negated
    # condition, which then crowd out the actual confirmed diagnosis.
    if any(lower.startswith(p) for p in _NEGATION_PREFIXES):
        return False
    tokens = text.split()
    # Single-token lab abbreviations: "HbA1c", "eGFR" — letters + digit + letters
    if len(tokens) == 1 and re.match(r"^[A-Za-z]+\d[A-Za-z\d]*$", tokens[0]):
        return False
    # Single-token medication names
    if len(tokens) == 1 and any(lower.endswith(s) for s in _DRUG_SUFFIXES):
        return False
    # Demographic / temporal phrases: all tokens are numeric, units, or non-clinical
    for token in tokens:
        if not _NUMERIC_RE.match(token) and token.lower() not in _NONCLINICAL_TOKENS:
            return True
    return False


# Negation cue words used for the context-window check in _entity_is_negated.
# This catches sub-chunks of negated phrases that the prefix filter misses —
# e.g. spaCy splits "no family history of sudden cardiac death" into the chunk
# "no family history" (filtered by prefix) AND "sudden cardiac death" (NOT
# filtered by prefix, but clearly negated by the cue word "no" that precedes
# it in the same clause).
_NEGATION_CONTEXT_RE = re.compile(
    r"\b(no|not|without|negative|denies|absent|ruled\s+out|no\s+evidence\s+of)\b",
    re.I,
)


def _entity_is_negated(
    entity_text: str,
    original_note: str,
    negated_spans: list[TextSpan],
) -> bool:
    """Return True if entity_text appears in a negated context in the original note.

    Two checks, applied in order:

    1. NegEx span overlap — catches named entities NegEx flagged (sparse coverage
       with en_core_web_sm, but zero false-positive rate).

    2. Context-window check — looks at the 70 characters before entity_text's
       first occurrence (trimmed to the current clause), and returns True if a
       negation cue word is present. This catches sub-chunks of negated phrases
       that the prefix filter in _is_useful_query cannot see because spaCy
       extracts them as independent noun chunks.
    """
    idx = original_note.lower().find(entity_text.lower())
    if idx < 0:
        return False
    end = idx + len(entity_text)

    # 1. NegEx span overlap
    if any(not (end <= ns.start or idx >= ns.end) for ns in negated_spans):
        return True

    # 2. Context-window: look at the text in the same clause before this entity
    context_start = max(0, idx - 70)
    context = original_note[context_start:idx]
    # Trim to current clause — don't look across sentence/clause boundaries
    for sep in (".", ";", ":"):
        last = context.rfind(sep)
        if last >= 0:
            context = context[last + 1:]
    return bool(_NEGATION_CONTEXT_RE.search(context))


def _extract_query_entities(
    note: str,
    negated_spans: list[TextSpan] | None = None,
) -> list[str]:
    """Return per-entity query strings extracted from note via spaCy NER and noun chunks.

    Running one retrieve() call per entity prevents a dominant condition from
    monopolising the top-k slots in both BM25 and the embedding space.
    Pure numbers, dosages, and sig abbreviations are filtered out because they
    produce spurious high-scoring matches against NIHSS score codes, BMI codes, etc.

    The note is abbreviation-expanded before NLP so that tokens like "MI" and
    "EF" (dropped by the len < 4 guard or the numeric filter) become full
    clinical terms that spaCy can extract as noun chunks.

    Two layers of negation filtering keep ruled-out conditions out of retrieval:
      1. Prefix filter (_is_useful_query): drops any chunk starting with "no ",
         "negative ", etc. — catches the broad "No X" patterns NegEx misses.
      2. NegEx span filter: drops entities whose position in the original note
         overlaps a span flagged by the NegEx pipeline (secondary, sparser).
    """
    doc = _get_spacy_nlp()(_expand_abbreviations(note))
    seen: set[str] = set()
    queries: list[str] = []
    for ent in doc.ents:
        t = ent.text.strip()
        if _is_useful_query(t) and t.lower() not in seen:
            if negated_spans and _entity_is_negated(t, note, negated_spans):
                continue
            seen.add(t.lower())
            queries.append(t)
    for chunk in doc.noun_chunks:
        t = chunk.text.strip()
        if _is_useful_query(t) and t.lower() not in seen:
            if negated_spans and _entity_is_negated(t, note, negated_spans):
                continue
            seen.add(t.lower())
            queries.append(t)
    return queries


def _tokenize(s: str) -> list[str]:
    """Cheap tokenizer for BM25. Real systems use a proper clinical tokenizer."""
    # Strip apostrophes so "Colles'" (ICD description) and "Colles" (clinical note)
    # both tokenize to "colles" and match. Without this, the possessive apostrophe
    # in ICD descriptions creates zero BM25 overlap for the query token.
    return [t for t in s.lower().replace("'", " ").replace(",", " ").replace(".", " ").split() if t]


@lru_cache(maxsize=1)
def _embedding_fn():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-en-v1.5"
    )


def _chroma_collection(code_system: str):
    client = chromadb.PersistentClient(path=str(_CHROMA_DIR))
    return client.get_or_create_collection(
        name=f"{code_system.lower().replace('-', '_')}_codes",
        embedding_function=_embedding_fn(),
    )


def _load_bm25(code_system: str) -> tuple[BM25Okapi, list[dict]]:
    """Load (and cache) the BM25 index. Documents are stored as JSON next to the index."""
    if code_system in _BM25_CACHE:
        return _BM25_CACHE[code_system]

    docs_path = _CHROMA_DIR / f"{code_system}_docs.json"
    if not docs_path.exists():
        raise FileNotFoundError(
            f"BM25 corpus not found at {docs_path}. Run `python -m src.data.build_indices` first."
        )

    with docs_path.open() as f:
        docs = json.load(f)

    tokenized = [_tokenize(d["description"]) for d in docs]
    bm25 = BM25Okapi(tokenized)
    _BM25_CACHE[code_system] = (bm25, docs)
    return bm25, docs


def retrieve(
    note: str,
    code_system: str = "ICD-10-CM",
    top_k_bm25: int = 30,
    top_k_dense: int = 30,
    final_k: int = 20,
) -> list[CodeCandidate]:
    """Hybrid retrieval with reciprocal rank fusion.

    Returns up to ``final_k`` candidates merged from BM25 and dense retrieval.
    """
    # --- BM25 ---
    bm25, docs = _load_bm25(code_system)
    tokenized_query = _tokenize(note)
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ranked = sorted(
        enumerate(bm25_scores), key=lambda x: x[1], reverse=True
    )[:top_k_bm25]
    bm25_results = [(docs[i]["code"], docs[i]["description"], score) for i, score in bm25_ranked]

    # --- Dense ---
    collection = _chroma_collection(code_system)
    dense_results_raw = collection.query(query_texts=[note], n_results=top_k_dense)
    dense_results = list(
        zip(
            dense_results_raw["ids"][0],
            dense_results_raw["documents"][0],
            dense_results_raw["distances"][0],
        )
    )
    # Convert distance to similarity-ish score; lower distance = more similar
    dense_results = [(code, desc, 1.0 / (1.0 + dist)) for code, desc, dist in dense_results]

    # --- Reciprocal Rank Fusion ---
    # RRF score for a doc = sum over retrievers of 1 / (k + rank). k=60 is the
    # standard constant from the original RRF paper (Cormack et al. 2009).
    K = 60
    rrf_scores: dict[str, float] = {}
    descriptions: dict[str, str] = {}

    for rank, (code, desc, _) in enumerate(bm25_results):
        rrf_scores[code] = rrf_scores.get(code, 0.0) + 1.0 / (K + rank)
        descriptions[code] = desc
    for rank, (code, desc, _) in enumerate(dense_results):
        rrf_scores[code] = rrf_scores.get(code, 0.0) + 1.0 / (K + rank)
        descriptions[code] = desc

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:final_k]

    return [
        CodeCandidate(
            code=code,
            description=descriptions[code],
            code_system=code_system,  # type: ignore[arg-type]
            retrieval_score=score,
            retrieval_method="rrf",
        )
        for code, score in fused
    ]


# Maps common clinical shorthand to the ICD-preferred phrasing that the
# dense model recognises. bge-small-en-v1.5 ranks I10 ("Essential (primary)
# hypertension") poorly for the bare query "hypertension" because codes whose
# descriptions START with the word score higher. Adding the ICD phrasing as a
# second query retrieves I10 at rank 1 with distance 0.05 vs 0.18 for runner-up.
_QUERY_SYNONYMS: dict[str, list[str]] = {
    "hypertension": ["essential hypertension"],
    "diabetes": ["diabetes mellitus"],
    "heart failure": ["heart failure", "cardiac failure"],
    # Bare "copd" gets the generic phrase AND the unspecified description so
    # J44.9 always enters the candidate set. Without the second term, J44.89
    # ("other chronic obstructive pulmonary disease") crowds out J44.9 in both
    # BM25 and dense because its description is a strict superset of tokens.
    "copd": [
        "chronic obstructive pulmonary disease",
        "chronic obstructive pulmonary disease, unspecified",
    ],
    # spaCy splits "acute exacerbation of COPD" into the chunk "acute
    # exacerbation" + the entity "COPD". The chunk alone must resolve to the
    # J44.1 description so the reranker sees it alongside J44.9 and can pick
    # the right one based on the full note context.
    "acute exacerbation": [
        "chronic obstructive pulmonary disease with (acute) exacerbation",
    ],
    # Stable-COPD noun chunks extracted from notes like "stable COPD, GOLD B"
    # or "quarterly COPD follow-up". Both map to J44.9 (unspecified = stable).
    "stable copd": [
        "chronic obstructive pulmonary disease, unspecified",
    ],
    "quarterly copd follow-up": [
        "chronic obstructive pulmonary disease, unspecified",
    ],
    "copd follow-up": [
        "chronic obstructive pulmonary disease, unspecified",
    ],
    "afib": ["atrial fibrillation"],
    "ckd": ["chronic kidney disease"],
    "cad": ["coronary artery disease"],
    "dvt": ["deep vein thrombosis"],
    "pe": ["pulmonary embolism"],
    "uti": ["urinary tract infection"],
    # Ejection fraction (after "EF" abbreviation expansion) → heart failure
    # codes. Without this, "ejection fraction 45 %" produces no useful BM25
    # match because the numeric token dominates and ICD descriptions say
    # "reduced ejection fraction" not "ejection fraction 45".
    "ejection fraction": [
        "heart failure",
        "systolic heart failure",
        "diastolic heart failure",
    ],
    # Terminology mismatches where the clinical term used in notes does not
    # appear in the ICD-10 description, so BM25 finds zero token overlap.
    # Dense retrieval alone is insufficient when other queries (SAH, LP, ACS)
    # produce higher-scoring candidates.
    #
    # "costochondritis" → ICD M94.0 is "Chondrocostal junction syndrome [Tietze]"
    "costochondritis": ["chondrocostal junction syndrome"],
    # "migraines" (plural) → ICD descriptions use singular "migraine".
    # All G43.9x codes tie on ["migraine", "unspecified"] in BM25, so the
    # full description string is used to uniquely pin G43.909 (not intractable,
    # without status migrainosus) to rank 1 and push G43.911/G43.919 down.
    "migraines": [
        "migraine, unspecified, not intractable, without status migrainosus",
    ],
    "prior severe migraines": [
        "migraine, unspecified, not intractable, without status migrainosus",
    ],
    # "vasovagal syncope" → ICD R55 is "Syncope and collapse" (zero token overlap)
    "vasovagal syncope": ["syncope and collapse"],
    # Breast cancer: emit both the active-malignancy description and the
    # personal-history-of description so Z85.3 enters the candidate set for
    # surveillance visits where the cancer is no longer active.
    "breast cancer": [
        "malignant neoplasm of breast",
        "personal history of malignant neoplasm of breast",
    ],
    "left breast cancer": [
        "malignant neoplasm of breast",
        "personal history of malignant neoplasm of breast",
    ],
    # Cancer surveillance / post-treatment follow-up encounter codes (Z08/Z09).
    # Plain "surveillance" and "annual surveillance" retrieve contraceptive
    # surveillance codes (Z30.4x) at rank 1 — these synonyms override that.
    "oncology surveillance": [
        "encounter for follow-up examination after completed treatment for malignant neoplasm",
    ],
    "annual oncology surveillance visit": [
        "encounter for follow-up examination after completed treatment for malignant neoplasm",
    ],
    "annual surveillance": [
        "encounter for follow-up examination after completed treatment for malignant neoplasm",
    ],
    "cancer surveillance": [
        "encounter for follow-up examination after completed treatment for malignant neoplasm",
        "personal history of malignant neoplasm",
    ],
    # Rotator cuff: notes use "impingement tests" which retrieves M75.42
    # (Impingement syndrome) instead of M75.12 (Rotator cuff syndrome).
    # Mapping the diagnosis phrase to the exact ICD description overrides
    # the impingement-driven dense retrieval.
    "rotator cuff syndrome": [
        "rotator cuff syndrome, left shoulder",
        "rotator cuff syndrome, right shoulder",
        "rotator cuff syndrome, unspecified shoulder",
    ],
    "left shoulder rotator cuff syndrome": [
        "rotator cuff syndrome, left shoulder",
    ],
    "right shoulder rotator cuff syndrome": [
        "rotator cuff syndrome, right shoulder",
    ],
    # Primary OA: "left knee pain" (M25.562) outranks M17.12 in retrieval.
    # Issuing separate laterality-specific OA queries brings M17.1x into
    # the top-10 so the LLM can pick the correct side.
    "primary osteoarthritis": [
        "unilateral primary osteoarthritis, left knee",
        "unilateral primary osteoarthritis, right knee",
        "bilateral primary osteoarthritis of knee",
    ],
    # MDD severity: F33.2 (recurrent severe without psychotic features) falls
    # just outside the top-20 because F33.0/F33.1/F33.9 crowd it out. Adding
    # this synonym ensures F33.2 enters candidates so the LLM can use the
    # PHQ-9 score to select the correct severity code.
    "depressive symptoms": [
        "major depressive disorder, recurrent severe without psychotic features",
    ],
    # Ankle sprain: "sprain" alone retrieves sequela and follow-up codes first.
    # The initial-encounter description surfaces S93.40xA at rank 1 in BM25.
    "sprain": [
        "sprain of unspecified ligament of ankle, initial encounter",
    ],
    # Anemia in CKD: spaCy extracts bare "anemia" (the prepositional phrase
    # "of CKD" is stripped as a modifier), so "anemia of chronic kidney
    # disease" synonym never fires. Mapping the bare word ensures D63.1
    # enters the candidate set. D64.9 (unspecified) otherwise wins because
    # it has no etiology-specific qualifier to lower its BM25 rank.
    "anemia": [
        "anemia in chronic kidney disease",
        "anemia in other chronic diseases classified elsewhere",
        "iron deficiency anemia, unspecified",
    ],
    "anemia of chronic kidney disease": [
        "anemia in chronic kidney disease",
    ],
    "anemia of ckd": [
        "anemia in chronic kidney disease",
    ],
    # Secondary hyperparathyroidism: "secondary hyperparathyroidism" retrieves
    # E21.1 (not elsewhere classified = non-renal) over N25.81 (renal origin)
    # because E21.1 is a shorter description with higher BM25 density. Adding
    # the renal-origin phrasing ensures N25.81 enters the candidate set so the
    # LLM can select the etiology-correct code based on documented CKD context.
    "secondary hyperparathyroidism": [
        "secondary hyperparathyroidism of renal origin",
    ],
    "renal origin": [
        "secondary hyperparathyroidism of renal origin",
    ],
    # CKD stage: after abbreviation expansion "CKD stage 4" → "chronic kidney
    # disease stage 4", but spaCy strips the trailing numeric modifier so the
    # extracted chunk is "chronic kidney disease stage" (no digit). Issuing
    # explicit stage-specific synonym queries ensures all N18.3x–N18.5 codes
    # enter the candidate set and the LLM can select the documented stage.
    "chronic kidney disease stage": [
        "chronic kidney disease, stage 4 (severe)",
        "chronic kidney disease, stage 3b",
        "chronic kidney disease, stage 3a",
        "chronic kidney disease, stage 5",
        "chronic kidney disease, stage 2 (mild)",
    ],
    # Colles fracture: "a Colles fracture" is the extracted entity. The 'A'
    # article is low-IDF noise; adding the full right-radius description gives
    # S52.531A a direct BM25 rank-1 boost over the sequela and left-side variants.
    "a colles fracture": [
        "Colles fracture of right radius, initial encounter for closed fracture",
    ],
}


def retrieve_decomposed(
    note: str,
    code_system: str = "ICD-10-CM",
    top_k_bm25: int = 30,
    top_k_dense: int = 30,
    final_k: int = 20,
    negated_spans: list[TextSpan] | None = None,
) -> list[CodeCandidate]:
    """Hybrid retrieval with per-entity query decomposition.

    Runs retrieve() for the full note and for each clinical entity/noun chunk
    extracted by spaCy. Candidate sets are unioned by taking the max RRF score
    per code across all queries, then trimmed to final_k. This ensures secondary
    conditions (e.g. hypertension) are not crowded out by dominant ones
    (e.g. type 2 diabetes mellitus) in either BM25 or the embedding space.

    Each extracted entity is also expanded through _QUERY_SYNONYMS to bridge
    gaps between clinical shorthand and ICD-preferred terminology.

    negated_spans: NegEx-detected negated entity spans from the pipeline. Passed
    to _extract_query_entities so NegEx-flagged entities are excluded as queries
    in addition to the prefix-based filter that handles the broader "No X" class.
    """
    entities = _extract_query_entities(note, negated_spans=negated_spans)
    expanded: list[str] = []
    seen: set[str] = set()
    for e in entities:
        if e.lower() not in seen:
            seen.add(e.lower())
            expanded.append(e)
        for syn in _QUERY_SYNONYMS.get(e.lower(), []):
            if syn.lower() not in seen:
                seen.add(syn.lower())
                expanded.append(syn)

    queries = [note] + expanded

    merged: dict[str, CodeCandidate] = {}
    for query in queries:
        for candidate in retrieve(query, code_system, top_k_bm25, top_k_dense, final_k):
            existing = merged.get(candidate.code)
            if existing is None or candidate.retrieval_score > existing.retrieval_score:
                merged[candidate.code] = candidate

    return sorted(merged.values(), key=lambda c: c.retrieval_score, reverse=True)[:final_k]
