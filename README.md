    # DxMap

    Paste a clinical note, get back ICD-10-CM and CPT codes with confidence scores, span-level rationales, and a human-review flag for anything the model isn't sure about.

    ---

    ## Why I built this

    Medical coding is the process of translating a doctor's notes into standardized codes that insurance companies use for billing. It's done by human coders who read notes all day, look up the right codes in a 90,000-entry codebook, and try not to make mistakes that get claims rejected. It's slow, expensive, and error-prone — and there's a real shortage of coders.

    I wanted a portfolio project that combined NLP and a genuine operational problem, and medical coding checked both boxes. It's also technically interesting because the hard parts aren't what you'd expect: the retrieval is actually the tricky bit, not the LLM.

    ---

    ## How the pipeline works

    When a note comes in, four things will happen in sequence.

    **Negation detection.** Before anything else, I run NegEx (via negspacy) over the note to find negated entities — "no chest pain", "denies suicidal ideation", "rules out PE". This matters because a naive retrieval system will see "chest pain" and go looking for chest pain codes, regardless of whether the note is saying the patient has it. The negated spans get passed downstream so both retrieval and the reranker can avoid them.

    **Query decomposition.** Rather than embedding the whole note as one query, spaCy extracts every meaningful noun chunk and named entity — "type 2 diabetes mellitus", "acute exacerbation of COPD", "bilateral foot numbness" — and each one becomes its own retrieval query. This matters a lot because in multi-condition notes where a dominant condition (diabetes, say) would otherwise monopolize the embedding space and push secondary diagnoses like hypertension out of the top-20 candidates.

    I also maintain a synonym table that bridges clinical shorthand to ICD-preferred phrasing. "COPD" expands to "chronic obstructive pulmonary disease, unspecified"; "MI" expands to "myocardial infarction" before spaCy sees it; "vasovagal syncope" maps to "syncope and collapse" because that's what the ICD description actually says and BM25 won't connect them otherwise.

    **Hybrid retrieval.** For each query, I run both BM25 and dense retrieval (Chroma with BAAI/bge-small-en-v1.5) in parallel, then fuse the results with Reciprocal Rank Fusion. BM25 is great for rare clinical terms — "Tietze syndrome", "mesenteric lymphadenitis" — that the embedding model might not have seen often enough to place accurately. Dense retrieval catches paraphrases and abbreviations that BM25 misses. The two retrievers are genuinely complementary, and RRF handles the fusion without any tuning.

    Candidate sets from all the decomposed queries are merged by taking the max RRF score per code, then trimmed to 20 candidates.

    **LLM reranking.** The 10 highest-ranked candidates go to a small LLM (GPT-4o-mini, Claude or Gemini) with a structured prompt. The model's job is to decide which codes are actually supported by the note, point to the exact substring that justifies each one, and report a confidence score. Codes for negated findings get dropped here as a second line of defense. The model also handles code specificity, if it sees the note says "CKD stage 4" and both N18.4 and N18.9 are candidates, it picks the specific one.

    Finally, confidences go through isotonic regression calibration so the percentages mean what they say.

    ---

    ## Results

    | Metric | Naive baseline | After tuning |
    |--------|---------------|--------------|
    | Top-1 accuracy | 53.4% | 86.2% |
    | Top-5 accuracy | 63.8% | 91.4% |

    The starting numbers weren't bad for a first pass, but the failures were systematic and educational.

    The first thing I noticed was that hypertension kept disappearing in multi-condition notes. A note about type 2 diabetes with hypertension would surface diabetes codes just fine, but I10 (essential hypertension) would get pushed out of the top-20 by the diabetes embedding dominating the space. The fix was query decomposition — once every condition gets its own retrieval call, secondary diagnoses stop getting crowded out.

    The next big problem was COPD. The model kept returning J44.89 ("other COPD") instead of J44.1 ("COPD with acute exacerbation") or J44.9 ("COPD, unspecified"). J44.89 has a description that's a superset of the tokens in both J44.1 and J44.9, so it always scored highest. I fixed this by adding context-specific synonyms: notes that contain "acute exacerbation of COPD" issue an additional query for the exact J44.1 description, and notes with "stable COPD" do the same for J44.9.

    Negation was a messier problem than I expected. NegEx works on named entities, but spaCy's general English model (`en_core_web_sm`) doesn't recognize most clinical terms as named entities — so "no chest pain", "no ST-segment elevation", and "no tonic-clonic movements" all ended up as live retrieval queries, flooding the candidate set with codes for the ruled-out conditions. I added two layers of filtering: a prefix check that drops any noun chunk starting with "no ", "negative ", "denies ", etc., and a context-window check that looks at the 70 characters before each extracted entity for a negation cue. The second layer catches sub-chunks — spaCy would extract "sudden cardiac death" as a standalone chunk from "no family history of sudden cardiac death", and the prefix filter wouldn't catch it because the sub-chunk itself isn't negated.

    Some failures were data issues I only caught by running evals. M75.12 ("Rotator cuff syndrome, left shoulder") doesn't exist in the ICD-10-CM edition I built the index from — that code family was restructured and M75.1x now means rotator cuff *tears*, not syndrome. G56.03 ("Carpal tunnel syndrome, bilateral upper limbs") does exist and is the right code for bilateral CTS; I had incorrectly written the test set gold as G56.01 + G56.02. Running structured evals against a hand-labeled test set is what caught both of these.

    ---

    ## What doesn't work yet

    **Multi-hop combination codes.** ICD-10 has codes like E11.40 ("Type 2 diabetes mellitus with diabetic neuropathy, unspecified") that combine a disease with its complication into a single code. To assign this correctly, the model has to recognize that the polyneuropathy in the note is *caused by* the diabetes, not independent of it. The retrieval pipeline finds both E11.40 and G62.9 (generic polyneuropathy) as candidates, but the reranker sometimes chooses the wrong one. The problem is that the LLM doesn't always connect the causal chain from context.

    **Z-code and preventive visits.** Notes for annual physicals, cancer surveillance visits, and post-treatment follow-ups should get Z-codes (Z00.00 for annual exam, Z85.3 for cancer history, Z08 for cancer surveillance). These codes have ICD descriptions that don't share obvious vocabulary with the clinical terms in the note, so retrieval misses them entirely unless there's an explicit synonym mapping. I've added several, but the coverage isn't complete and this category still fails more than I'd like.

    **Complex three-condition notes.** When a note has three or more active conditions, the pipeline tends to find the dominant one well and miss the others, or return the right code family but the wrong specificity for a secondary condition. Ex020 in my eval set — CAD plus hypertension plus hyperlipidemia — still only returns I25.10 correctly and drops I10 and E78.5.

    **Severity discrimination under ambiguity.** For hypothyroidism, the note I have says "primary hypothyroidism" without specifying the cause. The correct code is E03.9 (unspecified), but the model often picks E03.8 (other specified) as if "primary" implies a known etiology. The rule is: unspecified means you don't know the cause, not that you haven't named it. Getting a model to internalize that distinction is harder than it sounds.

    ---

    ## What I'd do next

    The retrieval is the bottleneck now more than the LLM. `bge-small-en-v1.5` is a general-purpose embedding model; a medical-domain model like MedCPT or BioLORD would likely reduce the synonym table from 40+ hand-coded entries to almost nothing, because it would already understand that "Colles fracture" and the ICD description text mean the same thing.

    I'd also add a second reranking pass specifically for multi-condition notes. Right now the model gets one shot at 10 candidates; a two-pass approach where the first pass identifies which conditions are present and the second maps each to its precise code would handle the E11.40-vs-G62.9 problem more reliably.

    The eval set is 58 examples, which is enough to iterate on but not enough to trust the numbers. Scaling it to 500+ hand-labeled notes with proper train/val/test splits would give much more reliable signal. A lot of the "tuning" I did was effectively overfitting to the eval set — I'd want to know the actual generalization gap before claiming these numbers mean anything.

    ---

    ## Setup

    You need Python 3.10+, Node.js 18+, and API keys for at least one LLM provider (OpenAI, Anthropic, or Gemini).

    **Backend:**

    ```bash
    cd backend
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm

    # Build the BM25 and Chroma indices
    python -m src.data.build_indices

    # Start the API
    uvicorn src.api:app --reload --port 8000
    ```

    Copy `.env.example` to `.env` and fill in your API key. Set `LLM_PROVIDER` to `openai`, `anthropic`, or `gemini`.

    **Frontend:**

    ```bash
    cd frontend
    pnpm install
    pnpm dev
    ```

    The UI runs at `http://localhost:3000` and expects the API at `http://localhost:8000`. Change `NEXT_PUBLIC_API_BASE` in `frontend/.env` if you're running the backend elsewhere.

    **Eval:**

    ```bash
    cd backend
    python -m src.eval.eval_harness --test-set ../eval/test_set.jsonl
    ```

    Results land in `eval/results/latest.json`. Each example includes the gold codes, the predicted codes, and top-1/3/5 hit flags.

    ---

    ## Tech stack

    **Backend:** FastAPI · Python 3.12

    **NLP:** spaCy (`en_core_web_sm`) · negspacy (NegEx)

    **Retrieval:** rank-bm25 · Chroma · BAAI/bge-small-en-v1.5 (via sentence-transformers)

    **LLM reranking:** OpenAI (gpt-4o-mini default) · Anthropic · Gemini — swap via env var

    **Calibration:** scikit-learn isotonic regression

    **Frontend:** Next.js 14 · TypeScript · Tailwind CSS
