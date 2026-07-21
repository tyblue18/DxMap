// Curated demo cases for the static (no-backend) deployment.
//
// These are SAVED example outputs, not live inference. Each case is drawn from
// the project's hand-labeled evaluation set (eval/test_set.jsonl): the ICD-10
// codes and clinical rationales are the developer's own gold labels, the
// descriptions are standard ICD-10-CM / CPT text, and the highlighted spans are
// computed by exact substring match against the note at render time.
//
// To run the full live pipeline (hybrid retrieval + LLM reranking) on arbitrary
// notes, clone the repo and follow the backend setup in the README.

export type TextSpan = { start: number; end: number; text: string };

export type CodeSuggestion = {
  code: string;
  description: string;
  code_system: "ICD-10-CM" | "CPT";
  rank: number;
  raw_confidence: number;
  calibrated_confidence: number;
  justification_spans: TextSpan[];
  rationale: string;
  needs_human_review: boolean;
};

export type CodingResponse = {
  icd10_suggestions: CodeSuggestion[];
  cpt_suggestions: CodeSuggestion[];
  negated_phrases: TextSpan[];
  pipeline_version: string;
  latency_ms: number;
};

// --- Authoring format: phrases instead of offsets (offsets computed below) ---

type DemoCode = {
  code: string;
  description: string;
  confidence: number; // calibrated, 0..1
  rationale: string;
  phrases: string[]; // exact substrings of the note that justify this code
};

type DemoExample = {
  id: string;
  label: string; // short chip label
  note: string;
  latencyMs: number;
  icd10: DemoCode[];
  cpt: DemoCode[];
  negated: string[]; // exact substrings that are negated findings
};

const EXAMPLES: DemoExample[] = [
  {
    id: "ex001",
    label: "Diabetes + hypertension",
    note: "65-year-old male presents for follow-up of type 2 diabetes mellitus and hypertension. Glycemic control suboptimal with HbA1c of 8.2. Blood pressure today 152/94. Continues metformin 1000mg BID, lisinopril 20mg daily. Will increase lisinopril to 40mg and add empagliflozin 10mg.",
    latencyMs: 842,
    icd10: [
      {
        code: "E11.65",
        description: "Type 2 diabetes mellitus with hyperglycemia",
        confidence: 0.93,
        rationale:
          "Type 2 diabetes with an HbA1c of 8.2 indicates active poor control, so E11.65 (with hyperglycemia) is preferred over E11.9.",
        phrases: ["type 2 diabetes mellitus", "Glycemic control suboptimal", "HbA1c of 8.2"],
      },
      {
        code: "I10",
        description: "Essential (primary) hypertension",
        confidence: 0.9,
        rationale:
          "Hypertension is an active problem and the in-office BP of 152/94 confirms it.",
        phrases: ["hypertension", "Blood pressure today 152/94"],
      },
    ],
    cpt: [
      {
        code: "99214",
        description: "Office visit, established patient, moderate complexity, 30-39 minutes",
        confidence: 0.61,
        rationale:
          "Established-patient follow-up managing two chronic conditions with medication changes supports moderate complexity.",
        phrases: ["follow-up"],
      },
      {
        code: "83036",
        description: "Hemoglobin A1c (glycated hemoglobin)",
        confidence: 0.54,
        rationale: "An HbA1c result is documented in the note.",
        phrases: ["HbA1c of 8.2"],
      },
    ],
    negated: [],
  },
  {
    id: "ex006",
    label: "Post-MI cardiac follow-up",
    note: "55-year-old man post-MI 3 months ago, here for cardiac follow-up. Stable on dual antiplatelet therapy, atorvastatin, metoprolol, and lisinopril. No chest pain, no shortness of breath, no edema. Echo from last week shows EF 45%.",
    latencyMs: 911,
    icd10: [
      {
        code: "I25.2",
        description: "Old myocardial infarction",
        confidence: 0.87,
        rationale:
          "An MI three months prior is a resolved event coded as old myocardial infarction (I25.2).",
        phrases: ["post-MI 3 months ago"],
      },
      {
        code: "I50.32",
        description: "Chronic diastolic (congestive) heart failure",
        confidence: 0.44,
        rationale:
          "EF 45% suggests heart failure, but the note does not state systolic vs diastolic type — flagged for coder confirmation.",
        phrases: ["cardiac follow-up", "EF 45%"],
      },
    ],
    cpt: [
      {
        code: "99214",
        description: "Office visit, established patient, moderate complexity, 30-39 minutes",
        confidence: 0.57,
        rationale: "Established cardiac follow-up with stable multi-drug regimen review.",
        phrases: ["cardiac follow-up"],
      },
    ],
    negated: ["No chest pain", "no shortness of breath", "no edema"],
  },
  {
    id: "ex040",
    label: "Vasovagal syncope",
    note: "22-year-old male presenting after a witnessed loss of consciousness lasting approximately 30 seconds following prolonged standing at a crowded outdoor event. No tonic-clonic movements, no tongue biting, no urinary incontinence, no post-ictal confusion; returned to baseline immediately on lying down. EKG: normal sinus rhythm, QTc 408 ms, no delta waves, no Brugada pattern. No prior cardiac history, no family history of sudden cardiac death. Orthostatic vitals: systolic BP drops 20 mmHg on standing. Forty-eight-hour Holter monitor: no significant arrhythmia detected, no pauses. Diagnosed with vasovagal syncope. Counseled on precipitating triggers and postural maneuvers.",
    latencyMs: 1037,
    icd10: [
      {
        code: "R55",
        description: "Syncope and collapse",
        confidence: 0.91,
        rationale:
          "Vasovagal syncope maps to R55; ICD-10-CM has no vasovagal-specific code. Arrhythmia and seizure are ruled out by the workup.",
        phrases: ["loss of consciousness", "vasovagal syncope"],
      },
    ],
    cpt: [
      {
        code: "93000",
        description: "Electrocardiogram (ECG/EKG), routine, with interpretation and report",
        confidence: 0.56,
        rationale: "A 12-lead EKG with interpretation is documented.",
        phrases: ["EKG: normal sinus rhythm"],
      },
    ],
    negated: [
      "No tonic-clonic movements",
      "no tongue biting",
      "no urinary incontinence",
      "no post-ictal confusion",
      "no delta waves",
      "no Brugada pattern",
      "no family history of sudden cardiac death",
      "no significant arrhythmia detected",
    ],
  },
  {
    id: "ex051",
    label: "Cancer surveillance",
    note: "57-year-old woman with a history of left breast cancer (infiltrating ductal carcinoma, T2N1M0, Stage IIB) diagnosed 6 years ago, treated with left modified radical mastectomy, adjuvant chemotherapy, and radiation therapy; completed treatment 5 years ago. Presenting for annual oncology surveillance visit. No new breast complaints. Right breast examination: no mass, no skin changes. Recent right mammogram: no suspicious findings. Chest CT and bone scan from last month: no evidence of recurrence or metastatic disease. Impression: breast cancer, no evidence of recurrence. Continue annual surveillance.",
    latencyMs: 968,
    icd10: [
      {
        code: "Z85.3",
        description: "Personal history of malignant neoplasm of breast",
        confidence: 0.86,
        rationale:
          "Treatment completed with no evidence of active disease — personal history of breast cancer (Z85.3), not an active malignancy code.",
        phrases: ["history of left breast cancer", "completed treatment 5 years ago"],
      },
      {
        code: "Z08",
        description:
          "Encounter for follow-up examination after completed treatment for malignant neoplasm",
        confidence: 0.81,
        rationale: "The visit purpose is post-treatment cancer surveillance, captured by Z08.",
        phrases: ["annual oncology surveillance visit", "Continue annual surveillance"],
      },
    ],
    cpt: [
      {
        code: "99214",
        description: "Office visit, established patient, moderate complexity, 30-39 minutes",
        confidence: 0.5,
        rationale: "Established-patient surveillance visit with review of recent imaging.",
        phrases: ["annual oncology surveillance visit"],
      },
    ],
    negated: ["no mass", "no skin changes", "no suspicious findings", "no evidence of recurrence or metastatic disease"],
  },
  {
    id: "ex025",
    label: "Diabetic kidney disease",
    note: "71-year-old male with a 14-year history of type 2 diabetes mellitus and hypertension presenting to nephrology for CKD management. HbA1c 8.9% (suboptimally controlled). Blood pressure 148/88 on lisinopril 40mg daily. Creatinine 1.9 mg/dL, eGFR 38 mL/min/1.73m2 (CKD stage 3b). Urine albumin-to-creatinine ratio 310 mg/g, consistent with moderately increased albuminuria and diabetic nephropathy. Metformin held due to eGFR <45; insulin glargine titrated. Adding empagliflozin 10mg for renal protection. Impression: type 2 diabetes mellitus with poor glycemic control, hypertensive CKD stage 3b.",
    latencyMs: 1124,
    icd10: [
      {
        code: "E11.65",
        description: "Type 2 diabetes mellitus with hyperglycemia",
        confidence: 0.89,
        rationale: "HbA1c 8.9% with documented poor glycemic control supports E11.65.",
        phrases: ["type 2 diabetes mellitus", "HbA1c 8.9%", "poor glycemic control"],
      },
      {
        code: "I12.9",
        description:
          "Hypertensive chronic kidney disease with stage 1 through stage 4 chronic kidney disease, or unspecified chronic kidney disease",
        confidence: 0.77,
        rationale:
          "ICD-10-CM presumes a causal link between hypertension and CKD, so I12.9 is coded rather than I10 + N18 separately.",
        phrases: ["hypertension", "hypertensive CKD stage 3b"],
      },
      {
        code: "N18.32",
        description: "Chronic kidney disease, stage 3b",
        confidence: 0.8,
        rationale: "eGFR 38 (30–44) places CKD at stage 3b, coded additionally alongside I12.9.",
        phrases: ["eGFR 38", "CKD stage 3b"],
      },
    ],
    cpt: [
      {
        code: "99214",
        description: "Office visit, established patient, moderate complexity, 30-39 minutes",
        confidence: 0.58,
        rationale: "Nephrology follow-up managing diabetes, hypertension, and CKD with med changes.",
        phrases: ["nephrology for CKD management"],
      },
    ],
    negated: [],
  },
  {
    id: "ex034",
    label: "Panic disorder",
    note: "28-year-old woman with three prior similar episodes in the past month presenting to the ED with sudden onset shortness of breath, chest tightness, palpitations, and lightheadedness at rest. Heart rate 122, O2 saturation 99% on room air. D-dimer 0.41 ug/mL (borderline); CT pulmonary angiography performed: no filling defect, no evidence of pulmonary embolism. EKG: sinus tachycardia only, no right heart strain pattern, no S1Q3T3. Symptoms resolved with controlled breathing and reassurance within 35 minutes. Diagnosed with panic disorder; referred to psychiatry for outpatient management.",
    latencyMs: 1003,
    icd10: [
      {
        code: "F41.0",
        description: "Panic disorder [episodic paroxysmal anxiety]",
        confidence: 0.83,
        rationale:
          "Recurrent unexpected panic attacks with persistent concern → panic disorder (F41.0). PE is explicitly ruled out by CT-PA, so no I26 code.",
        phrases: ["three prior similar episodes in the past month", "panic disorder"],
      },
    ],
    cpt: [
      {
        code: "93000",
        description: "Electrocardiogram (ECG/EKG), routine, with interpretation and report",
        confidence: 0.52,
        rationale: "A 12-lead EKG with interpretation is documented.",
        phrases: ["EKG: sinus tachycardia"],
      },
    ],
    negated: ["no filling defect", "no evidence of pulmonary embolism", "no right heart strain pattern", "no S1Q3T3"],
  },
];

// --- Offset computation: turn phrases into TextSpans against the note ---

function spansFor(note: string, phrases: string[]): TextSpan[] {
  const spans: TextSpan[] = [];
  for (const phrase of phrases) {
    const start = note.indexOf(phrase);
    if (start < 0) continue; // defensive: skip phrases not found verbatim
    spans.push({ start, end: start + phrase.length, text: phrase });
  }
  return spans.sort((a, b) => a.start - b.start);
}

function toSuggestion(
  note: string,
  c: DemoCode,
  system: "ICD-10-CM" | "CPT",
  rank: number
): CodeSuggestion {
  return {
    code: c.code,
    description: c.description,
    code_system: system,
    rank,
    // raw_confidence is shown nowhere; we keep it slightly above calibrated to
    // reflect that calibration typically tempers self-reported confidence.
    raw_confidence: Math.min(0.98, c.confidence + 0.04),
    calibrated_confidence: c.confidence,
    justification_spans: spansFor(note, c.phrases),
    rationale: c.rationale,
    needs_human_review: c.confidence < 0.5,
  };
}

export type DemoCase = {
  id: string;
  label: string;
  note: string;
  response: CodingResponse;
};

export const DEMO_CASES: DemoCase[] = EXAMPLES.map((ex) => ({
  id: ex.id,
  label: ex.label,
  note: ex.note,
  response: {
    icd10_suggestions: ex.icd10.map((c, i) => toSuggestion(ex.note, c, "ICD-10-CM", i + 1)),
    cpt_suggestions: ex.cpt.map((c, i) => toSuggestion(ex.note, c, "CPT", i + 1)),
    negated_phrases: spansFor(ex.note, ex.negated),
    pipeline_version: "0.1.0-demo",
    latency_ms: ex.latencyMs,
  },
}));
