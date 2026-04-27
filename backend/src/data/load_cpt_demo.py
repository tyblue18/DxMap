"""Load a small, curated subset of CPT codes for demo purposes.

LICENSING NOTE — read this carefully:

  CPT (Current Procedural Terminology) is a copyrighted code set owned by the
  American Medical Association (AMA). The full code set cannot be redistributed
  without an AMA license.

  This file contains a *small, hand-curated subset* of commonly used CPT codes
  with their descriptions, included for educational and portfolio demonstration
  purposes only. This is NOT a substitute for a licensed CPT data feed and
  cannot be used in any production system.

  For a production system, you would obtain a CPT data file via an AMA license
  or use a vendor that has licensed the codes (e.g., Optum, 3M, several EHR
  vendors). The pipeline architecture supports swapping this file out for the
  full licensed corpus with no code changes — only the data file changes.

The codes below are drawn from the CMS "Most Frequently Reported CPT" public
listings and are formatted from publicly available billing examples.
"""

from __future__ import annotations

# Curated subset: top E&M, common procedures, and common imaging codes.
# Descriptions are paraphrased from public CMS billing guidance, not lifted from
# the AMA CPT manual.
_CPT_DEMO: list[dict[str, str]] = [
    # Evaluation and management — office visits
    {"code": "99202", "description": "Office visit, new patient, low-complexity, 15-29 minutes"},
    {"code": "99203", "description": "Office visit, new patient, moderate-complexity, 30-44 minutes"},
    {"code": "99204", "description": "Office visit, new patient, moderate-to-high complexity, 45-59 minutes"},
    {"code": "99205", "description": "Office visit, new patient, high-complexity, 60-74 minutes"},
    {"code": "99212", "description": "Office visit, established patient, straightforward, 10-19 minutes"},
    {"code": "99213", "description": "Office visit, established patient, low complexity, 20-29 minutes"},
    {"code": "99214", "description": "Office visit, established patient, moderate complexity, 30-39 minutes"},
    {"code": "99215", "description": "Office visit, established patient, high complexity, 40-54 minutes"},
    # Preventive
    {"code": "99395", "description": "Periodic preventive medicine evaluation, established patient, age 18-39"},
    {"code": "99396", "description": "Periodic preventive medicine evaluation, established patient, age 40-64"},
    # Imaging
    {"code": "70450", "description": "CT scan, head or brain, without contrast"},
    {"code": "70551", "description": "MRI, brain, without contrast"},
    {"code": "70553", "description": "MRI, brain, without and with contrast"},
    {"code": "71045", "description": "Chest X-ray, single view"},
    {"code": "71046", "description": "Chest X-ray, two views"},
    {"code": "73721", "description": "MRI, lower extremity joint, without contrast"},
    {"code": "76700", "description": "Abdominal ultrasound, complete"},
    # Lab
    {"code": "80053", "description": "Comprehensive metabolic panel"},
    {"code": "80061", "description": "Lipid panel"},
    {"code": "85025", "description": "Complete blood count (CBC) with automated differential"},
    {"code": "83036", "description": "Hemoglobin A1c (glycated hemoglobin)"},
    {"code": "84443", "description": "Thyroid stimulating hormone (TSH)"},
    # Cardiology
    {"code": "93000", "description": "Electrocardiogram (ECG/EKG), routine, with interpretation and report"},
    {"code": "93306", "description": "Echocardiogram, complete, with Doppler"},
    {"code": "93880", "description": "Carotid artery duplex ultrasound, bilateral"},
    # Procedures — minor surgery
    {"code": "10060", "description": "Incision and drainage of abscess, simple"},
    {"code": "11042", "description": "Debridement, subcutaneous tissue, first 20 sq cm"},
    {"code": "12001", "description": "Simple repair of superficial wounds, 2.5 cm or less"},
    # GI
    {"code": "43239", "description": "Upper GI endoscopy with biopsy"},
    {"code": "45378", "description": "Diagnostic colonoscopy"},
    {"code": "45380", "description": "Colonoscopy with biopsy"},
    {"code": "45385", "description": "Colonoscopy with polyp removal by snare"},
    # Vaccines and injections
    {"code": "90471", "description": "Immunization administration, single vaccine"},
    {"code": "90686", "description": "Influenza vaccine, quadrivalent, intramuscular"},
    {"code": "96372", "description": "Therapeutic injection, subcutaneous or intramuscular"},
    # Mental health
    {"code": "90791", "description": "Psychiatric diagnostic evaluation"},
    {"code": "90834", "description": "Psychotherapy, 45 minutes"},
    {"code": "90837", "description": "Psychotherapy, 60 minutes"},
]


def load_cpt_codes() -> list[dict[str, str]]:
    return list(_CPT_DEMO)
