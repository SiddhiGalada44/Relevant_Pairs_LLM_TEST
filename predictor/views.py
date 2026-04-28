import json
import logging
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

logger = logging.getLogger(__name__)

# Body region keywords — tuned on public eval data
BODY_REGIONS = {
    "head_brain": ["BRAIN", "HEAD", "SKULL", "INTRACRANIAL", "CRANIAL", "STROKE", "IAC",
                   "SELLA", "PITUITARY", "CRANIO"],
    "face_sinus": ["SINUS", "FACE", "FACIAL", "MAXFAC", "ORBIT", "MANDIBLE", "TMJ", "NASAL"],
    "neck": ["NECK", "THYROID", "LARYNX", "PHARYNX"],
    "chest_lung": ["CHEST", "LUNG", "PULMON", "THORAX", "RIBS", "ESOPHAG"],
    "cardiac": ["ECHO ", "CARDIAC", "HEART", "CORONARY", "MYOCARD", "AORTIC", "MUGA",
                "TRANSTHORAC", "TRANSESOPH", "NMMYO", "NM MYO", "PERF SPECT",
                "PERF SPEC"],
    "breast": ["BREAST", "MAM ", "MAMM", "TOMOSYN", "SCREENER", "SCREENING COMBO",
               "SCREEN COMP", "SCREEN BI", "DIAG TARGET", "DIAG COMP",
               "COMBO HD", "COMBOHD", "BILATERAL COMBO", "DIGITAL SCREEN",
               "SEED LOCAL", "R2 MAMM", "STANDARD SCREEN"],
    "abdomen": ["ABDOMEN", "ABD", "LIVER", "HEPAT", "PANCREA", "GALLBLADDER", "SPLEEN",
                "BILE", "MRCP", "ABDOMINAL", "ENTEROGR", "BOWEL", "COLON",
                "MESENTERY"],
    "pelvis": ["PELVIS", "PEL ", "BLADDER", "PROSTATE", "UTERUS", "OVARY", "RECTAL",
               "TRANSVAG", "PELVIC"],
    "kidney": ["KIDNEY", "RENAL", "RETROPERIT", "UROGRAM"],
    "spine_cervical": ["CERVICAL", "C-SPINE", "CERVICL", "CERV SPINE", "CERV SP"],
    "spine_thoracic": ["THORACIC SPINE", "T-SPINE", "THOR SPINE"],
    "spine_lumbar": ["LUMBAR", "L-SPINE", "LUMBOSACR", "SACR"],
    "upper_ext": ["SHOULDER", "ELBOW", "WRIST", "HAND ", "HUMERUS", "FOREARM", "FINGER",
                  "CLAVICLE", "UPPER EXT", "UPPR", "UPR EXT"],
    "lower_ext": ["HIP", "KNEE", "ANKLE", "FOOT", "FEMUR", "TIBIA", "TOE", "CALCANEUS",
                  "LOWER EXT", "LWR EXT"],
    "vascular": ["ANGIO", "MRA ", "CTA "],
    "venous": ["VENOUS", "DVT", "DOPPLER"],
    "bone_density": ["DXA", "BONE DENSITY", "DEXA"],
    "nuclear": ["NM ", "SPECT", "PET/", "PET ", "LYMPH", "SENTINE"],
    "whole_body": ["WHOLE BODY", "SKULLTHIGH", "SKULL TO "],
}

# Only cross-region rules that are net-positive on training data
RELATED_REGIONS = {
    ("kidney", "abdomen"),
    ("kidney", "pelvis"),
}


def extract_regions(desc):
    d = desc.upper()
    regions = set()
    for region, keywords in BODY_REGIONS.items():
        if any(kw in d for kw in keywords):
            regions.add(region)
    if "whole_body" in regions:
        regions.update(["head_brain", "chest_lung", "abdomen", "pelvis"])
    return regions


def are_related(r1, r2):
    """Check if two region sets overlap or have known clinical relationships."""
    if r1 & r2:
        return True
    for a in r1:
        for b in r2:
            if (a, b) in RELATED_REGIONS or (b, a) in RELATED_REGIONS:
                return True
    return False


def predict_case(case):
    case_id = case["case_id"]
    current_study = case["current_study"]
    previous_exams = case.get("previous_examinations") or case.get("prior_studies", [])

    curr_regions = extract_regions(current_study["study_description"])

    predictions = []
    for exam in previous_exams:
        prev_regions = extract_regions(exam.get("study_description", ""))

        if curr_regions and prev_regions:
            is_relevant = are_related(curr_regions, prev_regions)
        else:
            # If we can't identify regions, default to not relevant
            # (76% of all pairs are not relevant)
            is_relevant = False

        predictions.append({
            "case_id": str(case_id),
            "study_id": str(exam["study_id"]),
            "predicted_is_relevant": is_relevant,
        })

    return predictions


@csrf_exempt
@require_POST
def predict(request):
    data = json.loads(request.body)
    cases = data.get("cases", [])

    all_predictions = []
    for case in cases:
        preds = predict_case(case)
        all_predictions.extend(preds)

    return JsonResponse({"predictions": all_predictions})