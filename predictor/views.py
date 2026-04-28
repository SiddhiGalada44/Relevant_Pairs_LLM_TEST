import json
import hashlib
import logging
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from google import genai

logger = logging.getLogger(__name__)
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
CACHE = {}

# --- Body region mappings ---
REGION_KEYWORDS = {
    "head": ["BRAIN", "HEAD", "SKULL", "NEURO", "INTRACRANIAL", "CRANIAL", "ORBIT", "SELLA", "IAC", "FACE", "SINUS"],
    "neck": ["NECK", "THYROID", "CAROTID", "SOFT TISSUE NECK"],
    "chest": ["CHEST", "LUNG", "THORAX", "CARDIAC", "HEART", "PULMONARY", "RIBS"],
    "abdomen": ["ABDOMEN", "LIVER", "KIDNEY", "RENAL", "PELVIS", "PANCREAS", "GALLBLADDER", "BOWEL", "COLON", "APPENDIX"],
    "spine": ["SPINE", "CERVICAL", "LUMBAR", "THORACIC", "SACRAL", "VERTEBR"],
    "upper_extremity": ["SHOULDER", "ELBOW", "WRIST", "HAND", "HUMERUS", "FOREARM", "FINGER"],
    "lower_extremity": ["HIP", "KNEE", "ANKLE", "FOOT", "FEMUR", "TIBIA", "TOE"],
    "vascular": ["ANGIO", "AORTA", "VENOUS", "ARTERIAL", "VASCULAR", "MRA", "CTA"],
}


def extract_regions(description):
    desc = description.upper()
    regions = set()
    for region, keywords in REGION_KEYWORDS.items():
        if any(kw in desc for kw in keywords):
            regions.add(region)
    return regions


def heuristic_predict(current_desc, prev_desc):
    """
    Returns:
      "relevant"     - definitely relevant (same anatomy)
      "not_relevant"  - definitely not relevant (no overlap at all)
      "ambiguous"    - can't tell, need LLM
    """
    current_regions = extract_regions(current_desc)
    prev_regions = extract_regions(prev_desc)

    # If we couldn't identify either, it's ambiguous
    if not current_regions or not prev_regions:
        return "ambiguous"

    # Direct overlap = relevant (high confidence)
    if current_regions & prev_regions:
        return "relevant"

    # If regions are completely unrelated AND both are clearly identified,
    # still send to LLM — the report text might reveal a connection
    return "ambiguous"


def make_cache_key(current_study, previous_exam):
    blob = json.dumps({
        "current_desc": current_study.get("study_description", ""),
        "current_date": current_study.get("study_date", ""),
        "prev_desc": previous_exam.get("study_description", ""),
        "prev_date": previous_exam.get("study_date", ""),
        "prev_report": previous_exam.get("report_text", ""),
    }, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def llm_predict_batch(case_id, current_study, exams_to_check):
    """Call LLM only for ambiguous exams, all in one batched call."""
    current_info = (
        f"Study: {current_study['study_description']}\n"
        f"Date: {current_study['study_date']}"
    )

    prior_blocks = []
    for i, (exam, _) in enumerate(exams_to_check):
        report = exam.get("report_text", "N/A")
        if len(report) > 3000:
            report = report[:3000] + "... [truncated]"
        prior_blocks.append(
            f"--- Exam {i+1} ---\n"
            f"Study ID: {exam['study_id']}\n"
            f"Description: {exam.get('study_description', 'N/A')}\n"
            f"Date: {exam.get('study_date', 'N/A')}\n"
            f"Report: {report}"
        )

    prompt = (
        "You are a radiology AI. A radiologist is about to read a current exam. "
        "Decide which previous exams are relevant for comparison.\n\n"
        "RELEVANT means: same/overlapping anatomy, useful for comparison, "
        "or clinically connected findings.\n"
        "NOT RELEVANT means: completely different body part with no clinical link.\n\n"
        "Respond ONLY with a JSON array like:\n"
        '[{"study_id": "123", "relevant": true}, {"study_id": "456", "relevant": false}]\n'
        "No other text.\n\n"
        f"CURRENT EXAM:\n{current_info}\n\n"
        f"PREVIOUS EXAMS:\n\n" + "\n\n".join(prior_blocks)
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        return {str(item["study_id"]): bool(item["relevant"]) for item in json.loads(raw)}

    except Exception as e:
        logger.error(f"LLM failed for case {case_id}: {e}")
        # Default to True on failure
        return {str(exam["study_id"]): True for exam, _ in exams_to_check}


def predict_case(case):
    case_id = case["case_id"]
    current_study = case["current_study"]
    previous_exams = case.get("previous_examinations") or case.get("prior_studies", [])

    if not previous_exams:
        return []

    results = {}
    ambiguous = []  # only these go to the LLM

    for exam in previous_exams:
        sid = str(exam["study_id"])

        # Check cache first
        ckey = make_cache_key(current_study, exam)
        if ckey in CACHE:
            results[sid] = CACHE[ckey]
            continue

        # Run heuristic
        verdict = heuristic_predict(
            current_study["study_description"],
            exam.get("study_description", "")
        )

        if verdict == "relevant":
            results[sid] = True
            CACHE[ckey] = True
        elif verdict == "not_relevant":
            results[sid] = False
            CACHE[ckey] = False
        else:
            ambiguous.append((exam, ckey))

    # Only call LLM for ambiguous cases
    if ambiguous:
        logger.info(f"Case {case_id}: {len(previous_exams)} priors, "
                     f"{len(ambiguous)} ambiguous → sending to LLM")
        llm_results = llm_predict_batch(case_id, current_study, ambiguous)
        for exam, ckey in ambiguous:
            sid = str(exam["study_id"])
            results[sid] = llm_results.get(sid, True)
            CACHE[ckey] = results[sid]
    else:
        logger.info(f"Case {case_id}: {len(previous_exams)} priors, "
                     f"all resolved by heuristic — no LLM call")

    predictions = []
    for exam in previous_exams:
        predictions.append({
            "case_id": str(case_id),
            "study_id": str(exam["study_id"]),
            "predicted_is_relevant": results.get(str(exam["study_id"]), True),
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