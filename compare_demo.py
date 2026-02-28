# compare_demo.py
import pandas as pd
import json
import os
import difflib
import logging
from typing import Dict, List, Tuple, Any

# Setup
CSV_PATH = "gis_records.csv"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Configuration: weights & normalization dictionaries
CONFIG = {
    "weights": {
        "vegetation_present": 15,   # penalty when vegetation present
        "guy_guard_missing": 35,    # penalty when missing
        "per_mismatch": 25          # penalty per mismatch
    },
    "pole_type_aliases": {
        "wood": ["wood", "wooden"],
        "steel": ["steel", "stl", "metal"],
        "composite": ["composite", "fiberglass", "fibreglass"]
    },
    "boolean_true_values": {"yes", "y", "true", "1", "t"},
    "boolean_false_values": {"no", "n", "false", "0", "f"}
}

# Helper functions
def normalize_bool(val: Any) -> str:
    if pd.isna(val):
        return "unknown"
    s = str(val).strip().lower()
    if s in CONFIG["boolean_true_values"]:
        return "yes"
    if s in CONFIG["boolean_false_values"]:
        return "no"
    return "unknown"

def normalize_str(val: Any) -> str:
    if pd.isna(val):
        return ""
    return str(val).strip()

def normalize_pole_type(val: Any) -> Tuple[str, float]:
    """
    Return (canonical_type, confidence_score)
    Uses exact alias mapping, otherwise fuzzy match across known aliases.
    """
    s = normalize_str(val).lower()
    if not s:
        return "", 0.0
    # direct alias match
    for canon, aliases in CONFIG["pole_type_aliases"].items():
        if s in aliases:
            return canon, 1.0
    # fuzzy match against canonical keys and aliases
    candidates = []
    for canon, aliases in CONFIG["pole_type_aliases"].items():
        for a in aliases:
            candidates.append((a, canon))
    choices = [c[0] for c in candidates]
    match = difflib.get_close_matches(s, choices, n=1, cutoff=0.6)
    if match:
        # find the canonical for that alias
        for a, canon in candidates:
            if a == match[0]:
                # confidence scaled by similarity ratio
                ratio = difflib.SequenceMatcher(None, s, a).ratio()
                return canon, float(ratio)
    # fallback: return original lowercased with low confidence
    return s, 0.2

def load_gis(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str)  # read as strings to avoid dtype surprises
    expected_cols = {"pole_id", "expected_vegetation", "expected_guy_guard", "pole_type", "has_conduit_riser"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    df = df.fillna("")  # convert NaNs to empty string for normalization
    return df

# Comparison returns structured mismatches
def compare(ai: Dict[str, Any], gis: Dict[str, Any]) -> List[Dict[str, Any]]:
    mismatches = []

    # vegetation
    gis_veg = normalize_bool(gis.get("expected_vegetation", ""))
    ai_veg = normalize_bool(ai.get("vegetation", ""))
    if gis_veg != "unknown" and ai_veg != "unknown" and gis_veg != ai_veg:
        mismatches.append({
            "field": "vegetation",
            "gis": gis.get("expected_vegetation"),
            "ai": ai.get("vegetation"),
            "gis_normalized": gis_veg,
            "ai_normalized": ai_veg,
            "severity": "medium"
        })

    # guy guard
    gis_gg = normalize_bool(gis.get("expected_guy_guard", ""))
    ai_gg = normalize_bool(ai.get("guy_guard", ""))
    if gis_gg != "unknown" and ai_gg != "unknown" and gis_gg != ai_gg:
        mismatches.append({
            "field": "guy_guard",
            "gis": gis.get("expected_guy_guard"),
            "ai": ai.get("guy_guard"),
            "gis_normalized": gis_gg,
            "ai_normalized": ai_gg,
            "severity": "high"
        })

    # pole type with fuzzy matching
    gis_type_canon, gis_conf = normalize_pole_type(gis.get("pole_type", ""))
    ai_type_canon, ai_conf = normalize_pole_type(ai.get("pole_type", ""))
    if gis_type_canon and ai_type_canon and gis_type_canon != ai_type_canon:
        mismatches.append({
            "field": "pole_type",
            "gis": gis.get("pole_type"),
            "ai": ai.get("pole_type"),
            "gis_canonical": gis_type_canon,
            "ai_canonical": ai_type_canon,
            "gis_confidence": gis_conf,
            "ai_confidence": ai_conf,
            "severity": "medium"
        })

    # conduit riser
    gis_riser = normalize_bool(gis.get("has_conduit_riser", ""))
    ai_riser = normalize_bool(ai.get("has_conduit_riser", ""))
    if gis_riser != "unknown" and ai_riser != "unknown" and gis_riser != ai_riser:
        mismatches.append({
            "field": "has_conduit_riser",
            "gis": gis.get("has_conduit_riser"),
            "ai": ai.get("has_conduit_riser"),
            "gis_normalized": gis_riser,
            "ai_normalized": ai_riser,
            "severity": "low"
        })

    return mismatches

# Risk computation uses weights and allows adjusting by confidences
def compute_risk(ai: Dict[str, Any], mismatches: List[Dict[str, Any]]) -> Tuple[int, str, Dict[str, str]]:
    """
    Returns (score:int, status:str, assessment:dict)
    """
    score = 100

    # Penalties based on detected conditions (use normalized helpers)
    if normalize_bool(ai.get("vegetation", "No")) == "yes":
        score -= CONFIG["weights"]["vegetation_present"]
    if normalize_bool(ai.get("guy_guard", "Yes")) == "no":
        score -= CONFIG["weights"]["guy_guard_missing"]

    # Penalty per mismatch; allow severity scaling
    for m in mismatches:
        severity = m.get("severity", "medium")
        if severity == "high":
            penalty = CONFIG["weights"]["per_mismatch"] * 1.5
        elif severity == "low":
            penalty = CONFIG["weights"]["per_mismatch"] * 0.5
        else:
            penalty = CONFIG["weights"]["per_mismatch"]
        score -= penalty

    # clamp and convert
    score = max(0, min(100, int(round(score))))

    # status classification (simple backward-compatible label)
    if score >= 80:
        status = "OK"
    elif score >= 50:
        status = "WARNING"
    else:
        status = "HIGH RISK"

    # richer assessment
    assessment = generate_assessment(score, mismatches)

    return score, status, assessment

def generate_assessment(score, mismatches):
    if score >= 90 and not mismatches:
        return {
            "risk_level": "SAFE",
            "data_status": "VERIFIED",
            "action": "No review required",
            "confidence": "High"
        }
    elif score >= 70:
        return {
            "risk_level": "CAUTION",
            "data_status": "Minor discrepancies",
            "action": "Community validation suggested",
            "confidence": "Medium"
        }
    else:
        return {
            "risk_level": "HIGH RISK",
            "data_status": "Significant discrepancy",
            "action": "Field inspection recommended",
            "confidence": "Low"
        }

# Utility to fetch a single GIS record as dict
def get_gis_record(df: pd.DataFrame, pole_id: str) -> Dict[str, Any]:
    rec = df[df["pole_id"] == str(pole_id)]
    if rec.empty:
        return None
    return rec.iloc[0].to_dict()

# Main flow (single AI record example)
if __name__ == "__main__":
    # Example AI result - replace with real AI/OCR outputs (including confidences if available)
    ai_result = {
        "pole_id": "P0010",
        "vegetation": "No",
        "guy_guard": "Yes",
        "pole_type": "Steel",
        "has_conduit_riser": "Yes"
    }

    df = load_gis(CSV_PATH)
    gis_row = get_gis_record(df, ai_result["pole_id"])
    if gis_row is None:
        logging.error(f"No GIS record found for pole_id {ai_result['pole_id']}")
    else:
        mismatches = compare(ai_result, gis_row)
        score, status, assessment = compute_risk(ai_result, mismatches)

        # Print nice summary
        print("=== Comparison Result ===")
        print(f"Pole ID: {ai_result['pole_id']}")
        print("GIS record:")
        for k, v in gis_row.items():
            print(f"  {k}: {v}")
        print("\nAI detected:")
        for k, v in ai_result.items():
            if k != "pole_id":
                print(f"  {k}: {v}")

        print("\nMismatches (structured):")
        if mismatches:
            for m in mismatches:
                print(" -", json.dumps(m, indent=2))
        else:
            print(" - None")

        # Improved human-friendly assessment output
        print(f"\nRisk score: {score}  Status: {status}")
        print("\nSystem Assessment:")
        print(f"  Risk level: {assessment['risk_level']}")
        print(f"  Data status: {assessment['data_status']}")
        print(f"  Next action: {assessment['action']}")
        print(f"  Confidence: {assessment.get('confidence','Unknown')}")

        report = {
            "pole_id": ai_result["pole_id"],
            "gis": gis_row,
            "ai": ai_result,
            "mismatches": mismatches,
            "risk_score": score,
            "status": status,
            "assessment": assessment
        }
        out_path = os.path.join(REPORTS_DIR, f"{ai_result['pole_id']}.json")
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {out_path}")