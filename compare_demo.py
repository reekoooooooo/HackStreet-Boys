import pandas as pd
import json
import os

CSV_PATH = "gis_records.csv"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

ai_result = {
    "pole_id": "P0004",
    "vegetation": "No",            
    "guy_guard": "Yes",             
    "pole_type": "Steel",            
    "has_conduit_riser": "Yes"       
}

df = pd.read_csv(CSV_PATH)

def get_gis_record(pole_id):
    rec = df[df["pole_id"] == pole_id]
    if rec.empty:
        return None
    return rec.iloc[0].to_dict()

def compare(ai, gis):
    mismatches = []
    gis_veg = gis["expected_vegetation"]
    if str(ai["vegetation"]).strip().lower() != str(gis_veg).strip().lower():
        mismatches.append(f"Vegetation mismatch: GIS={gis_veg} vs AI={ai['vegetation']}")
    gis_gg = gis["expected_guy_guard"]
    if str(ai["guy_guard"]).strip().lower() != str(gis_gg).strip().lower():
        mismatches.append(f"Guy guard mismatch: GIS={gis_gg} vs AI={ai['guy_guard']}")
    gis_type = gis["pole_type"]
    if str(ai["pole_type"]).strip().lower() != str(gis_type).strip().lower():
        mismatches.append(f"Pole type mismatch: GIS={gis_type} vs AI={ai['pole_type']}")
    gis_riser = gis["has_conduit_riser"]
    if str(ai["has_conduit_riser"]).strip().lower() != str(gis_riser).strip().lower():
        mismatches.append(f"Conduit riser mismatch: GIS={gis_riser} vs AI={ai['has_conduit_riser']}")
    return mismatches

def compute_risk(ai, mismatches):
    score = 100
    if str(ai.get("vegetation", "No")).strip().lower() == "yes":
        score -= 15
    if str(ai.get("guy_guard", "Yes")).strip().lower() == "no":
        score -= 35
    score -= 25 * len(mismatches)
    score = max(0, min(100, score))
    if score >= 80:
        status = "OK"
    elif score >= 50:
        status = "WARNING"
    else:
        status = "HIGH RISK"
    return score, status

gis_row = get_gis_record(ai_result["pole_id"])
if gis_row is None:
    print(f"No GIS record found for pole_id {ai_result['pole_id']}")
else:
    mismatches = compare(ai_result, gis_row)
    score, status = compute_risk(ai_result, mismatches)

    print("=== Comparison Result ===")
    print(f"Pole ID: {ai_result['pole_id']}")
    print("GIS record:")
    for k, v in gis_row.items():
        print(f"  {k}: {v}")
    print("AI detected:")
    for k, v in ai_result.items():
        if k != "pole_id":
            print(f"  {k}: {v}")
    print("\nMismatches:")
    if mismatches:
        for m in mismatches:
            print(" -", m)
    else:
        print(" - None")

    print(f"\nRisk score: {score}  Status: {status}")

    report = {
        "pole_id": ai_result["pole_id"],
        "gis": gis_row,
        "ai": ai_result,
        "mismatches": mismatches,
        "risk_score": score,
        "status": status
    }
    out_path = os.path.join(REPORTS_DIR, f"{ai_result['pole_id']}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {out_path}")

