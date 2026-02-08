"""
LUMINARK OVERWATCH - Partner Bundle Export
===========================================
Exports comprehensive system analysis and validation reports as a ZIP bundle
containing CSV, PDF, JSON, and README files.

Bundle Contents:
- README.txt: Bundle overview and interpretation guide
- systems_report.csv: All monitored systems with SAP diagnostics
- systems_report.json: Full JSON export of system data
- validation_report.csv: Response validation results
- alerts.csv: All alerts and interventions
- overwatch_summary.pdf: Visual summary report (HTML for now)
"""

import io
import json
import csv
import zipfile
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import asdict


def generate_readme(overview: Dict[str, Any], timestamp: str) -> str:
    """Generate README.txt content for the bundle"""
    return f"""LUMINARK OVERWATCH - Partner Bundle
=====================================
Generated: {timestamp}
Version: 1.1.0

CONTENTS
--------
1. systems_report.csv - All monitored systems with SAP stage diagnostics
2. systems_report.json - Full JSON export with complete metrics
3. alerts.csv - Alert history with severity and context
4. validation_summary.json - Response validation statistics
5. overwatch_summary.html - Visual summary report

SYSTEM STATUS OVERVIEW
----------------------
Total Systems: {overview.get('total_systems', 0)}
Online: {overview.get('status_counts', {}).get('online', 0)}
Degraded: {overview.get('status_counts', {}).get('degraded', 0)}
Critical: {overview.get('status_counts', {}).get('critical', 0)}

ACTIVE ALERTS
-------------
Emergency: {overview.get('alert_counts', {}).get('emergency', 0)}
Critical: {overview.get('alert_counts', {}).get('critical', 0)}
Warning: {overview.get('alert_counts', {}).get('warning', 0)}

SAP FRAMEWORK GUIDE
-------------------
Stage 0 (Plenara): Void/Crisis - Both unstable
Stage 1 (Spark): Emergence - First impulse
Stage 2 (Polarity): Duality - Tension building
Stage 3 (Motion): Execution - Active work
Stage 4 (Foundation): Structure - Stability
Stage 5 (Threshold): CRITICAL DECISION POINT - Bifurcation risk
Stage 6 (Integration): Harmony - Merging dualities
Stage 7 (Illusion): Testing - Reality checks
Stage 8 (Rigidity): TRAP RISK - Crystallization
Stage 9 (Renewal): Transcendence - Aligned

INVERSION PRINCIPLE
-------------------
Even stages (2,4,6,8): Physically stable, Consciously unstable
Odd stages (1,3,5,7,9): Physically unstable, Consciously stable
High inversion = misalignment requiring intervention

CONTACT
-------
LUMINARK Creative Studio
Meridian Axiom Alignment Technologies (Ma'at)
https://luminark.io
"""


def systems_to_csv(systems: List[Dict[str, Any]]) -> str:
    """Convert systems list to CSV string"""
    if not systems:
        return "id,name,status,sap_stage,inversion_level,last_update\n"

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "name", "status", "sap_stage", "inversion_level", "last_update"])

    for sys in systems:
        writer.writerow([
            sys.get("id", ""),
            sys.get("name", ""),
            sys.get("status", ""),
            sys.get("sap_stage", 0),
            sys.get("inversion_level", 0),
            sys.get("last_update", "")
        ])

    return output.getvalue()


def alerts_to_csv(alerts: List[Dict[str, Any]]) -> str:
    """Convert alerts list to CSV string"""
    if not alerts:
        return "system_id,level,title,message,sap_context,timestamp,acknowledged\n"

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["system_id", "level", "title", "message", "sap_context", "timestamp", "acknowledged"])

    for alert in alerts:
        writer.writerow([
            alert.get("system_id", ""),
            alert.get("level", ""),
            alert.get("title", ""),
            alert.get("message", ""),
            alert.get("sap_context", ""),
            alert.get("timestamp", ""),
            alert.get("acknowledged", False)
        ])

    return output.getvalue()


def generate_html_summary(overview: Dict[str, Any], systems: List[Dict[str, Any]], timestamp: str) -> str:
    """Generate HTML summary report"""
    systems_rows = ""
    for sys in systems:
        status_class = sys.get("status", "unknown")
        systems_rows += f"""
        <tr class="{status_class}">
            <td>{sys.get('name', 'Unknown')}</td>
            <td><span class="badge {status_class}">{sys.get('status', 'unknown').upper()}</span></td>
            <td>{sys.get('sap_stage', 0)}</td>
            <td>{sys.get('inversion_level', 0)}/10</td>
        </tr>
        """

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>LUMINARK OVERWATCH Summary</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0a;
            color: #ffffff;
            padding: 32px;
            line-height: 1.5;
        }}
        .header {{
            border-bottom: 2px solid #00ff41;
            padding-bottom: 20px;
            margin-bottom: 32px;
        }}
        h1 {{
            color: #00ff41;
            font-size: 2rem;
            margin-bottom: 8px;
        }}
        .subtitle {{ color: #888; font-size: 0.9rem; }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 32px;
        }}
        .stat-card {{
            background: #111;
            border: 1px solid #333;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2.5rem;
            font-weight: 700;
            color: #00ff41;
        }}
        .stat-label {{ color: #888; font-size: 0.85rem; margin-top: 4px; }}
        .stat-card.critical .stat-value {{ color: #ff3366; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: #111;
            border-radius: 12px;
            overflow: hidden;
        }}
        th, td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid #333;
        }}
        th {{ background: #1a1a1a; color: #888; font-size: 0.8rem; text-transform: uppercase; }}
        .badge {{
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .badge.online {{ background: rgba(0,255,65,0.2); color: #00ff41; }}
        .badge.degraded {{ background: rgba(255,170,0,0.2); color: #ffaa00; }}
        .badge.critical {{ background: rgba(255,51,102,0.2); color: #ff3366; }}
        .footer {{
            margin-top: 32px;
            padding-top: 16px;
            border-top: 1px solid #333;
            color: #666;
            font-size: 0.85rem;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>LUMINARK OVERWATCH</h1>
        <div class="subtitle">AI Regulatory System - Partner Bundle Report</div>
        <div class="subtitle">Generated: {timestamp}</div>
    </div>

    <div class="stats">
        <div class="stat-card">
            <div class="stat-value">{overview.get('total_systems', 0)}</div>
            <div class="stat-label">Total Systems</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{overview.get('status_counts', {}).get('online', 0)}</div>
            <div class="stat-label">Online</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{overview.get('status_counts', {}).get('degraded', 0)}</div>
            <div class="stat-label">Degraded</div>
        </div>
        <div class="stat-card critical">
            <div class="stat-value">{overview.get('status_counts', {}).get('critical', 0)}</div>
            <div class="stat-label">Critical</div>
        </div>
    </div>

    <h2 style="color: #00ff41; margin-bottom: 16px;">Monitored Systems</h2>
    <table>
        <thead>
            <tr>
                <th>System</th>
                <th>Status</th>
                <th>SAP Stage</th>
                <th>Inversion</th>
            </tr>
        </thead>
        <tbody>
            {systems_rows or '<tr><td colspan="4" style="text-align:center;color:#666;">No systems registered</td></tr>'}
        </tbody>
    </table>

    <div class="footer">
        LUMINARK Overwatch &middot; Meridian Axiom Alignment Technologies (Ma'at)
    </div>
</body>
</html>"""


def create_partner_bundle(
    overview: Dict[str, Any],
    systems: List[Dict[str, Any]],
    alerts: List[Dict[str, Any]],
    validation_stats: Optional[Dict[str, Any]] = None
) -> bytes:
    """
    Create a ZIP bundle containing all Overwatch reports.

    Returns:
        bytes: ZIP file content
    """
    timestamp = datetime.now().isoformat()
    timestamp_short = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create in-memory ZIP
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # README
        readme = generate_readme(overview, timestamp)
        zf.writestr("README.txt", readme)

        # Systems CSV
        systems_csv = systems_to_csv(systems)
        zf.writestr("systems_report.csv", systems_csv)

        # Systems JSON (full data)
        systems_json = json.dumps({
            "timestamp": timestamp,
            "overview": overview,
            "systems": systems
        }, indent=2)
        zf.writestr("systems_report.json", systems_json)

        # Alerts CSV
        alerts_csv = alerts_to_csv(alerts)
        zf.writestr("alerts.csv", alerts_csv)

        # Validation stats (if provided)
        if validation_stats:
            validation_json = json.dumps(validation_stats, indent=2)
            zf.writestr("validation_summary.json", validation_json)

        # HTML Summary
        html_summary = generate_html_summary(overview, systems, timestamp)
        zf.writestr("overwatch_summary.html", html_summary)

    return zip_buffer.getvalue()


def save_partner_bundle(
    filepath: str,
    overview: Dict[str, Any],
    systems: List[Dict[str, Any]],
    alerts: List[Dict[str, Any]],
    validation_stats: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save partner bundle to a file.

    Args:
        filepath: Output file path (will add .zip if needed)
        overview: Overwatch overview data
        systems: List of system data
        alerts: List of alert data
        validation_stats: Optional validation statistics

    Returns:
        str: Path to saved file
    """
    if not filepath.endswith(".zip"):
        filepath += ".zip"

    bundle = create_partner_bundle(overview, systems, alerts, validation_stats)

    with open(filepath, "wb") as f:
        f.write(bundle)

    return filepath


# ============================================================
# Demo / Test
# ============================================================

if __name__ == "__main__":
    # Demo data
    overview = {
        "timestamp": datetime.now().isoformat(),
        "total_systems": 3,
        "status_counts": {"online": 1, "degraded": 1, "critical": 1},
        "alert_counts": {"emergency": 0, "critical": 1, "warning": 2}
    }

    systems = [
        {"id": "api-gateway", "name": "API Gateway", "status": "online", "sap_stage": 6.2, "inversion_level": 2, "last_update": "2026-02-08T22:00:00"},
        {"id": "ml-inference", "name": "ML Inference", "status": "degraded", "sap_stage": 4.3, "inversion_level": 7, "last_update": "2026-02-08T22:00:00"},
        {"id": "data-pipeline", "name": "Data Pipeline", "status": "critical", "sap_stage": 0.8, "inversion_level": 0, "last_update": "2026-02-08T22:00:00"},
    ]

    alerts = [
        {"system_id": "data-pipeline", "level": "critical", "title": "Void State", "message": "System at Stage 0", "sap_context": "SAP 0.8", "timestamp": "2026-02-08T22:00:00", "acknowledged": False},
        {"system_id": "ml-inference", "level": "warning", "title": "High Inversion", "message": "Inversion level 7/10", "sap_context": "SAP 4.3", "timestamp": "2026-02-08T21:55:00", "acknowledged": False},
    ]

    # Create bundle
    output_path = save_partner_bundle(
        "/tmp/luminark_overwatch_bundle",
        overview,
        systems,
        alerts,
        {"total_validations": 150, "pass_rate": 0.87}
    )

    print(f"Partner bundle saved to: {output_path}")
    print("\nBundle contents:")
    with zipfile.ZipFile(output_path, "r") as zf:
        for name in zf.namelist():
            info = zf.getinfo(name)
            print(f"  {name}: {info.file_size} bytes")
