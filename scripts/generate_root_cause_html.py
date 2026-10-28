import json
from pathlib import Path

def generate_html():
    json_path = Path("outputs/detection/repair/root_cause_report.json")
    if not json_path.exists():
        print(f"Missing {json_path}")
        return
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NutriVision AI — Detection Root Cause Audit Report</title>
    <style>
        :root {{
            --bg: #0f172a;
            --surface: #1e293b;
            --border: #334155;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --primary: #38bdf8;
            --danger: #f87171;
            --warning: #fbbf24;
            --success: #4ade80;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
            line-height: 1.6;
            margin: 0;
            padding: 40px 20px;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
        }}
        .header {{
            border-bottom: 1px solid #334155;
            padding-bottom: 24px;
            margin-bottom: 32px;
        }}
        h1 {{
            font-size: 28px;
            color: #38bdf8;
            margin: 0 0 8px 0;
        }}
        .meta {{
            color: #94a3b8;
            font-size: 14px;
        }}
        .summary-card {{
            background-color: #1e293b;
            border-left: 4px solid #f87171;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 32px;
        }}
        .finding-card {{
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 16px;
        }}
        .finding-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 12px;
        }}
        .badge {{
            font-size: 12px;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 20px;
            text-transform: uppercase;
        }}
        .badge-critical {{ background: rgba(248, 113, 113, 0.2); color: #f87171; border: 1px solid #f87171; }}
        .badge-high {{ background: rgba(251, 191, 36, 0.2); color: #fbbf24; border: 1px solid #fbbf24; }}
        .badge-medium {{ background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid #38bdf8; }}
        .badge-clean {{ background: rgba(74, 222, 128, 0.2); color: #4ade80; border: 1px solid #4ade80; }}
        .evidence {{
            background: #090d16;
            padding: 12px;
            border-radius: 6px;
            font-family: monospace;
            font-size: 13px;
            margin: 10px 0;
            border: 1px solid #1e293b;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>NutriVision AI — Detection Root Cause Audit Report</h1>
            <div class="meta">Phase: {data['audit_metadata']['phase']} | Generated: {data['audit_metadata']['generated_at']} | Status: {data['audit_metadata']['status']}</div>
        </div>

        <div class="summary-card">
            <h3 style="margin-top:0; color:#f87171;">Executive Diagnostic Finding</h3>
            <p>{data['executive_summary']}</p>
        </div>

        <h2>Ranked Diagnostic Findings</h2>
"""
    for f in data['ranked_findings']:
        sev_class = 'badge-medium'
        if 'CRITICAL' in f['severity']:
            sev_class = 'badge-critical'
        elif 'HIGH' in f['severity']:
            sev_class = 'badge-high'
        elif 'CLEAN' in f['severity']:
            sev_class = 'badge-clean'

        html += f"""
        <div class="finding-card">
            <div class="finding-header">
                <h3 style="margin:0;">#{f['rank']}: {f['category']}</h3>
                <span class="badge {sev_class}">{f['severity']}</span>
            </div>
            <div class="evidence"><strong>Evidence:</strong> {f['evidence']}</div>
            <div><strong>Impact:</strong> {f['impact']}</div>
        </div>
        """

    html += """
    </div>
</body>
</html>
"""
    out_p = Path("outputs/detection/repair/root_cause_report.html")
    out_p.write_text(html, encoding="utf-8")
    print(f"Wrote {out_p}")

if __name__ == "__main__":
    generate_html()
