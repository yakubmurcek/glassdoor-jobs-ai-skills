import csv
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# File paths
INPUT_CSV = "data/outputs/us_relevant_30_reprocessed.csv"
OUTPUT_MD = "/Users/yakub/.gemini/antigravity/brain/634894df-6e79-49fe-8c26-a0bdb35021e3/quality_report.md"

def generate_report():
    try:
        with open(INPUT_CSV, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            rows = list(reader)
    except Exception as e:
        logging.error(f"Failed to read input CSV: {e}")
        return

    with open(OUTPUT_MD, 'w', encoding='utf-8') as f:
        f.write(f"# Quality Assurance Report: Skill Extraction\n\n")
        f.write(f"**Input File**: `{INPUT_CSV}`\n")
        f.write(f"**Total Parsed Rows**: {len(rows)}\n\n")
        f.write("> **Note**: This report displays the extracted skills against the job context to verify accuracy.\n\n")

        for i, row in enumerate(rows, 1):
            title = row.get('job_title', 'N/A')
            company = row.get('company', 'N/A')
            desc = row.get('job_desc_text', '')[:300].replace('\n', ' ') + "..."
            
            # AI & Tier
            tier = row.get('desc_tier_llm', 'N/A')
            ai_rationale = row.get('desc_rationale_llm', 'N/A')
            is_real_ai = row.get('is_real_ai', 'N/A')
            
            # Skills
            hard = row.get('hardskills', '')
            soft = row.get('softskills', '')
            
            f.write(f"## {i}. {title} @ {company}\n")
            f.write(f"- **Description Snippet**: {desc}\n")
            f.write(f"- **AI Tier**: `{tier}` (Is Real AI: `{is_real_ai}`)\n")
            f.write(f"- **AI Rationale**: {ai_rationale}\n")
            f.write(f"- **Hard Skills**: {hard}\n")
            f.write(f"- **Soft Skills**: {soft}\n")
            f.write("\n---\n")

    logging.info(f"Report generated at {OUTPUT_MD}")

if __name__ == "__main__":
    generate_report()
