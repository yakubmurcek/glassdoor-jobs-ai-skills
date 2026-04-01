import pandas as pd
import re

def verify_india_data():
    df = pd.read_csv('data/inputs/in_relevant.csv', sep=None, engine='python')
    print(f"Loaded {len(df)} rows.")

    # Check Region
    if 'state' in df.columns:
        print("\n--- Top States in IN Dataset --=")
        print(df['state'].value_counts().head(10))

    # Check Job Family Coverage
    _JOB_FAMILY_PATTERNS = [
        ("Management", r"(?i)manager|director|architect|tech\s*lead|vp\s|head\sof|leiter|führung|\bproducer\b|\bpo\b|product\s*owner|scrum\s*master|koordinator"),
        ("Security", r"(?i)secur|cyber|\bsoc\b|\bsiem\b|penetration|sicherheit|firewall"),
        ("QA & Testing", r"(?i)\bqa\b|test(?:er|ing|automatisierung)\b|test\s*(?:engineer|auto)|quality\s*(?:assurance|engineer|analyst)|sdet|\bqe\b|qualität"),
        ("DevOps & Cloud", r"(?i)devops|devsecops|site\s*reliab|\bsre\b|cloud|platform|infrastructure|infrastruktur"),
        ("Data & AI", r"(?i)data\s*eng|data\s*scien|machine\s*learn|\bai\b|\bml\b|data\s*analy|business\s*(?:intel|analyst|systems\s*analyst)|daten|\bki\b|künstliche\s*intelligenz"),
        ("Systems & Embedded", r"(?i)system\w*\s*eng|embedded|firmware|mainframe|systemadministrator|systemingenieur|netzwerk|network|it[- ]?support|service\s*eng|helpdesk|\bit\s*techniker"),
        ("Frontend & Design", r"(?i)front[\s-]?end|frontend|\bui[\s/]ux\b|ux\s*design|ui\s*design|\bui\s*(?:develop|engineer)|grafik|game\s*design|art\s*lead"),
        ("Sr+ Software Engineer", r"(?i)(?:senior|staff|principal|lead)\s.*(?:software\s*eng|backend\s*eng|full[\s-]?stack\s*eng|\bdeveloper\b|\bengineer\b)"),
        ("Software Developer", r"(?i)software\s*develop|full[\s-]?stack|fullstack|\.net\s*develop|java\s*develop|web\s*develop|back[\s-]?end\s*develop|programmer[\s/]*analyst|application\s*develop|react\s*(?:native\s*)?develop|php\s*develop|python\s*develop|ruby\s*develop|angular\s*develop|wordpress\s*develop|javascript\s*develop|flutter\s*develop|html\s*develop|coldfusion\s*develop|\bweb\s*app\w*\s*(?:develop|program)|(?:it|junior|senior)\s*develop|^develop|entwickler|programmierer|informatik|\bdeveloper\b|\bentwicklung\b"),
        ("Software Engineer", r"(?i)software\s*eng|backend\s*eng|full[\s-]?stack\s*eng|integration\s*eng|product\s*eng|java\s*eng|validation\s*eng|softwareentwickler|application\s*eng|requirements\s*eng|\bengineer\b|ingenieur"),
    ]

    df['job_family'] = 'Other'
    for family_name, pattern in _JOB_FAMILY_PATTERNS:
        mask = df['job_title'].str.contains(pattern, na=False, regex=True) & (df['job_family'] == 'Other')
        df.loc[mask, 'job_family'] = family_name

    coverage = (df['job_family'] != 'Other').mean()
    print(f"\n--- Job Family Coverage: {coverage:.2%} ---")
    print(df['job_family'].value_counts())

    print("\n--- Top Unmatched Job Titles ('Other') ---")
    print(df[df['job_family'] == 'Other']['job_title'].value_counts().head(20))

    # Check Sectors
    _SECTOR_TO_NACE = {
        "Information Technology": "J", "Informationstechnologie": "J",
        "Media & Communication": "J", "Telecommunications": "J", "Media": "J", "Medien & Kommunikation": "J",
        "Manufacturing": "C", "Produktion": "C",
        "Aerospace & Defense": "C", "Pharmaceutical & Biotechnology": "C", "Luft- & Raumfahrt, Verteidigung": "C",
        "Financial Services": "K", "Insurance": "K", "Finanzen": "K",
        "Management & Consulting": "M", "Legal": "M", "Management & Beratung": "M",
        "Healthcare": "Q", "Gesundheitswesen": "Q",
        "Retail & Wholesale": "G", "Einzel- & Großhandel": "G",
        "Education": "P", "Bildungwesen": "P",
        "Human Resources & Staffing": "N", "Personalwesen": "N",
        "Government & Public Administration": "O",
        "Transportation & Logistics": "H", "Transport & Logistik": "H",
        "Construction, Repair & Maintenance Services": "F", "Bauwesen, Reparatur & Instandhaltung": "F",
        "Real Estate": "L",
        "Hotels & Travel Accommodation": "I", "Restaurants & Food Service": "I",
        "Agriculture": "A",
        "Arts, Entertainment & Recreation": "R",
        "Nonprofit & NGO": "S", "Personal Consumer Services": "S",
        "Energy, Mining & Utilities": "D", "Energie, Bergbau, Versorgungswirtschaft": "D",
    }
    
    df['sector_nace'] = df['sector'].map(_SECTOR_TO_NACE).fillna('Unknown')
    sector_coverage = (df['sector_nace'] != 'Unknown').mean()
    print(f"\n--- Sector Coverage: {sector_coverage:.2%} ---")
    print("\n--- Top Unmatched Sectors ('Unknown') ---")
    print(df[df['sector_nace'] == 'Unknown']['sector'].value_counts().head(10))


if __name__ == '__main__':
    verify_india_data()
