import pandas as pd
import re

# Latest patterns from cli.py
_JOB_FAMILY_PATTERNS = [
    ("Management", r"(?i)manager|director|architect|tech\s*lead|vp\s|head\sof|leiter|führung|\bproducer\b|\bpo\b|product\s*owner|scrum\s*master|koordinator|technical\s*lead"),
    ("Security", r"(?i)secur|cyber|\bsoc\b|\bsiem\b|penetration|sicherheit|firewall|\bgrc\b"),
    ("QA & Testing", r"(?i)\bqa\b|test(?:er|ing|automatisierung)\b|test\s*(?:engineer|auto)|quality\s*(?:assurance|engineer|analyst)|sdet|\bqe\b|qualität"),
    ("DevOps & Cloud", r"(?i)devops|devsecops|site\s*reliab|\bsre\b|cloud|platform|infrastructure|infrastruktur"),
    ("Data & AI", r"(?i)data\s*eng|data\s*scien|machine\s*learn|\bai\b|\bml\b|data\s*analy|business\s*(?:intel|analyst|systems\s*analyst)|daten|\bki\b|künstliche\s*intelligenz"),
    ("Systems & Embedded", r"(?i)system\w*\s*eng|embedded|firmware|mainframe|systemadministrator|systemingenieur|netzwerk|network|it[- ]?support|service\s*eng|helpdesk|\bit\s*techniker|database\s*admin|\bdba\b"),
    ("Frontend & Design", r"(?i)front[\s-]?end|frontend|\bui[\s/]ux\b|ux\s*design|ui\s*design|\bui\s*(?:develop|engineer)|grafik|game\s*design|art\s*lead|web\s*design|product\s*design|software\s*design"),
    ("Sr+ Software Engineer", r"(?i)(?:senior|staff|principal|lead)\s.*(?:software\s*eng|backend\s*eng|full[\s-]?stack\s*eng|\bdeveloper\b|\bengineer\b)"),
    ("Software Developer", r"(?i)software\s*develop|full[\s-]?stack|fullstack|\.net\s*develop|java\s*develop|web\s*develop|back[\s-]?end\s*develop|programmer[\s/]*analyst|application\s*develop|react\s*(?:native\s*)?develop|php\s*develop|python\s*develop|ruby\s*develop|angular\s*develop|wordpress\s*develop|javascript\s*develop|flutter\s*develop|html\s*develop|coldfusion\s*develop|\bweb\s*app\w*\s*(?:develop|program)|(?:it|junior|senior)\s*develop|^develop|entwickler|programmierer|informatik|\bdeveloper\b|\bentwicklung\b|mobile\s*app"),
    ("Software Engineer", r"(?i)software\s*eng|backend\s*eng|full[\s-]?stack\s*eng|integration\s*eng|product\s*eng|java\s*eng|validation\s*eng|softwareentwickler|application\s*eng|requirements\s*eng|analyst|specialist|associate|\bengineer\b|ingenieur"),
]

_SECTOR_TO_NACE = {
    "Information Technology": "J", "Informationstechnologie": "J",
    "Media & Communication": "J", "Telecommunications": "J", "Media": "J", "Medien & Kommunikation": "J", "Media and communication": "J",
    "Manufacturing": "C", "Produktion": "C",
    "Aerospace & Defense": "C", "Aerospace and defence": "C", "Pharmaceutical & Biotechnology": "C", "Luft- & Raumfahrt, Verteidigung": "C", "Pharmaceutical and biotechnology": "C", 
    "Financial Services": "K", "Insurance": "K", "Finanzen": "K", "Finance": "K",
    "Management & Consulting": "M", "Legal": "M", "Management & Beratung": "M", "Management and consulting": "M",
    "Healthcare": "Q", "Gesundheitswesen": "Q",
    "Retail & Wholesale": "G", "Einzel- & Großhandel": "G", "Retail and wholesale": "G",
    "Education": "P", "Bildungwesen": "P",
    "Human Resources & Staffing": "N", "Personalwesen": "N", "Human resources and staffing": "N",
    "Government & Public Administration": "O", "Government and public administration": "O",
    "Transportation & Logistics": "H", "Transport & Logistik": "H", "Transportation and logistics": "H",
    "Construction, Repair & Maintenance Services": "F", "Bauwesen, Reparatur & Instandhaltung": "F", "Construction, repair and maintenance": "F",
    "Real Estate": "L", "Real estate": "L",
    "Hotels & Travel Accommodation": "I", "Restaurants & Food Service": "I", "Hotel and travel accommodation": "I", "Hotels and travel accommodation": "I", "Restaurants and food service": "I",
    "Agriculture": "A",
    "Arts, Entertainment & Recreation": "R", "Arts, entertainment and recreation": "R",
    "Nonprofit & NGO": "S", "Personal Consumer Services": "S", "Non-profit and NGO": "S", "Personal consumer services": "S",
    "Energy, Mining & Utilities": "D", "Energie, Bergbau, Versorgungswirtschaft": "D", "Energy, mining, utilities": "D",
}

def check():
    df = pd.read_csv("data/inputs/in_relevant.csv", sep=";", encoding="utf-8-sig")
    print(f"Total rows: {len(df)}")
    
    # 1. Sector Coverage (Case Insensitive)
    df["nace"] = df["sector"].map(_SECTOR_TO_NACE).fillna("Unknown")
    
    cov = (df["nace"] != "Unknown").mean()
    print(f"Sector Coverage: {cov:.1%}")
    if cov < 0.99:
        print("\nTop Missing Sectors (Still Missing):")
        print(df[df["nace"] == "Unknown"]["sector"].value_counts().head(5))

    # 2. Family Coverage
    def get_family(title):
        if pd.isna(title): return "Other"
        for fam, pat in _JOB_FAMILY_PATTERNS:
            if re.search(pat, str(title), re.IGNORECASE):
                return fam
        return "Other"
    
    df["family"] = df["job_title"].apply(get_family)
    cov_fam = (df["family"] != "Other").mean()
    print(f"\nJob Family Coverage: {cov_fam:.1%}")
    if cov_fam < 0.90:
        print("\nTop Missing Titles:")
        print(df[df["family"] == "Other"]["job_title"].value_counts().head(10))

if __name__ == "__main__":
    check()
