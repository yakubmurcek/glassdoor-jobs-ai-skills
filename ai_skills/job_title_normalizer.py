# -*- coding: utf-8 -*-
"""Job Title Normalizer - Sjednocení názvů pozic do homogenních skupin.

Tento modul mapuje různorodé názvy pozic na standardizované kategorie
pro statistickou analýzu a agregaci dat.
"""

import re
from typing import Dict, Optional

# =============================================================================
# DEFINICE HOMOGENNÍCH SKUPIN POZIC
# =============================================================================

# Priority patterns - jsou vyhodnoceny v pořadí (první match vyhrává)
# Formát: (priority, regex_pattern, normalized_name)
JOB_TITLE_PATTERNS = [
    # === MANAGEMENT & LEADERSHIP ===
    (10, r'\b(cto|chief\s*technology\s*officer)\b', 'CTO'),
    (10, r'\b(cio|chief\s*information\s*officer)\b', 'CIO'),
    (10, r'\b(ciso|chief\s*information\s*security\s*officer)\b', 'CISO'),
    (10, r'\bdirector\b.*\b(software|engineering|development|it)\b', 'Director of Engineering'),
    (10, r'\bvp\b.*\b(engineering|technology)\b', 'VP of Engineering'),
    (10, r'\bhead\s*of\b.*\b(engineering|development)\b', 'Head of Engineering'),
    (10, r'\bengineering\s*manager\b|\bmanager.*software\s*engineering\b', 'Engineering Manager'),
    (10, r'\bsoftware\s*development\s*manager\b', 'Software Development Manager'),
    (10, r'\btechnical\s*lead\b|\btech\s*lead\b', 'Tech Lead'),
    (10, r'\bteam\s*lead\b.*\b(software|developer|engineer)\b', 'Tech Lead'),
    (10, r'\bfounding\s*engineer\b', 'Founding Engineer'),
    
    # === AI / ML / DATA SCIENCE ===
    (20, r'\b(ai|artificial\s*intelligence)\s*(software\s*)?(engineer|developer)\b', 'AI/ML Engineer'),
    (20, r'\bml\s*engineer\b|\bmachine\s*learning\s*engineer\b', 'AI/ML Engineer'),
    (20, r'\bmlops\s*engineer\b', 'MLOps Engineer'),
    (20, r'\bdata\s*scientist\b', 'Data Scientist'),
    (20, r'\bresearch\s*(scientist|engineer)\b.*\b(ml|ai|machine\s*learning)\b', 'AI/ML Research Scientist'),
    (20, r'\b(nlp|natural\s*language)\s*engineer\b', 'NLP Engineer'),
    (20, r'\bcomputer\s*vision\s*engineer\b', 'Computer Vision Engineer'),
    (20, r'\bai\s*architect\b', 'AI Architect'),
    (20, r'\bai\s*automation\s*engineer\b', 'AI/ML Engineer'),
    
    # === DATA ENGINEERING & ANALYTICS ===
    (30, r'\bdata\s*engineer\b', 'Data Engineer'),
    (30, r'\bdata\s*architect\b', 'Data Architect'),
    (30, r'\betl\s*developer\b', 'ETL Developer'),
    (30, r'\bdata\s*analyst\b', 'Data Analyst'),
    (30, r'\bdata\s*analytics\b', 'Data Analyst'),
    (30, r'\bbi\s*(developer|engineer|analyst)\b|\bbusiness\s*intelligence\b', 'BI Developer'),
    (30, r'\banalytics\s*engineer\b', 'Analytics Engineer'),
    (30, r'\bdatabase\s*(administrator|admin|dba)\b', 'Database Administrator'),
    (30, r'\bdatabase\s*(developer|engineer)\b', 'Database Developer'),
    (30, r'\bbusiness\s*analyst\b', 'Business Analyst'),
    (30, r'\bbusiness\s*systems\s*analyst\b', 'Business Systems Analyst'),
    (30, r'\bsystems\s*analyst\b', 'Systems Analyst'),
    
    # === CLOUD & INFRASTRUCTURE ===
    (40, r'\bcloud\s*(platform\s*)?engineer\b', 'Cloud Engineer'),
    (40, r'\bcloud\s*architect\b', 'Cloud Architect'),
    (40, r'\b(aws|azure|gcp)\s*(cloud\s*)?(engineer|architect)\b', 'Cloud Engineer'),
    (40, r'\baws\s*architect\b', 'Cloud Architect'),
    (40, r'\bcloud\s*security\s*engineer\b', 'Cloud Security Engineer'),
    (40, r'\bcloud\s*support\s*engineer\b', 'Cloud Support Engineer'),
    (40, r'\binfrastructure\s*(as\s*code\s*)?engineer\b', 'Infrastructure Engineer'),
    (40, r'\bplatform\s*engineer\b', 'Platform Engineer'),
    
    # === DEVOPS & SRE ===
    (50, r'\bdevsecops\s*engineer\b', 'DevSecOps Engineer'),
    (50, r'\bdevops\s*engineer\b|\bdev\s*ops\s*engineer\b', 'DevOps Engineer'),
    (50, r'\bdevops\s*architect\b', 'DevOps Architect'),
    (50, r'\bsre\b|\bsite\s*reliability\s*engineer\b', 'Site Reliability Engineer'),
    (50, r'\brelease\s*engineer\b', 'Release Engineer'),
    (50, r'\bbuild\s*engineer\b', 'Build Engineer'),
    (50, r'\bsystems?\s*administrator\b', 'Systems Administrator'),
    (50, r'\bsystems?\s*engineer\b', 'Systems Engineer'),
    (50, r'\bautomation\s*engineer\b', 'Automation Engineer'),
    
    # === SECURITY ===
    (60, r'\bcybersecurity\s*(engineer|analyst)\b', 'Cybersecurity Analyst'),
    (60, r'\bcyber\s*analyst\b', 'Cybersecurity Analyst'),
    (60, r'\bsoc\s*analyst\b', 'SOC Analyst'),
    (60, r'\bsecurity\s*operations\s*(analyst|engineer)\b', 'Security Operations Analyst'),
    (60, r'\bvulnerability\s*analyst\b', 'Vulnerability Analyst'),
    (60, r'\bsecurity\s*engineer\b', 'Security Engineer'),
    (60, r'\bsecurity\s*architect\b', 'Security Architect'),
    (60, r'\binfosec\s*engineer\b|\binformation\s*security\s*engineer\b', 'Information Security Engineer'),
    (60, r'\bsecurity\s*analyst\b', 'Security Analyst'),
    (60, r'\bpenetration\s*tester\b|\bpentest\b', 'Penetration Tester'),
    (60, r'\bsecurity\s*consultant\b', 'Security Consultant'),
    (60, r'\bapplication\s*security\b', 'Application Security Engineer'),
    
    # === QA & TESTING ===
    (70, r'\bsdet\b|\bsoftware\s*development\s*engineer\s*in\s*test\b', 'SDET'),
    (70, r'\bqa\s*automation\s*engineer\b', 'QA Automation Engineer'),
    (70, r'\bqa\s*engineer\b|\bquality\s*assurance\s*engineer\b', 'QA Engineer'),
    (70, r'\btest\s*engineer\b', 'Test Engineer'),
    (70, r'\btest\s*automation\s*engineer\b|\bautomation\s*test\s*engineer\b', 'Test Automation Engineer'),
    (70, r'\bqa\s*analyst\b|\bquality\s*analyst\b|\bquality\s*assurance\s*analyst\b', 'QA Analyst'),
    (70, r'\bqa\s*tester\b|\btester\b', 'QA Tester'),
    (70, r'\bperformance\s*(test\s*)?engineer\b', 'Performance Engineer'),
    (70, r'\bquality\s*engineer\b|\bsoftware\s*quality\s*engineer\b', 'Quality Engineer'),
    
    # === DESIGN ROLES ===
    (75, r'\bui/ux\s*designer\b', 'UI/UX Designer'),
    (75, r'\bux\s*designer\b', 'UX Designer'),
    (75, r'\bui\s*designer\b', 'UI Designer'),
    (75, r'\bproduct\s*designer\b', 'Product Designer'),
    (75, r'\bweb\s*designer\b', 'Web Designer'),
    (75, r'\bgame\s*designer\b', 'Game Designer'),
    (75, r'\bdesign\s*engineer\b', 'Design Engineer'),
    
    # === FULL STACK DEVELOPMENT ===
    (80, r'\bfull[\s-]*stack\s*(software\s*)?(developer|engineer)\b', 'Full Stack Developer'),
    (80, r'\bfull[\s-]*stack\b', 'Full Stack Developer'),
    
    # === FRONTEND DEVELOPMENT ===
    (90, r'\bfront[\s-]*end\s*(developer|engineer)\b', 'Frontend Developer'),
    (90, r'\bfrontend\s*(developer|engineer)\b', 'Frontend Developer'),
    (90, r'\breact\s*(developer|engineer)\b', 'Frontend Developer'),
    (90, r'\bangular\s*(developer|engineer)\b', 'Frontend Developer'),
    (90, r'\bvue\s*(developer|engineer|\.?js)\b', 'Frontend Developer'),
    (90, r'\bui\s*(developer|engineer)\b', 'UI Developer'),
    (90, r'\bui/ux\s*(developer|engineer)\b', 'UI/UX Developer'),
    (90, r'\bweb\s*developer\b', 'Web Developer'),
    
    # === BACKEND DEVELOPMENT ===
    (100, r'\bback[\s-]*end\s*(developer|engineer)\b', 'Backend Developer'),
    (100, r'\bbackend\s*(developer|engineer)\b', 'Backend Developer'),
    (100, r'\bnode\.?js\s*(developer|engineer)\b', 'Backend Developer'),
    (100, r'\bpython\s*(developer|engineer)\b', 'Python Developer'),
    (100, r'\bjava\s*(developer|engineer)\b(?!\s*script)', 'Java Developer'),
    (100, r'\b\.?net\s*(developer|engineer)\b', '.NET Developer'),
    (100, r'\bc\#\s*(developer|engineer)\b', '.NET Developer'),
    (100, r'\bruby\s*(developer|engineer)\b', 'Ruby Developer'),
    (100, r'\bgo(lang)?\s*(developer|engineer)\b', 'Go Developer'),
    (100, r'\brust\s*(developer|engineer)\b', 'Rust Developer'),
    (100, r'\bphp\s*(developer|engineer)\b', 'PHP Developer'),
    (100, r'\bscala\s*(developer|engineer)\b', 'Scala Developer'),
    (100, r'\bc\+\+\s*(developer|engineer)\b', 'C++ Developer'),
    
    # === MOBILE DEVELOPMENT ===
    (110, r'\bmobile\s*(developer|engineer)\b', 'Mobile Developer'),
    (110, r'\bandroid\s*(developer|engineer)\b', 'Android Developer'),
    (110, r'\bios\s*(developer|engineer)\b', 'iOS Developer'),
    (110, r'\bswift\s*(developer|engineer)\b', 'iOS Developer'),
    (110, r'\bkotlin\s*(developer|engineer)\b', 'Android Developer'),
    (110, r'\breact\s*native\s*(developer|engineer)\b', 'Mobile Developer'),
    (110, r'\bflutter\s*(developer|engineer)\b', 'Mobile Developer'),
    
    # === EMBEDDED & SYSTEMS ===
    (120, r'\bembedded\s*(software\s*)?(developer|engineer)\b', 'Embedded Software Engineer'),
    (120, r'\bfirmware\s*engineer\b', 'Firmware Engineer'),
    (120, r'\bhardware\s*engineer\b', 'Hardware Engineer'),
    (120, r'\biot\s*engineer\b', 'IoT Engineer'),
    
    # === SPECIALIZED ROLES ===
    (130, r'\bapi\s*(developer|engineer)\b', 'API Developer'),
    (130, r'\bintegration\s*engineer\b', 'Integration Engineer'),
    (130, r'\bgame\s*(developer|engineer)\b', 'Game Developer'),
    (130, r'\bgraphics\s*(software\s*)?(developer|engineer)\b', 'Graphics Engineer'),
    (130, r'\bblockchain\s*(developer|engineer)\b', 'Blockchain Developer'),
    (130, r'\bsalesforce\s*(administrator|developer)\b', 'Salesforce Admin/Developer'),
    (130, r'\bproduct\s*engineer\b', 'Product Engineer'),
    (130, r'\bapplication\s*engineer\b', 'Application Engineer'),
    (130, r'\bsolutions\s*engineer\b', 'Solutions Engineer'),
    (130, r'\bforward\s*deployed\s*engineer\b', 'Forward Deployed Engineer'),
    
    # === ARCHITECT ROLES ===
    (140, r'\bsoftware\s*architect\b', 'Software Architect'),
    (140, r'\bsolutions?\s*architect\b', 'Solutions Architect'),
    (140, r'\benterprise\s*architect\b', 'Enterprise Architect'),
    (140, r'\btechnical\s*architect\b', 'Technical Architect'),
    (140, r'\bsystems?\s*architect\b', 'Systems Architect'),
    (140, r'\bapplication\s*architect\b', 'Application Architect'),
    
    # === GENERIC SOFTWARE ROLES (lower priority - catch-all) ===
    (200, r'\bsoftware\s*development\s*engineer\b', 'Software Development Engineer'),
    (200, r'\bsoftware\s*design\s*engineer\b', 'Software Design Engineer'),
    (200, r'\bsenior\s*(software\s*)?(developer|engineer)\b', 'Senior Software Engineer'),
    (200, r'\bstaff\s*(software\s*)?(developer|engineer)\b', 'Staff Software Engineer'),
    (200, r'\bprincipal\s*(software\s*)?(developer|engineer)\b', 'Principal Software Engineer'),
    (200, r'\blead\s*(software\s*)?(developer|engineer)\b', 'Lead Software Engineer'),
    (200, r'\bjunior\s*(software\s*)?(developer|engineer)\b', 'Junior Software Engineer'),
    (200, r'\bentry[\s-]*level\s*(software\s*)?(developer|engineer)\b', 'Junior Software Engineer'),
    (200, r'\bsoftware\s*(developers|engineers)\b', 'Software Developer'),  # plurály
    (200, r'\bweb\s*developers\b', 'Web Developer'),  # plurál
    (200, r'\b(software\s*)?developer\b', 'Software Developer'),
    (200, r'\bsoftware\s*engineer\b', 'Software Engineer'),
    (200, r'\bprogrammer\b', 'Programmer'),
    (200, r'\bcoder\b', 'Programmer'),
    (200, r'\bengineer\b', 'Engineer'),
    
    # === IT / SUPPORT ROLES ===
    (210, r'\bit\s*analyst\b', 'IT Analyst'),
    (210, r'\bit\s*specialist\b', 'IT Specialist'),
    (210, r'\bit\s*consultant\b', 'IT Consultant'),
    (210, r'\btechnical\s*support\b|\btechnical\s*customer\s*support\b', 'Technical Support'),
    
    # === PROJECT / PRODUCT MANAGEMENT ===
    (15, r'\bproject\s*manager\b', 'Project Manager'),
    (15, r'\bproduct\s*owner\b', 'Product Owner'),
    (15, r'\bscrum\s*master\b', 'Scrum Master'),
    (15, r'\bdelivery\s*manager\b|\bit\s*delivery\s*manager\b', 'Delivery Manager'),
    (15, r'\bprogram\s*manager\b', 'Program Manager'),
    
    # === SECURITY ADDITIONAL ===
    (60, r'\b(grc|governance.*compliance)\s*analyst\b', 'GRC Analyst'),
    (60, r'\bcyber\s*threat\s*(intelligence\s*)?analyst\b', 'Threat Intelligence Analyst'),
    (60, r'\bthreat\s*intelligence\s*analyst\b', 'Threat Intelligence Analyst'),
    (60, r'\bcyber\s*(security\s*)?(risk|defense)\s*analyst\b', 'Cybersecurity Analyst'),
    (60, r'\binformation\s*assurance\s*analyst\b', 'Information Assurance Analyst'),
    (60, r'\bsecurity\s*compliance\s*analyst\b', 'Security Compliance Analyst'),
    (60, r'\bsecurity\s*operations\s*center\s*(\(soc\)\s*)?analyst\b', 'SOC Analyst'),
    (60, r'\bidentity\s*(and\s*)?access\s*management\s*analyst\b', 'IAM Analyst'),
    (60, r'\brisk\s*management\s*framework\s*analyst\b|\brmf\s*analyst\b', 'RMF Analyst'),
    
    # === GAME INDUSTRY ===
    (130, r'\btechnical\s*artist\b', 'Technical Artist'),
    (130, r'\bgame\s*producer\b', 'Game Producer'),
    (130, r'\btechnical\s*producer\b', 'Technical Producer'),
    (130, r'\blevel\s*designer\b', 'Level Designer'),
    (130, r'\bgame\s*artist\b', 'Game Artist'),
    (130, r'\b3d\s*artist\b', '3D Artist'),
    (130, r'\b3d\s*animator\b|\banimator\b', 'Animator'),
    (130, r'\buser\s*experience\s*designer\b', 'UX Designer'),
    
    # === ARCHITECT ADDITIONAL ===
    (140, r'\bmobile\s*architect\b', 'Mobile Architect'),
    (140, r'\bnetwork\s*architect\b', 'Network Architect'),
    (140, r'\binformation\s*architect\b', 'Information Architect'),
    (140, r'\baem\s*architect\b', 'AEM Architect'),
    
    # === CONSULTING ===
    (210, r'\bdevops\s*consultant\b', 'DevOps Consultant'),
    (210, r'\btechnical\s*consultant\b', 'Technical Consultant'),
    (210, r'\bsap\s*.*consultant\b', 'SAP Consultant'),
]

# Kompilované patterny pro efektivitu
_COMPILED_PATTERNS = None


def _compile_patterns():
    """Zkompiluje regex patterny pro rychlejší vyhodnocení."""
    global _COMPILED_PATTERNS
    if _COMPILED_PATTERNS is None:
        _COMPILED_PATTERNS = [
            (priority, re.compile(pattern, re.IGNORECASE), normalized)
            for priority, pattern, normalized in JOB_TITLE_PATTERNS
        ]
    return _COMPILED_PATTERNS


def normalize_job_title(job_title: str) -> str:
    """Normalizuje název pozice na standardní kategorii.
    
    Args:
        job_title: Originální název pozice
        
    Returns:
        Normalizovaný název pozice nebo 'Other' pokud není rozpoznán
    """
    if not job_title or not isinstance(job_title, str):
        return 'Other'
    
    job_title = job_title.strip()
    if not job_title:
        return 'Other'
    
    patterns = _compile_patterns()
    
    # Najdi první match podle priority
    matches = []
    for priority, pattern, normalized in patterns:
        if pattern.search(job_title):
            matches.append((priority, normalized))
    
    if matches:
        # Vrať match s nejnižší prioritou (nejvyšší priorita = nejnižší číslo)
        matches.sort(key=lambda x: x[0])
        return matches[0][1]
    
    return 'Other'


def normalize_job_titles_batch(job_titles: list) -> list:
    """Normalizuje seznam názvů pozic.
    
    Args:
        job_titles: Seznam originálních názvů pozic
        
    Returns:
        Seznam normalizovaných názvů pozic
    """
    return [normalize_job_title(title) for title in job_titles]


def get_job_title_categories() -> Dict[str, list]:
    """Vrátí slovník kategorií a jejich vzorů.
    
    Returns:
        Dict kde klíč je normalizovaný název a hodnota je seznam matchujících patternů
    """
    categories = {}
    for priority, pattern, normalized in JOB_TITLE_PATTERNS:
        if normalized not in categories:
            categories[normalized] = []
        categories[normalized].append(pattern)
    return categories


def analyze_job_titles(job_titles: list) -> Dict[str, int]:
    """Analyzuje distribuci normalizovaných pozic.
    
    Args:
        job_titles: Seznam originálních názvů pozic
        
    Returns:
        Dict s četností každé normalizované kategorie (seřazeno sestupně)
    """
    normalized = normalize_job_titles_batch(job_titles)
    counts = {}
    for title in normalized:
        counts[title] = counts.get(title, 0) + 1
    
    # Seřadit sestupně podle četnosti
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


if __name__ == "__main__":
    # Testovací příklady
    test_titles = [
        "Senior Software Engineer",
        "Full Stack Developer",
        "DevOps Engineer (Terraform)",
        "Cloud Security Engineer – GCP & AWS",
        "Machine Learning Engineer",
        "Data Scientist",
        "QA Engineer",
        "Frontend Developer",
        "Backend Developer",
        "iOS Developer",
        "Android Developer",
        "Software Development Engineer in Test (SDET)",
        "Junior Full Stack Developer",
        "Principal Software Engineer - Storage Systems",
        "Embedded Software Engineer (Wireless)",
        "Security Analyst",
        "Site Reliability Engineer (SRE) - AWS",
        "AI Software Developer",
        "Python Developer",
        "Java Developer - New York, NY",
        ".Net Developer",
        "Unknown Position Name XYZ",
    ]
    
    print("=== Test normalizace názvů pozic ===\n")
    for title in test_titles:
        normalized = normalize_job_title(title)
        print(f"  {title}")
        print(f"    -> {normalized}\n")
