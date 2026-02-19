# -*- coding: utf-8 -*-
"""Job Title Normalizer - Sjednocení názvů pozic do homogenních skupin.

Tento modul mapuje různorodé názvy pozic na standardizované kategorie
pro statistickou analýzu a agregaci dat.
"""

import re
from typing import Dict, List, Optional, Tuple

# =============================================================================
# DEFINICE HOMOGENNÍCH SKUPIN POZIC
# =============================================================================

# Priority patterns - jsou vyhodnoceny v pořadí (první match vyhrává)
# Formát: (priority, regex_pattern, normalized_name)
JOB_TITLE_PATTERNS: List[Tuple[int, str, str]] = [
    # === MANAGEMENT & LEADERSHIP ===
    (10, r'\b(cto|chief\s*technology\s*officer)\b', 'Engineering Manager'),
    (10, r'\b(cio|chief\s*information\s*officer)\b', 'Engineering Manager'),
    (10, r'\b(ciso|chief\s*information\s*security\s*officer)\b', 'CISO'),
    (10, r'\bdirector\b.*\b(software|engineering|development|it)\b', 'Engineering Manager'),
    (10, r'\bvp\b.*\b(engineering|technology)\b', 'VP of Engineering'),
    (10, r'\bhead\s*of\b.*\b(engineering|development)\b', 'Engineering Manager'),
    (10, r'\bengineering\s*manager\b|\bmanager.*software\s*engineering\b', 'Engineering Manager'),
    (10, r'\bsoftware\s*development\s*manager\b', 'Engineering Manager'),
    (10, r'\btechnical\s*lead\b|\btech\s*lead\b', 'Tech Lead'),
    (10, r'\bteam\s*lead\b.*\b(software|developer|engineer)\b', 'Tech Lead'),
    (10, r'\bfounding\s*engineer\b', 'Engineer'),
    (10, r'\bassociate\s*director\b.*\b(engineer|architect|tech|software|it)\b', 'Engineering Manager'),
    (10, r'\b(engineering|development)\s*director\b', 'Engineering Manager'),
    
    # === AI / ML / DATA SCIENCE ===
    (20, r'\b(ai|artificial\s*intelligence)\s*(software\s*)?(engineer|developer)\b', 'AI/ML Engineer'),
    (20, r'\bml\s*engineer\b|\bmachine\s*learning\s*engineer\b', 'AI/ML Engineer'),
    (20, r'\bmlops\s*engineer\b', 'AI/ML Engineer'),
    (20, r'\bdata\s*scientist\b', 'AI/ML Engineer'),
    (20, r'\bresearch\s*(scientist|engineer)\b.*\b(ml|ai|machine\s*learning)\b', 'AI/ML Research Scientist'),
    (20, r'\b(nlp|natural\s*language)\s*engineer\b', 'NLP Engineer'),
    (20, r'\bcomputer\s*vision\s*engineer\b', 'Engineer'),
    (20, r'\bai\s*architect\b', 'Solutions Architect'),
    (20, r'\bai\s*automation\s*engineer\b', 'AI/ML Engineer'),
    (20, r'\bai\s*(trainer|specialist)\b', 'AI/ML Engineer'),
    
    # === DATA ENGINEERING & ANALYTICS ===
    (30, r'\bdata\s*engineer\b', 'Data Engineer'),
    (30, r'\bdata\s*architect\b', 'Solutions Architect'),
    (30, r'\betl\s*developer\b', 'Software Developer'),
    (30, r'\bdata\s*analyst\b', 'Data Engineer'),
    (30, r'\bdata\s*analytics\b', 'Data Engineer'),
    (30, r'\bbi\s*(developer|engineer|analyst)\b|\bbusiness\s*intelligence\b', 'Software Developer'),
    (30, r'\banalytics\s*engineer\b', 'AI/ML Engineer'),
    (30, r'\bdatabase\s*(administrator|admin|dba)\b', 'Database Administrator'),
    (30, r'\b(sql\s*server\s*|mysql\s*|postgres\s*|oracle\s*)?dba\b', 'Database Administrator'),
    (30, r'\bdatabase\s*(developer|engineer)\b', 'Database Developer'),
    (30, r'\bdatabase\s*architect\b', 'Solutions Architect'),
    (30, r'\bdatabase\s*(analyst|designer)\b', 'Database Developer'),
    (30, r'\bdatabase\s*administration\b', 'Database Administrator'),
    (30, r'\bbusiness\s*analyst\b', 'IT Analyst'),
    (30, r'\bbusiness\s*systems\s*analyst\b', 'IT Analyst'),
    (30, r'\bsystems\s*analyst\b', 'IT Analyst'),
    
    # === CLOUD & INFRASTRUCTURE ===
    (40, r'\bcloud\s*(platform\s*)?engineer\b', 'Cloud Engineer'),
    (40, r'\bcloud\s*architect\b', 'Solutions Architect'),
    (40, r'\b(aws|azure|gcp)\s*(cloud\s*)?(engineer|architect|developer|consultant|specialist)\b', 'Cloud Engineer'),
    (40, r'\baws\s*architect\b', 'Solutions Architect'),
    (40, r'\bcloud\s*(devops|infrastructure)\s*engineer\b', 'Cloud Engineer'),
    (40, r'\bcloud\s*security\s*engineer\b', 'Security Engineer'),
    (40, r'\bcloud\s*support\s*engineer\b', 'IT Analyst'),
    (40, r'\binfrastructure\s*(as\s*code\s*)?engineer\b', 'Infrastructure Engineer'),
    (40, r'\bplatform\s*engineer\b', 'Platform Engineer'),
    (40, r'\bcloud\s*(engineering|devops)\b', 'Cloud Engineer'),
    
    # === DEVOPS & SRE ===
    (50, r'\bdevsecops\s*engineer\b', 'DevSecOps Engineer'),
    (50, r'\bdevops\s*engineer\b|\bdev\s*ops\s*engineer\b', 'DevOps Engineer'),
    (50, r'\bdevops\s*architect\b', 'Solutions Architect'),
    (50, r'\bdevops\s*(analyst|consultant|coach|lead|manager|specialist|associate)\b', 'DevOps Engineer'),
    (50, r'\bdevops\b', 'DevOps Engineer'),
    (50, r'\bsre\b|\bsite\s*reliability\s*engineer\b', 'Site Reliability Engineer'),
    (50, r'\brelease\s*engineer\b', 'Release Engineer'),
    (50, r'\bbuild\s*engineer\b', 'Build Engineer'),
    (50, r'\bsystems?\s*administrator\b', 'IT Analyst'),
    (50, r'\bsystems?\s*engineers?\b', 'Systems Engineer'),
    (50, r'\bautomation\s*(architect|engineer)\b', 'Automation Engineer'),
    
    # === SECURITY ===
    (60, r'\bcybersecurity\s*(engineer|analyst|specialist)\b', 'Cybersecurity Analyst'),
    (60, r'\bcyber\s*security\s*(engineer|analyst|specialist)\b', 'Cybersecurity Analyst'),
    (60, r'\bcyber\s*analyst\b', 'Cybersecurity Analyst'),
    (60, r'\bcybersecurity\s*architect\b', 'Security Engineer'),
    (60, r'\bcyber\s*security\s*architect\b', 'Security Engineer'),
    (60, r'\bsoc\s*analyst\b', 'Security Analyst'),
    (60, r'\bsecurity\s*operations\s*(analyst|engineer)\b', 'Security Analyst'),
    (60, r'\bvulnerability\s*(assessment\s*)?analyst\b', 'Security Analyst'),
    (60, r'\bsecurity\s*engineer\b', 'Security Engineer'),
    (60, r'\bsecurity\s*architect\b', 'Security Engineer'),
    (60, r'\binfosec\s*engineer\b|\binformation\s*security\s*engineer\b', 'Security Engineer'),
    (60, r'\bsecurity\s*analyst\b', 'Security Analyst'),
    (60, r'\b(cyber|network|information|it|cloud|data)\s*security\b', 'Security Engineer'),
    (60, r'\bsecurity\s*(specialist|lead|manager|consultant|director)\b', 'Security Engineer'),
    (60, r'\bpenetration\s*tester\b|\bpentest\b', 'Security Analyst'),
    (60, r'\bsecurity\s*consultant\b', 'Security Consultant'),
    (60, r'\bapplication\s*security\b', 'Security Engineer'),
    (60, r'\bsecurity\s*engineering\b', 'Security Engineer'),
    
    # === QA & TESTING ===
    (70, r'\bsdet\b|\bsoftware\s*development\s*engineer\s*in\s*test\b', 'SDET'),
    (70, r'\bqa\s*automation\s*engineer\b', 'QA Automation Engineer'),
    (70, r'\bqa\s*engineer\b|\bquality\s*assurance\s*engineer\b', 'QA Engineer'),
    (70, r'\btest\s*engineer\b', 'Test Engineer'),
    (70, r'\btest\s*automation\s*(engineer|architect)\b|\bautomation\s*test\s*engineer\b', 'Test Automation Engineer'),
    (70, r'\bqa\s*analyst\b|\bquality\s*analyst\b|\bquality\s*assurance\s*analyst\b', 'QA Engineer'),
    (70, r'\bqa\s*(test\s*)?analyst\b', 'QA Engineer'),
    (70, r'\bqa\s*tester\b|\btester\b', 'QA Tester'),
    (70, r'\bqa\s*(manager|lead|director)\b', 'QA Engineer'),
    (70, r'\b(qa|quality)\s*testing\s*lead\b', 'QA Engineer'),
    (70, r'\bquality\s*assurance\s*(manager|lead|director|specialist|test)\b', 'QA Engineer'),
    (70, r'\bquality\s*assurance\b', 'QA Engineer'),
    (70, r'\buat\s*(testing|coordinator|analyst|tester)\b', 'QA Tester'),
    (70, r'\bperformance\s*(test\s*)?engineer\b', 'Engineer'),
    (70, r'\bquality\s*engineer\b|\bsoftware\s*quality\s*engineer\b', 'Quality Engineer'),
    (70, r'\bquality\s*engineering\b', 'Quality Engineer'),
    
    # === DESIGN ROLES ===
    (75, r'\bui/ux\s*designer\b', 'UI/UX Designer'),
    (75, r'\bux\s*designer\b', 'UI/UX Designer'),
    (75, r'\bui\s*designer\b', 'UI/UX Designer'),
    (75, r'\bproduct\s*designer\b', 'UI/UX Designer'),
    (75, r'\bweb\s*designer\b', 'UI/UX Designer'),
    (75, r'\bweb\s*\&?\s*graphic\s*designer\b', 'UI/UX Designer'),
    (75, r'\bgraphic\s*\&?\s*web\s*design\b', 'UI/UX Designer'),
    (75, r'\bwebsite\s*designer\b', 'UI/UX Designer'),
    (75, r'\bgame\s*designer\b', 'Game Development'),
    (75, r'\bgame\s*design\b', 'Game Development'),
    (75, r'\bdesign\s*engineer\b', 'Engineer'),
    
    # === FULL STACK DEVELOPMENT ===
    (80, r'\bfull[\s-]*stack\s*(software\s*)?(developer|engineer)\b', 'Full Stack Developer'),
    (80, r'\bfull[\s-]*stack\b', 'Full Stack Developer'),
    
    # === FRONTEND DEVELOPMENT ===
    (90, r'\bfront[\s-]*end\s*(developer|engineer)s?\b', 'Frontend Developer'),
    (90, r'\bfrontend\s*(developer|engineer)s?\b', 'Frontend Developer'),
    (90, r'\bfront[\s-]*end\s*(swe|web|ui)\b', 'Frontend Developer'),
    (90, r'\bfrontend\s*engineering\b', 'Frontend Developer'),
    (90, r'\breact\s*(developer|engineer|\.?js|dev)\b', 'Frontend Developer'),
    (90, r'\bangular\s*(developer|engineer)\b', 'Frontend Developer'),
    (90, r'\bvue\s*(developer|engineer|\.?js)\b', 'Frontend Developer'),
    (90, r'\bui\s*(developer|engineer)\b', 'UI/UX Designer'),
    (90, r'\bui/ux\s*(developer|engineer)\b', 'UI/UX Designer'),
    (90, r'\bweb\s*developer\b', 'Web Developer'),
    (90, r'\bweb\s*development\b', 'Web Developer'),
    
    # === BACKEND DEVELOPMENT ===
    (100, r'\bback[\s-]*end\s*(developer|engineer)\b', 'Backend Developer'),
    (100, r'\bbackend\s*(developer|engineer)\b', 'Backend Developer'),
    (100, r'\bnode\.?js\s*(developer|engineer)\b', 'Backend Developer'),
    (100, r'\bpython\s*(developer|engineer)\b', 'Software Developer'),
    (100, r'\bjava\s*(developer|engineer)\b(?!\s*script)', 'Java Developer'),
    (100, r'\b\.?net\s*(developer|engineer)\b', '.NET Developer'),
    (100, r'\bc\#\s*(developer|engineer)\b', '.NET Developer'),
    (100, r'\bruby\s*(developer|engineer)\b', 'Software Developer'),
    (100, r'\bgo(lang)?\s*(developer|engineer)\b', 'Software Developer'),
    (100, r'\brust\s*(developer|engineer)\b', 'Software Developer'),
    (100, r'\bphp\s*(developer|engineer)\b', 'Software Developer'),
    (100, r'\bscala\s*(developer|engineer)\b', 'Software Developer'),
    (100, r'\bc\+\+\s*(developer|engineer)\b', 'Software Developer'),
    
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
    (120, r'\bhardware\s*engineer\b', 'Engineer'),
    (120, r'\biot\s*engineer\b', 'IoT Engineer'),
    
    # === SPECIALIZED ROLES ===
    (130, r'\bapi\s*(developer|engineer)\b', 'Software Developer'),
    (130, r'\bintegration\s*engineer\b', 'Engineer'),
    (130, r'\bgame\s*(developer|engineer)\b', 'Game Development'),
    (130, r'\bgame\s*(analyst|producer|streamer)\b', 'Game Development'),
    (130, r'\bgraphics\s*(software\s*)?(developer|engineer)\b', 'Engineer'),
    (130, r'\bblockchain\s*(developer|engineer)\b', 'Software Developer'),
    (130, r'\bsalesforce\s*(administrator|developer|admin|consultant|architect|lead|engineer|specialist|analyst)\b', 'Salesforce Admin/Developer'),
    (130, r'\bsalesforce\b', 'Salesforce Admin/Developer'),
    (130, r'\bcrm\s*(developer|architect|analyst|lead|consultant|engineer|specialist|manager)\b', 'Software Developer'),
    (130, r'\bproduct\s*engineer\b', 'Engineer'),
    (130, r'\bapplication\s*engineer\b', 'Engineer'),
    (130, r'\bsolutions\s*engineer\b', 'Engineer'),
    (130, r'\bforward\s*deployed\s*engineer\b', 'Engineer'),
    
    # === ARCHITECT ROLES ===
    (140, r'\bsoftware\s*architect\b', 'Software Architect'),
    (140, r'\bsolutions?\s*architect\b', 'Solutions Architect'),
    (140, r'\benterprise\s*architect\b', 'Solutions Architect'),
    (140, r'\btechnical\s*architect\b', 'Solutions Architect'),
    (140, r'\bsystems?\s*architect\b', 'Solutions Architect'),
    (140, r'\bapplication\s*architect\b', 'Solutions Architect'),
    
    # === GENERIC SOFTWARE ROLES (lower priority - catch-all) ===
    (200, r'\bsoftware\s*development\s*engineer\b', 'Software Development Engineer'),
    (200, r'\bsoftware\s*design\s*engineer\b', 'Software Design Engineer'),
    (200, r'\bsenior\s*(software\s*)?(developer|engineer)s?\b', 'Senior Software Engineer'),
    (200, r'\bstaff\s*(software\s*)?(developer|engineer)s?\b', 'Staff Software Engineer'),
    (200, r'\bprincipal\s*(software\s*)?(developer|engineer)s?\b', 'Principal Software Engineer'),
    (200, r'\blead\s*(software\s*)?(developer|engineer)s?\b', 'Lead Software Engineer'),
    (200, r'\bjunior\s*(software\s*)?(developer|engineer)s?\b', 'Junior Software Engineer'),
    (200, r'\bentry[\s-]*level\s*(software\s*)?(developer|engineer)s?\b', 'Junior Software Engineer'),
    (200, r'\bsoftware\s*engineers?\b', 'Software Engineer'),
    (200, r'\bsoftware\s*developers?\b', 'Software Developer'),
    (200, r'\bsoftware\s*engineering\b', 'Software Engineer'),
    (200, r'\bswe\b', 'Software Engineer'),
    (200, r'\bweb\s*developers?\b', 'Web Developer'),
    (200, r'\b(software\s*)?developers?\b', 'Software Developer'),
    (200, r'\bprogrammer\b', 'Programmer'),
    (200, r'\bcoder\b', 'Programmer'),
    (200, r'\bengineers?\b', 'Engineer'),
    
    # === IT / SUPPORT ROLES ===
    (210, r'\bit\s*analyst\b', 'IT Analyst'),
    (210, r'\bit\s*specialist\b', 'IT Analyst'),
    (210, r'\bit\s*consultant\b', 'IT Consultant'),
    (210, r'\bit\s*technician\b', 'IT Analyst'),
    (210, r'\bit\s*support\b', 'IT Analyst'),
    (210, r'\btechnical\s*support\b|\btechnical\s*customer\s*support\b', 'IT Analyst'),
    
    # === PROJECT / PRODUCT MANAGEMENT ===
    (15, r'\bproject\s*manager\b', 'Engineering Manager'),
    (15, r'\bproduct\s*owner\b', 'Engineering Manager'),
    (15, r'\bproduct\s*manager\b', 'Engineering Manager'),
    (15, r'\bscrum\s*master\b', 'Engineering Manager'),
    (15, r'\bdelivery\s*manager\b|\bit\s*delivery\s*manager\b', 'Engineering Manager'),
    (15, r'\bprogram\s*manager\b', 'Engineering Manager'),
    
    # === SECURITY ADDITIONAL ===
    (60, r'\b(grc|governance.*compliance)\s*analyst\b', 'Security Analyst'),
    (60, r'\bcyber\s*threat\s*(intelligence\s*)?analyst\b', 'Security Analyst'),
    (60, r'\bthreat\s*intelligence\s*analyst\b', 'Security Analyst'),
    (60, r'\bcyber\s*(security\s*)?(risk|defense)\s*analyst\b', 'Cybersecurity Analyst'),
    (60, r'\binformation\s*assurance\s*analyst\b', 'Security Analyst'),
    (60, r'\bsecurity\s*compliance\s*analyst\b', 'Security Analyst'),
    (60, r'\bsecurity\s*operations\s*center\s*(\(soc\)\s*)?analyst\b', 'Security Analyst'),
    (60, r'\bidentity\s*(and\s*)?access\s*management\s*analyst\b', 'Security Analyst'),
    (60, r'\brisk\s*management\s*framework\s*analyst\b|\brmf\s*analyst\b', 'Security Analyst'),
    
    # === GAME INDUSTRY ===
    (130, r'\btechnical\s*artist\b|\btech\s*artist\b', 'Game Development'),
    (130, r'\bgame\s*producer\b', 'Game Development'),
    (130, r'\bproducer.*\bgame\b|\bproducer.*\bgaming\b', 'Game Development'),
    (130, r'\btechnical\s*producer\b', 'Game Development'),
    (130, r'\blevel\s*designer\b', 'Game Development'),
    (130, r'\bgame\s*artist\b', 'Game Development'),
    (130, r'\b(2d|3d|environment|character|concept|vfx|marketing)\s*artist\b', 'Game Development'),
    (130, r'\b3d\s*(animator|modeler)\b|\banimator\b', 'Game Development'),
    (130, r'\buser\s*experience\s*designer\b', 'UI/UX Designer'),
    
    # === ARCHITECT ADDITIONAL ===
    (140, r'\bmobile\s*architect\b', 'Solutions Architect'),
    (140, r'\bnetwork\s*architect\b', 'Solutions Architect'),
    (140, r'\binformation\s*architect\b', 'Solutions Architect'),
    (140, r'\baem\s*architect\b', 'Solutions Architect'),
    
    # === CONSULTING ===
    (210, r'\bdevops\s*consultant\b', 'IT Analyst'),
    (210, r'\btechnical\s*consultant\b', 'IT Analyst'),
    (210, r'\bsap\s*.*consultant\b', 'IT Analyst'),
    (210, r'\b(java|python|\.net|c\+\+|ruby|php|scala|aws|azure|gcp)\s*consultant\b', 'IT Analyst'),
    (210, r'\bsolutions\s*consultant\b', 'Solutions Architect'),
    
    # === QUALIFIED ARCHITECT CATCH-ALL ===
    # Only match 'architect' when preceded by a tech qualifier
    (220, r'\b(java|python|pega|api|test|automation|data|crm|web|content|cybersecurity|cyber\s*security)\s*architect\b', 'Software Architect'),
    (220, r'\barchitect\b.*\b(software|cloud|data|security|system|infrastructure|application|solution)\b', 'Software Architect'),
    (220, r'\b(software|cloud|data|security|system|infrastructure|application|solution)\b.*\barchitect\b', 'Software Architect'),
    
    # === QUALIFIED CONSULTANT CATCH-ALL ===
    (220, r'\b(cloud|devops|engineering|software|it|technology|digital)\s*.*\bconsultant\b', 'IT Analyst'),
    
    # === PRODUCER (games/tech) ===
    (220, r'\b(senior\s*)?producer\b', 'Game Development'),

    # ================================================================
    # SECOND WAVE CATCH-ALLS (priority 225-240)
    # Broader but still safe within this tech/software job dataset
    # ================================================================
    
    # --- Architect (bare) — safe in tech jobs context ---
    (225, r'\barchitect\b', 'Software Architect'),
    
    # --- Tech-qualified analysts ---
    (225, r'\b(it|software|system|application|devops|cloud|risk|compliance|technology|technical|digital|automation|infrastructure|platform)\b.*\banalyst\b', 'IT Analyst'),
    (225, r'\banalyst\b.*\b(it|software|system|application|devops|cloud|risk|compliance|technology|technical|digital|automation|infrastructure|platform|qa|test|security|cyber)\b', 'IT Analyst'),
    
    # --- Tech-qualified managers ---
    (225, r'\b(engineering|software|qa|it|technical|development|cloud|security|infrastructure|technology|game|mobile|platform|devops|architecture|test|web|application|data)\b.*\bmanager\b', 'Engineering Manager'),
    (225, r'\bmanager\b.*\b(engineering|software|qa|it|technical|development|cloud|security|infrastructure|technology|game|mobile|platform|devops|test|web|application|data)\b', 'Engineering Manager'),
    
    # --- Tech-qualified leads ---
    (225, r'\b(software|engineering|development|technical|qa|test|it|security|data|web|cloud|devops|application|crm|automation|integration|backend|frontend|mobile|platform)\b.*\blead\b', 'Tech Lead'),
    (225, r'\blead\b.*\b(software|engineering|development|technical|qa|test|it|security|data|web|cloud|devops|application|crm|automation|integration|backend|frontend|mobile|platform|developer|engineer|architect)\b', 'Tech Lead'),
    
    # --- Tech-qualified specialists ---
    (225, r'\b(software|it|security|cloud|web|database|data|qa|test|application|technical|devops|api|network|infrastructure|platform|technology)\b.*\bspecialist\b', 'IT Analyst'),
    (225, r'\bspecialist\b.*\b(software|it|security|cloud|web|database|data|qa|test|application|technical|devops|api|network|infrastructure|platform|technology|engineering|development)\b', 'IT Analyst'),
    
    # --- Software/engineering variants ---
    (225, r'\bsoftware\s*(eningeer|engineeer|enigneer|enigeer)\b', 'Software Engineer'),
    (225, r'\bsoftware\s*(tester|testers)\b', 'QA Tester'),
    (225, r'\bsoftware\s*(technician|analyst|specialist)\b', 'Software Developer'),
    (225, r'\bsoftware\s*development\s*(lead|team\s*lead|manager|snr\s*manager)\b', 'Engineering Manager'),
    (225, r'\bsoftware\s*development\s*(analyst|specialist)\b', 'Software Developer'),
    (225, r'\bsoftware\s*development\s*in\s*test\b', 'SDET'),
    (225, r'\bvice\s*president\b.*\bengineering\b', 'Engineering Manager'),
    
    # --- QA/test variants ---
    (225, r'\bqa\s*(automation|jd)\b', 'QA Engineer'),
    (225, r'\btest\s*(lead|manager|analyst)\b', 'QA Engineer'),
    (225, r'\bquality\s*control\b', 'QA Engineer'),
    
    # --- Web variants ---
    (225, r'\bwebsite\s*(builder|support|administrator)\b', 'Web Developer'),
    (225, r'\bweb\s*(application|platform|site)\b', 'Web Developer'),
    (225, r'\bweb\s*administrator\b', 'Web Developer'),
    
    # --- Security variants ---
    (225, r'\bsoc\b.*\banalyst\b|\bsecurity\s*(operation|monitoring|admin)\b', 'Security Analyst'),
    (225, r'\bthreat\s*(research|analyst)\b', 'Security Analyst'),
    
    # --- Cloud variants ---
    (225, r'\b(aws|azure|gcp|cloud)\s*(sysops|dev|admin|administrator|networking|infra)\b', 'Cloud Engineer'),
    (225, r'\baws\b|\bazure\b|\bgcp\b', 'Cloud Engineer'),
    
    # --- Database variants ---
    (225, r'\bdatabase\s*(support|professional|administratior)\b', 'Database Administrator'),
    (225, r'\bsql\b.*\b(etl|develop)\b', 'Software Developer'),
    (225, r'\bdatabase\b', 'Database Developer'),
    
    # --- Mobile variants ---
    (225, r'\bmobile\s*(app|ui|lead)\b', 'Mobile Developer'),
    (225, r'\b(ios|android)\s*(mobile|app)\b', 'Mobile Developer'),
    (225, r'\bmobile\s*develiper\b', 'Mobile Developer'),
    
    # --- Data/AI variants ---
    (225, r'\bdata\s*(modeler|migration|management|strategy)\b', 'Data Engineer'),
    
    # --- Game variants ---
    (225, r'\bgame\s*(mathematician|programming|development|scout)\b', 'Game Development'),
    
    # --- Artist (bare) — safe in game/tech context ---
    (225, r'\bartist\b', 'Game Development'),
    
    # --- Co-op/intern/fresher ---
    (230, r'\b(co-op|coop)\b', 'Junior Software Engineer'),
    (230, r'\bintern\b', 'Junior Software Engineer'),
    (230, r'\bfresher\b', 'Junior Software Engineer'),
    
    # --- Consultant (bare) ---
    (230, r'\bconsultant\b', 'IT Analyst'),
    
    # --- IT/Technology bare ---
    (230, r'\binformation\s*technology\s*(technician|specialist|associate|professional|assistant)\b', 'IT Analyst'),
    (230, r'\bit\s*(professional|assistant|operations|delivery|admin)\b', 'IT Analyst'),
    (230, r'\bnetwork\s*administrator\b', 'IT Analyst'),
    
    # --- Application roles ---
    (230, r'\bapplication\s*(support|developer|development|analyst)\b', 'Software Developer'),
    
    # --- Technical staff / MTS ---
    (240, r'\b(senior|staff|principal|lead|junior)\s*(member\s*of\s*)?technical\s*staff\b', 'Software Engineer'),
    (240, r'\bmember\s*of\s*technical\s*staff\b', 'Software Engineer'),
    (240, r'\bdevelopment\s*(consultant|manager|lead|supervisor)\b', 'Engineering Manager'),
    (240, r'\bui\s*(/|or\s*)ux\b', 'UI/UX Designer'),
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


def _preprocess_title(title: str) -> str:
    """Vyčistí noise z názvu pozice před matchováním.
    
    Odstraní lokace, level čísla, podtržítka, a expanduje zkratky.
    """
    t = title.strip()
    # Replace underscores with spaces
    t = t.replace('_', ' ')
    # Remove trailing parenthetical codes: (REF1607W), (Revised 09-02-25), (Contract)
    t = re.sub(r'\s*\([^)]*\)\s*$', '', t)
    # Remove trailing location: ' - City, ST' or ' – City, ST'
    t = re.sub(r'\s*[-–]\s*[A-Z][a-z]+(\s+[A-Z][a-z]+)*,\s*[A-Z]{2}(\s*,\s*USA)?\s*$', '', t)
    # Remove trailing ' - Remote', ' – Remote', etc.
    t = re.sub(r'\s*[-–]\s*(Remote|Hybrid|On-?site|100%\s*Remote)\s*$', '', t, flags=re.IGNORECASE)
    # Remove trailing level numbers: ' I', ' II', ' III', ' IV', ' 1', ' 2', ' 3', ' 4'
    t = re.sub(r'\s+(I{1,3}|IV|[1-4])\s*$', '', t)
    # Remove trailing hash numbers like '#1'
    t = re.sub(r'\s*#\d+\s*$', '', t)
    # Handle Sr./Snr → Senior, Jr. → Junior
    t = re.sub(r'\bSr\.?\s', 'Senior ', t, flags=re.IGNORECASE)
    t = re.sub(r'\bSnr\.?\s', 'Senior ', t, flags=re.IGNORECASE)
    t = re.sub(r'\bJr\.?\s', 'Junior ', t, flags=re.IGNORECASE)
    # Remove trailing revision/date info: '1470 (Revised...)'
    t = re.sub(r'\s+\d{3,}\s*$', '', t)
    # Clean up extra whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    return t


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
    
    # Pre-process: strip noise
    job_title = _preprocess_title(job_title)
    
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
