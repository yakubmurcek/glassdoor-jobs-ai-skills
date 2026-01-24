
from ai_skills.education_extractor import extract_lowest_education

examples = [
    "Bachelor's degree, Master's degree",
    "Master's degree preferred, Bachelor's degree required",
    "Ph.D. or Master's degree",
    "High School Diploma or GED",
    "Associate's degree"
]

print("--- Testing Educational Extraction Logic ---")
print("Logic should pick the LOWEST level (Minimum Requirement)\n")

for text in examples:
    result = extract_lowest_education(text)
    print(f"Input: '{text}'\nDetected: '{result}'\n")
