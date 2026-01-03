from ai_skills.skill_normalizer import get_semantic_normalizer

def test_categorization():
    normalizer = get_semantic_normalizer()
    
    # Skills identified as problematic by the user
    test_cases = {
        "oop": "Software Engineering",
        "aws": "Data & Cloud",
        "azure": "Data & Cloud",
        "react": "Programming",
        "node.js": "Programming"
    }
    
    print("Checking categorizations...")
    results = normalizer.categorize_skills(list(test_cases.keys()))
    
    failed = False
    for skill, expected_family in test_cases.items():
        actual = results.get(skill)
        print(f"Skill: {skill:10} | Expected: {expected_family:20} | Actual: {actual}")
        if actual != expected_family:
            failed = True
            
    if failed:
        print("\nFAILURE: Some skills were misclassified.")
        exit(1)
    else:
        print("\nSUCCESS: All skills classified correctly.")
        exit(0)

if __name__ == "__main__":
    test_categorization()
