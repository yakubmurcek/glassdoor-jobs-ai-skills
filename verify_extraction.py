import sys
import logging
from ai_skills.deterministic_extractor import extract_hardskills_deterministic

# Set up simple logging
logging.basicConfig(level=logging.INFO)

test_cases = [
    ("SQL Server", ["sql server"]),
    ("SQL and SQL Server", ["sql", "sql server"]),
    ("React Native", ["react native"]),
    ("React and React Native", ["react", "react native"]),
    ("API Development", ["api development"]),
    ("API and API Development", ["api", "api development"]),
    ("Java Script", ["javascript"]), # "Java Script" variant -> "javascript"
    ("Java and JavaScript", ["java", "javascript"]),
]

def run_tests():
    logging.info("Starting extraction verification...")
    all_passed = True
    
    for text, expected in test_cases:
        # Expected is sorted canonical list
        expected = sorted(expected)
        
        result = extract_hardskills_deterministic(text)
        
        if result == expected:
            logging.info(f"PASS: '{text}' -> {result}")
        else:
            logging.error(f"FAIL: '{text}' -> Expected {expected}, Got {result}")
            all_passed = False
            
    if all_passed:
        logging.info("All test cases passed!")
    else:
        logging.error("Some test cases failed.")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
