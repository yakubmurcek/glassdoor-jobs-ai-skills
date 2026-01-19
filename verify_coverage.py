from ai_skills.skills_dictionary import HARDSKILLS, SKILL_TO_FAMILY

total_skills = len(HARDSKILLS)
mapped_skills = len(SKILL_TO_FAMILY)
missing_skills = HARDSKILLS - set(SKILL_TO_FAMILY.keys())

print(f"Total Unique Hardskills: {total_skills}")
print(f"Mapped Skills: {mapped_skills}")

if missing_skills:
    print(f"CRITICAL: {len(missing_skills)} skills are NOT mapped to any family!")
    print(f"Missing Examples: {list(missing_skills)[:20]}")
    exit(1)
else:
    print("SUCCESS: All skills are correctly mapped to a family.")
