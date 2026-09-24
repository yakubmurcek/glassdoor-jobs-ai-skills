import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_skills.config import AI_SKILLS, REAL_AI_SKILLS
from ai_skills.skills_dictionary import (
    HARDSKILL_VARIANTS, SOFTSKILL_VARIANTS, SKILL_TO_FAMILY
)

print(f"AI_SKILLS length: {len(AI_SKILLS)} (Codebook claims ~900)")
print(f"REAL_AI_SKILLS length: {len(REAL_AI_SKILLS)} (Codebook claims ~47)")
print(f"HARDSKILL_VARIANTS variants length: {len(HARDSKILL_VARIANTS)} (Codebook claims ~650)")
print(f"HARDSKILL_VARIANTS canonical length: {len(set(HARDSKILL_VARIANTS.values()))} (Codebook claims ~400)")
print(f"SOFTSKILL_VARIANTS variants length: {len(SOFTSKILL_VARIANTS)} (Codebook claims ~200)")
print(f"SOFTSKILL_VARIANTS canonical length: {len(set(SOFTSKILL_VARIANTS.values()))} (Codebook claims ~80)")
print(f"SKILL_TO_FAMILY mapped skills: {len(SKILL_TO_FAMILY)}")
print(f"SKILL_TO_FAMILY unique families: {len(set(SKILL_TO_FAMILY.values()))} (Codebook claims 24)")

try:
    from ai_skills.cli import _JOB_FAMILY_PATTERNS
    print(f"_JOB_FAMILY_PATTERNS length: {len(_JOB_FAMILY_PATTERNS)} (Codebook claims 10)")
except ImportError:
    print("Could not import _JOB_FAMILY_PATTERNS from cli")

try:
    from ai_skills.cli import _NACE_MAPPING
    print(f"_NACE_MAPPING length: {len(_NACE_MAPPING)}")
except ImportError:
    print("Could not import _NACE_MAPPING from cli")
