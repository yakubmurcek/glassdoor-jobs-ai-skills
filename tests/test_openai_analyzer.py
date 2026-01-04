
import unittest
from unittest.mock import MagicMock, patch
from ai_skills.openai_analyzer import OpenAIJobAnalyzer
from ai_skills.models import (
    JobAnalysisResult, 
    BatchAnalysisResponse, 
    JobAnalysisResultWithId,
    AITierBatchResponse,
    AITierResultWithId,
    SkillsBatchResponse,
    SkillsResultWithId,
    EducationBatchResponse,
    EducationResultWithId,
)

class TestOpenAIJobAnalyzer(unittest.TestCase):
    def setUp(self):
        # Patch the OpenAI client creation so we don't need a real API key
        with patch('ai_skills.openai_analyzer.OpenAI'):
            # Test with legacy monolithic mode for backwards compatibility
            self.analyzer = OpenAIJobAnalyzer(api_key="fake-key", use_decomposed=False)
            # Mock the client.responses.parse method
            self.analyzer.client.responses = MagicMock()

    def test_analyze_texts_chunking_and_logic(self):
        """Test legacy monolithic batching still works."""
        # Create dummy inputs
        texts = ["text1", "text2", "text3"]
        titles = ["title1", "title2", "title3"]
        
        # Set a small batch size to force multiple batches
        self.analyzer.batch_size = 2
        
        # Mock the API response
        # Use real objects to pass isinstance checks
        mock_response_1 = MagicMock()
        mock_response_1.output = BatchAnalysisResponse(results=[
            JobAnalysisResultWithId(
                id="job_0", 
                ai_tier="none",
                ai_skills_mentioned=[],
                confidence=1.0, 
                rationale="Reason 1",
                hardskills_raw=["python"],
                softskills_raw=["teamwork"],
                min_education_level=None
            ),
            JobAnalysisResultWithId(
                id="job_1", 
                ai_tier="core_ai", 
                ai_skills_mentioned=["TensorFlow"],
                confidence=0.9, 
                rationale="Reason 2",
                hardskills_raw=["tensorflow", "python"],
                softskills_raw=[],
                min_education_level="Bachelor's"
            )
        ])
        
        mock_response_2 = MagicMock()
        mock_response_2.output = BatchAnalysisResponse(results=[
            JobAnalysisResultWithId(
                id="job_2", 
                ai_tier="applied_ai",
                ai_skills_mentioned=[],
                confidence=0.8, 
                rationale="Reason 3",
                hardskills_raw=["pytorch"],
                softskills_raw=["communication"],
                min_education_level=None
            )
        ])
        
        # side_effect allows returning different values for consecutive calls
        self.analyzer.client.responses.parse.side_effect = [mock_response_1, mock_response_2]

        # Run analysis (titles are used for input construction, but not returned in the simple result model)
        results = self.analyzer.analyze_texts(texts, job_titles=titles)

        # Verification
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].ai_tier.value, "none")
        self.assertEqual(results[1].ai_tier.value, "core_ai")
        self.assertEqual(results[2].ai_tier.value, "applied_ai")
        
        # Verify the client was called twice (batch size 2, total 3 items)
        self.assertEqual(self.analyzer.client.responses.parse.call_count, 2)
        print("Test passed: Legacy monolithic batching works correctly.")


class TestDecomposedAnalyzer(unittest.TestCase):
    """Test the new decomposed task-based batching."""
    
    def setUp(self):
        with patch('ai_skills.openai_analyzer.OpenAI'):
            self.analyzer = OpenAIJobAnalyzer(api_key="fake-key", use_decomposed=True)
            self.analyzer.client.responses = MagicMock()

    def test_decomposed_combines_task_results(self):
        """Test that decomposed mode combines results from 3 tasks correctly."""
        texts = ["ML Engineer job description", "Web dev job description"]
        titles = ["ML Engineer", "Web Developer"]
        
        # Mock responses for each task
        # Task 1: AI Tier
        tier_response = MagicMock()
        tier_response.output = AITierBatchResponse(results=[
            AITierResultWithId(id="job_0", ai_tier="applied_ai", confidence=0.9, rationale="ML work"),
            AITierResultWithId(id="job_1", ai_tier="none", confidence=0.95, rationale="Standard web dev"),
        ])
        
        # Task 2: Skills
        skills_response = MagicMock()
        skills_response.output = SkillsBatchResponse(results=[
            SkillsResultWithId(id="job_0", ai_skills_mentioned=["tensorflow"], hardskills_raw=["python", "tensorflow"], softskills_raw=["teamwork"]),
            SkillsResultWithId(id="job_1", ai_skills_mentioned=[], hardskills_raw=["javascript", "react"], softskills_raw=["communication"]),
        ])
        
        # Task 3: Education (using correct field names)
        edu_response = MagicMock()
        edu_response.output = EducationBatchResponse(results=[
            EducationResultWithId(id="job_0", min_education_level="Master's", min_years_experience=3.0),
            EducationResultWithId(id="job_1", min_education_level=None, min_years_experience=None),
        ])
        
        # Return different responses for each task call
        self.analyzer.client.responses.parse.side_effect = [
            tier_response,    # AI Tier task
            skills_response,  # Skills task
            edu_response,     # Education task
        ]
        
        results = self.analyzer.analyze_texts(texts, job_titles=titles)
        
        # Verify results are combined correctly
        self.assertEqual(len(results), 2)
        
        # Job 0: ML Engineer
        self.assertEqual(results[0].ai_tier.value, "applied_ai")
        self.assertEqual(results[0].confidence, 0.9)
        self.assertEqual(results[0].ai_skills_mentioned, ["tensorflow"])
        self.assertEqual(results[0].hardskills_raw, ["python", "tensorflow"])
        self.assertEqual(results[0].min_education_level, "Master's")
        self.assertEqual(results[0].min_years_experience, 3.0)
        
        # Job 1: Web Developer
        self.assertEqual(results[1].ai_tier.value, "none")
        self.assertEqual(results[1].ai_skills_mentioned, [])
        self.assertEqual(results[1].hardskills_raw, ["javascript", "react"])
        self.assertEqual(results[1].min_education_level, None)
        
        # Verify 3 API calls (one per task)
        self.assertEqual(self.analyzer.client.responses.parse.call_count, 3)
        print("Test passed: Decomposed task-based batching combines results correctly.")


if __name__ == '__main__':
    unittest.main()
