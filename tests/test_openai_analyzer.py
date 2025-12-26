
import unittest
from unittest.mock import MagicMock, patch
from ai_skills.openai_analyzer import OpenAIJobAnalyzer
from ai_skills.models import JobAnalysisResult, BatchAnalysisResponse, JobAnalysisResultWithId

class TestOpenAIJobAnalyzer(unittest.TestCase):
    def setUp(self):
        # Patch the OpenAI client creation so we don't need a real API key
        with patch('ai_skills.openai_analyzer.OpenAI'):
            self.analyzer = OpenAIJobAnalyzer(api_key="fake-key")
            # Mock the client.responses.parse method
            self.analyzer.client.responses = MagicMock()

    def test_analyze_texts_chunking_and_logic(self):
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
                education_required=0
            ),
            JobAnalysisResultWithId(
                id="job_1", 
                ai_tier="core_ai", 
                ai_skills_mentioned=["TensorFlow"],
                confidence=0.9, 
                rationale="Reason 2",
                hardskills_raw=["tensorflow", "python"],
                softskills_raw=[],
                education_required=1
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
                education_required=0
            )
        ])
        
        # side_effect allows returning different values for consecutive calls
        self.analyzer.client.responses.parse.side_effect = [mock_response_1, mock_response_2]

        # Run analysis (titles are used for input construction, but not returned in the simple result model)
        results = self.analyzer.analyze_texts(texts, job_titles=titles)

        # Verification
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].ai_tier, "none")
        self.assertEqual(results[1].ai_tier, "core_ai")
        self.assertEqual(results[2].ai_tier, "applied_ai")
        
        # Verify the client was called twice (batch size 2, total 3 items)
        self.assertEqual(self.analyzer.client.responses.parse.call_count, 2)
        print("Test passed: Logic flow handles batching and response parsing correctly.")

if __name__ == '__main__':
    unittest.main()
