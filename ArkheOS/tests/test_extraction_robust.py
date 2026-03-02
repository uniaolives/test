import sys
import os
import asyncio
import unittest
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'ArkheOS/src'))

from arkhe.extraction import (
    GeminiExtractor, OllamaExtractor, LongDocumentProcessor, ExtractionReport,
    FinancialFact, Currency
)

class TestExtractionRobust(unittest.IsolatedAsyncioTestCase):
    async def test_gemini_extraction_success(self):
        extractor = GeminiExtractor(api_key="fake")
        report = await extractor.extract("some text", ExtractionReport)
        self.assertIsNotNone(report)
        self.assertEqual(report.model_used, "gemini-2.0-flash")

    async def test_ollama_extraction_success(self):
        extractor = OllamaExtractor()
        report = await extractor.extract("some text", ExtractionReport)
        self.assertIsNotNone(report)
        self.assertEqual(report.model_used, "llama3")

    async def test_retry_mechanism(self):
        extractor = GeminiExtractor(api_key="fake")
        extractor.max_retries = 2
        # Mocking to fail once then succeed
        calls = 0
        async def mocked_call(prompt):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise Exception("Network Error")
            return '{"facts": [], "document_name": "test", "model_used": "gemini-2.0-flash"}'

        with patch.object(GeminiExtractor, '_call_llm_internal', side_effect=mocked_call):
            # Speed up tests by patching sleep
            with patch('asyncio.sleep', return_value=None):
                report = await extractor.extract("text", ExtractionReport)
                self.assertIsNotNone(report)
                self.assertEqual(calls, 2)

    async def test_validation_retry(self):
        extractor = OllamaExtractor()
        extractor.max_retries = 2
        calls = 0
        async def mocked_call(prompt):
            nonlocal calls
            calls += 1
            if calls == 1:
                return '{"invalid": "json"}'
            return '{"facts": [], "document_name": "test", "model_used": "llama3"}'

        with patch.object(OllamaExtractor, '_call_llm_internal', side_effect=mocked_call):
            with patch('asyncio.sleep', return_value=None):
                report = await extractor.extract("text", ExtractionReport)
                self.assertIsNotNone(report)
                self.assertEqual(calls, 2)

    async def test_long_document_processing(self):
        extractor = GeminiExtractor(api_key="fake")
        # Overriding simulation to return different facts per chunk if we wanted,
        # but here we just test if gather works.
        processor = LongDocumentProcessor(extractor, chunk_size=10, overlap=2)
        text = "1234567890abcdefghij" # 20 chars -> multiple chunks

        report = await processor.process_document(text, "long_doc")
        self.assertIsNotNone(report)
        self.assertEqual(report.document_name, "long_doc")
        # In current simulation, each chunk returns the same fact, so deduplication keeps 1.
        self.assertEqual(len(report.facts), 1)

if __name__ == '__main__':
    unittest.main()
