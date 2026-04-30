from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import llama_binary


def import_llama_cli():
    model_management = types.ModuleType("comfy.model_management")
    model_management.processing_interrupted = lambda: False
    model_management.throw_exception_if_processing_interrupted = lambda: None

    comfy = types.ModuleType("comfy")
    comfy.model_management = model_management
    package = types.ModuleType("llm_text_processor_testpkg")
    package.__path__ = [str(PROJECT_ROOT)]

    with mock.patch.dict(
        sys.modules,
        {
            "comfy": comfy,
            "comfy.model_management": model_management,
            "llm_text_processor_testpkg": package,
            "llm_text_processor_testpkg.llama_binary": llama_binary,
        },
    ):
        sys.modules.pop("llm_text_processor_testpkg.llama_cli", None)
        return importlib.import_module("llm_text_processor_testpkg.llama_cli")


class ParseResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.llama_cli = import_llama_cli()

    def test_extracts_reasoning_after_terminal_spinner(self) -> None:
        text = (
            "> Красная машина врезается в столб     ... (truncated)\n\n"
            "|\b \b[Start thinking]\n"
            "Thinking Process:\n"
            "Analyze the request.\n"
            "[End thinking]\n\n"
            "A cinematic red car crash prompt.\n\n"
            "Exiting...\n"
        )

        response, reasoning, perf = self.llama_cli._parse_response(text)

        self.assertEqual(response, "A cinematic red car crash prompt.")
        self.assertEqual(reasoning, "Thinking Process:\nAnalyze the request.")
        self.assertEqual(perf, "")

    def test_extracts_normal_response_and_perf(self) -> None:
        text = (
            "> Красная машина врезается в столб     ... (truncated)\n\n"
            "A cinematic red car crash prompt.\n\n"
            "[ Prompt: 213.3 t/s | Generation: 9.3 t/s ]\n\n"
            "Exiting...\n"
        )

        response, reasoning, perf = self.llama_cli._parse_response(text)

        self.assertEqual(response, "A cinematic red car crash prompt.")
        self.assertEqual(reasoning, "")
        self.assertEqual(perf, "[ Prompt: 213.3 t/s | Generation: 9.3 t/s ]")


if __name__ == "__main__":
    unittest.main()
