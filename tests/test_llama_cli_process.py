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


class RunLlamaCliProcessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.llama_cli = import_llama_cli()

    def test_starts_process_in_new_session(self) -> None:
        process = mock.Mock()
        process.returncode = 0
        process.poll.return_value = 0
        process.communicate.return_value = (
            "... (truncated)\nGenerated response.\n[ Prompt: 1.0 t/s | Generation: 2.0 t/s ]",
            "",
        )

        with mock.patch.object(self.llama_cli.subprocess, "Popen", return_value=process) as popen:
            response, reasoning, perf = self.llama_cli.run_llama_cli(
                command=["llama-cli"],
                timeout_seconds=10,
            )

        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertEqual(response, "Generated response.")
        self.assertEqual(reasoning, "")
        self.assertEqual(perf, "[ Prompt: 1.0 t/s | Generation: 2.0 t/s ]")


if __name__ == "__main__":
    unittest.main()
