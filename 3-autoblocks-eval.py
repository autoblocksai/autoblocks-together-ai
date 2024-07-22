import os
import json
from typing import List
from together import AsyncTogether
from textwrap import dedent
from dataclasses import dataclass

from autoblocks._impl.testing.models import EvaluationOverride
from autoblocks.testing.run import run_test_suite
from autoblocks.testing.models import BaseTestCase
from autoblocks.testing.run import grid_search_ctx
from autoblocks.testing.util import md5
from autoblocks.testing.models import Threshold
from autoblocks.testing.evaluators import BaseLLMJudge
from autoblocks.testing.models import ScoreChoice

async_together_client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))

base_model = "meta-llama/Llama-3-8b-chat-hf"
top_oss_model = "meta-llama/Llama-3-70b-chat-hf"
finetuned_model = "YOUR_FINETUNED_MODEL_ID"
evaluator_model = "meta-llama/Llama-3-70b-chat-hf"
eval_dataset = "EvalDataset-100.json"


@dataclass
class TestCase(BaseTestCase):
    instruction: str
    expected_output: str

    def hash(self) -> str:
        """
        This hash serves as a unique identifier for a test case throughout its lifetime.
        """
        return md5(self.instruction)


with open(eval_dataset, "r", encoding="utf-8") as eval_file:
    """
    Initialize the eval dataset as test cases.
    """
    eval_data = json.load(eval_file)
    test_cases = [
        TestCase(instruction=test_case["instruction"], expected_output=test_case["output"])
        for test_case in eval_data[:30]
    ]


class Accuracy(BaseLLMJudge):
    id = "accuracy"
    threshold = Threshold(gte=1)
    model="gpt-4o"
    score_choices = [
        ScoreChoice(
            value=0,
            name="Not accurate",
        ),
        ScoreChoice(
            value=0.5,
            name="Somewhat accurate",
        ),
        ScoreChoice(
            value=1,
            name="Accurate",
        )
    ]

    def make_prompt(self, test_case: TestCase, output: str, recent_overrides: List[EvaluationOverride]) -> str:
        """
        Use an LLM as a judge to determine if the output is accurate based on the expected output.
        Once you've manually reviewed in Autoblocks, you can use recent_overrides to provide examples to the LLM judge. 
        """
        return dedent(
            f"""
                Is the output accurate based on the expected output?
                It should only be considered accurate if the expected output and output match.

                [Output]
                {output}

                [Expected Output]
                {test_case.expected_output}
            """).strip()


async def test_fn(test_case: TestCase) -> str:
    """
    Using the current model from the grid search context, generate a completion for the test case.
    """
    ctx = grid_search_ctx()
    model = ctx.get("model")
    completion = await async_together_client.chat.completions.create(
        messages=[
            {"role": "user", "content": test_case.instruction},
        ],
        model=model,
        max_tokens=1500,
    )
    return completion.choices[0].message.content


# Run the test suite using the Autoblocks SDK
# npx autoblocks testing exec -- python3 3-autoblocks-eval.py
run_test_suite(
    id="autoblocks-together-ai",
    test_cases=test_cases,
    evaluators=[Accuracy()],
    fn=test_fn,
    grid_search_params=dict(
        model=[base_model, top_oss_model],
    ),
    max_test_case_concurrency=5,
)