# PART 3 – an evaluation script to test the accuracy of the fine-tuned model vs the base model.

import os
import json
from together import AsyncTogether
from dataclasses import dataclass

from autoblocks.testing.run import run_test_suite
from autoblocks.testing.models import BaseTestCase
from autoblocks.testing.run import grid_search_ctx
from autoblocks.testing.util import md5
from autoblocks.testing.evaluators import BaseAccuracy

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
        for test_case in eval_data
    ]


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


class Accuracy(BaseAccuracy):
    """
    Use the Autoblocks accuracy evaluator to compare the ground truth to the model's output.
    """
    id = "accuracy"
    model="gpt-4o"

    def output_mapper(self, output: str) -> str:
        return output
    
    def expected_output_mapper(self, test_case: TestCase) -> str:
        return test_case.expected_output


# Run the test suite using the Autoblocks SDK
# npx autoblocks testing exec -- python3 3-eval.py
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