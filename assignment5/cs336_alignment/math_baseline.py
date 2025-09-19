from typing import *
from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import pandas as pd

"""
A conversation between User and Assistant. The User asks a question, and the Assistant
 solves it. The Assistant first thinks about the reasoning process in the mind and
 then provides the User with the answer. The reasoning process is enclosed within
 <think> </think> and answer is enclosed within <answer> </answer> tags, respectively,
 i.e., <think> reasoning process here </think> <answer> answer here </answer>.
 User: {question}
 Assistant: <think>
"""
def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    questions: List[str],
    answers: List[str],
    eval_smapling_params: SamplingParams,
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics,and serialize results to disk.
    """
    data = {
        "questions": [],
        "outputs": [],
        "format_rewards": [],
        "answer_rewards": [],
        "rewards": [],
    }
    outputs = vllm_model.generate(inputs=prompts, sampling_params=eval_smapling_params)
    for output, answer, question in zip(outputs, answers, questions):
        print("Start a")
        reward = reward_fn(output, answer)
        data["questions"].append(question)
        data["outputs"].append(output)
        data["format_rewards"].append(reward["format_reward"])
        data["answer_rewards"].append(reward["answer_reward"])
        data["rewards"].append(reward["reward"])
    # treat every column as a dict
    df = pd.DataFrame(data, orient="records", lines=True)
    df.to_json("data/eval_vllm.json")
    

def main():
    df = pd.read_parquet("data/countdown/0000.parquet")
    questions = df.head()["nums"]
    answers = df.head()["target"]
    prompts = "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. \nUser: " \
                + questions \
                + "\n Assistant: <think>"
    sampling_params = SamplingParams(
        temperature=1.0, 
        top_p=1.0, 
        max_tokens=1024, 
        stop=["</answer>"], 
        include_stop_str_in_output=True
    )
    vllm_model = LLM("Qwen/Qwen2.5-Math-1.5B")
    evaluate_vllm(
        vllm_model, 
        r1_zero_reward_fn, 
        prompts, 
        questions, 
        answers, 
        sampling_params
        )

if __name__ == "__main__":
    main()