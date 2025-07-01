import csv
import os
from itertools import cycle
from vllm import LLM, SamplingParams
import time


def prompt_for_question(q_shot, o_shot, a_shot, q_actual, o_actual):
    """
    q, o, a are question options answer for the prompt
    """
    prompt = "You have to pick the best answer in a multiple choice question by only outputing the letter. \
        You are given some examples and then the actual question\n"
    for q, o, a in zip(q_shot, o_shot, a_shot):
        prompt += f"Question: {q}\nOptions:\n"
        for i, oi in zip(["A", "B", "C", "D"], o):
            prompt += f"{i}: {oi}\n"
        prompt += f"Answer: {a}\n"

    prompt += f"Question: {q_actual}\nOptions:\n"
    for i, oi in zip(["A", "B", "C", "D"], o_actual):
        prompt += f"{i}: {oi}\n"
    prompt += f"Answer: "

    return prompt


if __name__ == "__main__":
    # Define the folder containing your csv files
    folder_path = "mmlu_dataset/raw"

    # Initialize lists for questions, options, and answers
    questions = []
    options = []
    answers = []

    # List all CSV files in the specified folder
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]

    # Open all files and create CSV readers
    file_readers = [open(file, 'r') for file in csv_files]
    csv_readers = [csv.reader(f) for f in file_readers]

    # Perform round-robin reading
    try:
        for reader in cycle(csv_readers):
            # Read the next line from each file in round-robin fashion
            row = next(reader, None)
            if row:
                # Parse the row
                question = row[0]
                option = row[1:5]  # Extract options as a list
                answer = row[5]

                # Append to respective lists
                questions.append(question)
                options.append(option)
                answers.append(answer)

            # Stop round-robin if all files are exhausted
            if all(reader is None or next(reader, None) is None for reader in csv_readers):
                break
    finally:
        # Close all files
        for f in file_readers:
            f.close()

    # Form prompts.
    shots = 5
    num_samples = 1000

    few_show_questions = questions[:5]
    few_show_options = options[:5]
    few_show_answers = answers[:5]

    questions = questions[5:num_samples+shots]
    options = options[5:num_samples+shots]
    answers = answers[5:num_samples+shots]

    indices = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3
    }

    prompts = []
    for q, o in zip(questions, options):
        prompts.append(prompt_for_question(few_show_questions, 
                                            few_show_options, 
                                            few_show_answers,
                                            q,
                                            o))

    # Load model and predict.
    sampling_params = SamplingParams(best_of=1,
                                    temperature=0.0,
                                    top_p=1,
                                    top_k=-1,
                                    # use_beam_search=False,
                                    max_tokens=1,
                                    presence_penalty=0,
                                    frequency_penalty=0,
                                    min_tokens=1)

    llm = LLM(
        # model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        # model="meta-llama/Llama-3.2-3B-Instruct",
        # model="AMead10/Llama-3.2-3B-Instruct-AWQ",
        # model="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        # model="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        model="meta-llama/Llama-3.2-1B-Instruct",
        tensor_parallel_size=1,
        max_model_len=10000,
        # max_num_batched_tokens=max_batched_req,
        # max_num_seqs=max_batched_req,
        # enable_chunked_prefill=True,
        # gpu_memory_utilization=0.45,
        enforce_eager=True, # TODO
        )

    print("\n"*5)

    start = time.time()
    for p, a in zip(prompts, answers):
        llm.generate([p], sampling_params=sampling_params, use_tqdm=False)
        print(f",{indices[a]}")

    print("TIME:", time.time() - start)