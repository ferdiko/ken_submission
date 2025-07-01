'''CLI to generate tokens to feed in to an LLM.

Checks if requested input with num_tokens already exists. If not, generates
approximately the number of tokens requested and dumps to a file.

Example:
    python3 generate_tokens.py 10 --force-overwrite

    Regenerates a string of approximately 10 tokens and dumps it default location `inputs/10_tokens`.
'''
import argparse
import os
import logging
import random

# Directory to save the input tokens to.
INPUT_DIRECTORY = "inputs"

logger = logging.getLogger(__name__)

def generate_input_tokens(num_tokens, output=None, force_overwrite=False):
    '''Generates and dumps tokens to file and returns file name.'''
    # Validate inputs.
    if num_tokens <= 0:
        raise ValueError("num_tokens must be an integer greater than 0.")

    # TODO(sarahyw): Generalize this.
    # Construct output file name.
    output_path = output if output is not None else f"{INPUT_DIRECTORY}/{num_tokens}_tokens"

    # If force-overwrite is true of file does not already exist, generate new tokens file.
    if force_overwrite or not os.path.exists(output_path):
        logger.info(f"Generating {num_tokens} tokens to file {output_path}.")
        dump_input_tokens(output_path, num_tokens)
        logger.info("Finished generating tokens.")
    else:
        logger.info(f"File {output_path} already exists. Skipping token generation.")

    return output_path


# Generates tokens and saves them to file. Creates or overwrites file.
def dump_input_tokens(output_path, total_tokens):
    with open(output_path, 'w') as f:
        token_string = generate_input_string(total_tokens)
        f.write(token_string)


def generate_input_string(num_tokens: int):
    '''Generates a string with approximately num_tokens tokens.
    
    Assumes that generally 1 token = 4 characters = "abc " (note the whitespace after abc).
    '''
    # TODO(sarahyw): Currently for each input sequence, the each token is the same string. May want to vary it.
    token = "abc"
    # Generate a list of random tokens
    tokens = [token for _ in range(num_tokens)]
    # Join tokens with spaces to form the final string
    return ' '.join(tokens)


def read_input_tokens_from_file(input_token_file_name):
    """Read tokens from the specified file into a list of strings."""
    with open(input_token_file_name, 'r') as f:
        tokens_list = [line.strip() for line in f.readlines()]
    return tokens_list


#######################################
## Helper functions for verification ##
#######################################

def generate_random_prompts(tokenizer, in_tokens):
    """Copied from https://github.com/ferdiko/vllm/blob/main/examples/run_single_server.py#L121"""
    prompts = []
    prompt_token_ids = []

    for n in in_tokens:
        # prompt_tokens = list(np.random.randint(0, 2000, size=n))
        prompt_tokens = [random.randint(0, 2000) for _ in range(n)]
        prompts.append(tokenizer.decode(prompt_tokens))
        prompt_token_ids.append(prompt_tokens)

    return prompts, prompt_token_ids


##################################################
## Main function for original input generation. ##
##################################################

def main():
    parser = argparse.ArgumentParser()

    # Required positional argument num_tokens defines approximately how many input tokens to use.
    parser.add_argument(
        "num_tokens",
        type=int,
        help="Number of tokens to generate.")
    
    # Optional argument output defines where to save the generated input string to.
    parser.add_argument(
        "-output",
        "-o",
        help="Output file name.")

    # Provide force-overwrite if regenerating the existing file is wanted. Defaults false.
    parser.add_argument(
        "--force-overwrite",
        help="increase output verbosity",
        action="store_true")

    args = parser.parse_args()

    generate_input_tokens(args.num_tokens, args.output, args.force_overwrite)


if __name__ == "__main__":
    main()
