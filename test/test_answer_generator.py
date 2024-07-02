from argparse import ArgumentParser
import csv, json
from model import (
    MistralAnswerGenerator,
    LLaMA3AnswerGenerator,
    PaLMAnswerGenerator,
    GPT4AnswerGenerator,
)
from utils import evaluate_example, load_csv_data, get_test_question
import tensorflow as tf
import tqdm
import PIL

# If using CPU for tensorflow
# tf.config.set_visible_devices([], "GPU")
from utils import load_csv_data, get_test_question, get_image


def run_vqa(
    question_generator, test_file, retrieval_results_file, output_file
):
    test_list, test_header = load_csv_data(test_file)
    if retrieval_results_file:
        retrieval_results = json.load(open(retrieval_results_file, "r"))
    else:
        retrieval_results = None
    result_list = []
    for it, example in tqdm.tqdm(enumerate(test_list)):
        question = get_test_question(it, test_list, test_header)
        # data_id = "E-VQA_{}".format(it)
        data_id = question["data_id"]
        if retrieval_results:
            answer = question_generator.llm_answering(question=question["question"], entry_section=retrieval_results["reranked_sections"][0])
        else:
            answer = question_generator.llm_answering(question=question["question"])

        result_list.append(
            {
                "data_id": data_id,
                "prediction": answer,
            }
        )
    with open(output_file, "w") as f:
        json.dump(result_list, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--retrieval_results", type=str)
    parser.add_argument("--answer_generator", type=str)
    parser.add_argument("--llm_checkpoint", type=str)
    parser.add_argument("--output_file", type=str, default="answer.json")
    
    args = parser.parse_args()
    test_file = args.test_file
    retrieval_results = args.retrieval_results
    vqa_results = args.vqa_results
    output_file = args.output_file
    if args.answer_generator.lower() == "mistral":
        answer_generator = MistralAnswerGenerator(model_path=args.llm_checkpoint,device="cuda")
    elif args.answer_generator.lower() == "llama3":
        answer_generator = LLaMA3AnswerGenerator(model_path=args.llm_checkpoint,device="cuda")
    elif args.answer_generator.lower() == "gpt4":
        answer_generator = GPT4AnswerGenerator()
    elif args.answer_generator.lower() == "palm":
        answer_generator = PaLMAnswerGenerator()
    else:
        raise ValueError("Invalid Answer Generator, Please choose from Mistral, LLaMA3, GPT4, PaLM")
    run_vqa(answer_generator, test_file, retrieval_results, output_file)
