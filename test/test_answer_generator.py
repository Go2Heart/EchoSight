import csv, json
from model import (
    MistralAnswerGenerator,
    LLaMA3AnswerGenerator,
    PaLMAnswerGenerator,
    OpenAIAnswerGenerator,
)
from utils import evaluate_example, load_csv_data, get_test_question
import tensorflow as tf
import tqdm
import PIL

# If using CPU for tensorflow
# tf.config.set_visible_devices([], "GPU")
from utils import load_csv_data, get_test_question, get_image


def main(
    question_generator, test_file, kb_file, retrieval_results_file, vqa_results=None
):
    test_list, test_header = load_csv_data(test_file)
    kb_dict = json.load(open(kb_file, "r"))
    retrieval_results = json.load(open(retrieval_results_file, "r"))
    if vqa_results:
        vqa_results = json.load(open(vqa_results, "r"))
        result_dict = {}
        for result in vqa_results:
            result_dict[result["data_id"]] = result["answer"]
    eval_score = 0
    result_list = []
    for it, example in tqdm.tqdm(enumerate(test_list)):
        question = get_test_question(it, test_list, test_header)
        ground_truth = question["wikipedia_url"]
        target_answer = question["answer"].split("|")
        evidence_section_id = question["evidence_section_id"]
        # data_id = "E-VQA_{}".format(it)
        data_id = question["data_id"]
        if vqa_results is not None:
            answer = result_dict[data_id]
        else:
            answer = question_generator.llm_answering(question=question["question"])
            # If provided with retrieval_results
            # answer = question_generator.llm_answering(question=question["question"], entry_section=retrieval_results["reranked_sections"][0])

        result_list.append(
            {
                "data_id": data_id,
                "prediction": answer,
            }
        )
    with open("answer.json", "w") as f:
        json.dump(result_list, f, indent=4)


if __name__ == "__main__":
    test_file = "/PATH/TO/TEST_FILE"
    kb_file = "/PATH/TO/KNOWLEDGE_BASE_JSON"

    retrieval_results = "reranker_resultes.json"
    vqa_results = "evqa_vqa.json"
    answer_generator = PaLMAnswerGenerator()

    main(answer_generator, test_file, kb_file, retrieval_results, None)
