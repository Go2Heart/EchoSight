from argparse import ArgumentParser
import json, tqdm
import torch
from model import (
    ClipRetriever,
    MistralAnswerGenerator,
    GPT4AnswerGenerator,
    reconstruct_wiki_article,
    PaLMAnswerGenerator,
    reconstruct_wiki_sections,
    WikipediaKnowledgeBaseEntry,
    BgeTextReranker,
)
from utils import load_csv_data, get_test_question, get_image, remove_list_duplicates
import PIL

iNat_image_path = "/PATH/TO/INAT_ID2NAME"


def eval_recall(candidates, ground_truth, top_ks=[1, 5, 10, 20, 100]):
    recall = {k: 0 for k in top_ks}
    for k in top_ks:
        if ground_truth in candidates[:k]:
            recall[k] = 1
    return recall


def run_test(
    test_file_path: str,
    knowledge_base_path: str,
    faiss_index_path: str,
    top_ks: list,
    retrieval_top_k: int,
    **kwargs
):
    test_list, test_header = load_csv_data(test_file_path)
    with open(iNat_image_path + "/val_id2name.json", "r") as f:
        iNat_id2name = json.load(f)

    if kwargs["resume_from"] is not None:
        resumed_results = json.load(open(kwargs["resume_from"], "r"))
        kb_dict = json.load(open(knowledge_base_path, "r"))
    else:
        retriever = ClipRetriever(device="cuda:0", model=kwargs["retriever_vit"])
        # retriever.save_knowledge_base_faiss(knowledge_base_path, scores_path=score_dict, save_path=faiss_index_path)
        retriever.load_knowledge_base(knowledge_base_path)
        retriever.load_faiss_index(faiss_index_path)

    recalls = {k: 0 for k in top_ks}
    reranked_recalls = {k: 0 for k in top_ks}

    hits = 0

    if kwargs["perform_vqa"]:
        from utils import evaluate_example
        import tensorflow as tf

        tf.config.set_visible_devices([], "GPU")  # disable GPU for tensorflow
        question_generator = MistralAnswerGenerator(
            model_path="/remote-home/share/huggingface_model/Mistral-7B-Instruct-v0.2",
            device="cuda:0",
            use_embedding_model=False,
        )
    if kwargs["perform_text_rerank"]:
        text_reranker = BgeTextReranker(
            model_path="/remote-home/share/huggingface_model/bge-reranker-v2-m3",
            device="cuda:0",
        )
        eval_score = 0

    if kwargs["perform_qformer_reranker"]:
        from lavis.models import load_model_and_preprocess

        blip_model, vis_processors, txt_processors = load_model_and_preprocess(
            name="blip2_reranker", model_type="pretrain", is_eval=True, device="cuda"
        )
        checkpoint_path = kwargs["qformer_ckpt_path"]

        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        msg = blip_model.load_state_dict(checkpoint, strict=False)
        blip_model = blip_model.half()
        blip_model.use_vanilla_qformer = True
        print("Missing keys {}".format(msg.missing_keys))
        from data_utils import squarepad_transform, targetpad_transform

        preprocess = targetpad_transform(1.25, 224)

    metric = "url matching"

    retrieval_result = {}
    for it, test_example in tqdm.tqdm(enumerate(test_list)):
        example = get_test_question(it, test_list, test_header)
        image = PIL.Image.open(
            get_image(
                example["dataset_image_ids"].split("|")[0],
                example["dataset_name"],
                iNat_id2name,
            )
        )
        ground_truth = example["wikipedia_url"]
        target_answer = example["answer"].split("|")
        if example["dataset_name"] == "infoseek":
            data_id = example["data_id"]
        else:
            data_id = "E-VQA_{}".format(it)
        print("wiki_url: ", example["wikipedia_url"])
        print("question: ", example["question"])
        if kwargs["resume_from"] is not None:
            resumed_result = resumed_results[data_id]
            top_k_wiki, retrieval_simlarities = resumed_result["retrieved_entries"]
            reranked_sections = resumed_result["reranked_sections"]
            retrieval_simlarities = retrieval_simlarities
            entries = [WikipediaKnowledgeBaseEntry(kb_dict[url]) for url in top_k_wiki]
        else:
            top_k = retriever.retrieve_image_faiss(image, top_k=retrieval_top_k)
            top_k_wiki = [retrieved_entry["url"] for retrieved_entry in top_k]
            top_k_wiki = remove_list_duplicates(top_k_wiki)
            entries = [retrieved_entry["kb_entry"] for retrieved_entry in top_k]
            entries = remove_list_duplicates(entries)
            seen = set()
            retrieval_simlarities = [
                top_k[i]["similarity"]
                for i in range(retrieval_top_k)
                if not (top_k[i]["url"] in seen or seen.add(top_k[i]["url"]))
            ]

        if kwargs["save_result"]:
            retrieval_result[data_id] = {
                "retrieved_entries": [entry.url for entry in entries[:20]],
                "retrieval_similarities": [
                    sim.item() for sim in retrieval_simlarities[:20]
                ],
            }
        if metric == "answer matching":
            entry_articles = [reconstruct_wiki_article(entry) for entry in entries]
            found = False
            for i, entry in enumerate(entry_articles):
                for answer in target_answer:
                    if answer.strip().lower() in entry.strip().lower():
                        found = True
                        break
                if found:
                    break
            if found:
                for k in top_ks:
                    if i < k:
                        recalls[k] += 1

        else:
            # in url_matching
            recall = eval_recall(top_k_wiki, ground_truth, top_ks)
            for k in top_ks:
                recalls[k] += recall[k]
        for k in top_ks:
            print("Avg Recall@{}: ".format(k), recalls[k] / (it + 1))

        if kwargs["perform_qformer_reranker"]:
            reference_image = preprocess(image).to("cuda").unsqueeze(0)
            sections = []
            section_to_entry = []
            for entry_id, entry in enumerate(entries):
                entry_sections = reconstruct_wiki_sections(entry)
                sections.extend(entry_sections)
                section_to_entry.extend([entry_id] * len(entry_sections))

            qformer_question = example["question"]
            qformer_articles = [txt_processors["eval"](article) for article in sections]
            with torch.cuda.amp.autocast():
                fusion_embs = blip_model.extract_features(
                    {"image": reference_image, "text_input": qformer_question},
                    mode="multimodal",
                )["multimodal_embeds"]
                rerank_step = 500  # to calculate the embedding in iteration
                for section_spilit in range(0, len(qformer_articles), rerank_step):
                    article_embs = blip_model.extract_features(
                        {
                            "text_input": qformer_articles[
                                section_spilit : section_spilit + rerank_step
                            ]
                        },
                        mode="text",
                    )["text_embeds_proj"][:, 0, :]
                    if section_spilit == 0:
                        article_embs_all = article_embs
                    else:
                        article_embs_all = torch.cat(
                            (article_embs_all, article_embs), dim=0
                        )
                print("article_embs_all shape: ", article_embs_all.shape)
                scores = torch.matmul(
                    article_embs_all.unsqueeze(1).unsqueeze(1),
                    fusion_embs.permute(0, 2, 1),
                ).squeeze()
                scores, _ = scores.max(-1)

                section_similarities = [
                    retrieval_simlarities[section_to_entry[i]]
                    for i in range(len(sections))
                ]
                alpha_1 = 0.5
                alpha_2 = 1 - alpha_1
                scores = (
                    alpha_1 * torch.tensor(section_similarities).to("cuda")
                    + alpha_2 * scores
                )
                # rank by scores high to low
                scores, reranked_index = torch.sort(scores, descending=True)
            top_k_wiki = remove_list_duplicates(
                [entries[section_to_entry[i]].url for i in reranked_index]
            )
            reranked_entries = remove_list_duplicates(
                [entries[section_to_entry[i]] for i in reranked_index]
            )
            reranked_sections = remove_list_duplicates(
                [sections[i] for i in reranked_index]
            )
            if kwargs["save_result"]:
                retrieval_result[data_id]["reranked_entries"] = [
                    entry.url for entry in reranked_entries[:20]
                ]
                retrieval_result[data_id]["reranked_sections"] = reranked_sections[:10]

        if metric == "answer matching":
            entry_sections = reranked_sections
            found = False
            for i, entry in enumerate(entry_sections):
                for answer in target_answer:
                    if answer.strip().lower() in entry_sections[i].strip().lower():
                        found = True
                        break
                if found:
                    break
            if found:
                for k in top_ks:
                    if i < k:
                        reranked_recalls[k] += 1

        else:
            recall = eval_recall(top_k_wiki, ground_truth, top_ks)
            for k in top_ks:
                reranked_recalls[k] += recall[k]

        for k in top_ks:
            print("Reranked Avg Recall@{}: ".format(k), reranked_recalls[k] / (it + 1))
        if kwargs["perform_text_rerank"]:
            if ground_truth in top_k_wiki[:5]:
                gt_index = top_k_wiki.index(ground_truth)
                index, hit = text_reranker.rerank_entry_sections(
                    example["question"], reranked_sections, top_k=5, gt_index=gt_index
                )
                temp = reranked_sections[0]
                reranked_sections[0] = reranked_sections[index]
                reranked_sections[index] = temp
            else:
                gt_index = -1
                hit = 0
            hits += hit
            print("Text Reranking Recalls", hits / (it + 1))

        if kwargs["perform_vqa"]:
            answer = question_generator.llm_answering(
                question=example["question"], entry_section=reranked_sections[0]
            )

            print("answer: ", answer)
            print("target answer: ", target_answer)
            score = evaluate_example(
                example["question"],
                reference_list=target_answer,
                candidate=answer,
                question_type=example["question_type"],
            )

            eval_score += score
            print("score: ", score, "iter: ", it + 1)
            print("eval score: ", eval_score / (it + 1))
    if kwargs["save_result"]:
        with open(kwargs["save_result_path"], "w") as f:
            json.dump(retrieval_result, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--knowledge_base", type=str, required=True)
    parser.add_argument("--faiss_index", type=str, required=True)
    parser.add_argument(
        "--top_ks",
        type=str,
        default="1,5,10,20,100",
        help="comma separated list of top k values, e.g. 1,5,10,20,100",
    )
    parser.add_argument("--perform_vqa", action="store_true")
    parser.add_argument("--perform_text_rerank", action="store_true")
    parser.add_argument("--perform_qformer_reranker", action="store_true")
    parser.add_argument("--qformer_ckpt_path", type=str, default=None)
    parser.add_argument("--retrieval_top_k", type=int, default=20)
    parser.add_argument("--save_result", action="store_true")
    parser.add_argument("--save_result_path", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument(
        "--retriever_vit", type=str, default="clip", help="clip or eva-clip"
    )
    args = parser.parse_args()

    test_config = {
        "test_file_path": args.test_file,
        "knowledge_base_path": args.knowledge_base,
        "faiss_index_path": args.faiss_index,
        "top_ks": [int(k) for k in args.top_ks.split(",")],
        "retrieval_top_k": args.retrieval_top_k,
        "perform_vqa": args.perform_vqa,
        "perform_text_rerank": args.perform_text_rerank,
        "perform_qformer_reranker": args.perform_qformer_reranker,
        "qformer_ckpt_path": args.qformer_ckpt_path,
        "save_result": args.save_result,
        "save_result_path": args.save_result_path,
        "resume_from": args.resume_from,
        "retriever_vit": args.retriever_vit,
    }
    print("test_config: ", test_config)
    run_test(**test_config)
