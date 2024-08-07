import gradio as gr
import torch
from model import ClipRetriever, reconstruct_wiki_sections, MistralAnswerGenerator

from utils import load_csv_data, get_test_question, get_image, remove_list_duplicates


def run(knowledge_base_path, faiss_index_path, **kwargs):
    retriever = ClipRetriever(device="cuda", model=kwargs["retriever_vit"])
    retriever.load_knowledge_base(knowledge_base_path)
    retriever.load_faiss_index(faiss_index_path)

    answer_generator = MistralAnswerGenerator(
        model_path=kwargs["llm_ckpt"], device="cuda:0", use_embedding_model=False
    )
    if kwargs["use_reranker"]:
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

        # preprocess = squarepad_transform(224)
        preprocess = targetpad_transform(1.25, 224)

    def vqa(image, question):
        top_k = retriever.retrieve_image_faiss(image, top_k=20)
        entries = [retrieved_entry["kb_entry"] for retrieved_entry in top_k]
        entries = remove_list_duplicates(entries)
        seen = set()
        retrieval_simlarities = [
                top_k[i]["similarity"]
                for i in range(20)
                if not (top_k[i]["url"] in seen or seen.add(top_k[i]["url"]))
            ]
        
        # retrieval_simlarities = [top_k[i]["similarity"] for i in range(20)]
        

        if kwargs["use_reranker"]:
            reference_image = preprocess(image).to("cuda").unsqueeze(0)
            sections = []
            section_to_entry = []
            for entry_id, entry in enumerate(entries):
                entry_sections = reconstruct_wiki_sections(entry)
                sections.extend(entry_sections)
                section_to_entry.extend([entry_id] * len(entry_sections))

            qformer_question = question
            qformer_articles = [txt_processors["eval"](article) for article in sections]
            with torch.cuda.amp.autocast():
                fusion_embs = blip_model.extract_features(
                    {"image": reference_image, "text_input": qformer_question},
                    mode="multimodal",
                )["multimodal_embeds"]
                for section_spilit in range(0, len(qformer_articles), 300):
                    article_embs = blip_model.extract_features(
                        {
                            "text_input": qformer_articles[
                                section_spilit : section_spilit + 300
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
            return reranked_sections[0], answer_generator.llm_answering(
                question, entry_section=reranked_sections[0]
            )
        else:
            return entries[0], answer_generator.llm_answering(question, entry=entries[0])

    # Create Gradio interface

    interface = gr.Interface(
        fn=vqa,
        inputs=[
            gr.Image(type="pil", label="Upload Image"),
            gr.Textbox(lines=1, placeholder="Type your question here...", label="Question"),
        ],
        outputs=[
            # markdown text for showing the retrieved section
            gr.Markdown(label="Retrieved Section"),
            gr.Textbox(lines=4, label="Answer"),
        ],
        title="✨EchoSight✨: Knowledge-based Visual Question Answering System",
        description="Upload an image and ask a question about it. ✨EchoSight✨ will provide a retrieved section and an answer based on the retrieval results.",
        examples=[
            ["demo/example1.png", "In addition to expositions, rock concerts, conferences and courses, what else does this building host?"],
            ["demo/example2.png", "Who designed this museum?"],
            ["demo/example3.png", "In which country is this mountain located?"],
            ["demo/example4.png", "What has bulgaria traditionally used this plant for?"],
            ["demo/example5.png", "How do the upper and undersides of the female of this butterfly compare to those of the male?"],
        ],
        theme="default",
        allow_flagging="never",
        live=False
    )

    # Launch the interface
    interface.launch()


if __name__ == "__main__":
    app_config = {
        "use_reranker": True,
        "retriever_vit": "eva-clip",
        "knowledge_base_path": "/PATH/TO/KNOWLEDGE_BASE_FAISS_JSON",
        "faiss_index_path": "/PATH/TO/KNOWLEDGE_BASE_FAISS_INDEX/",
        "qformer_ckpt_path": "/PATH/TO/QFORMER_CHECKPOINT",
        "llm_ckpt":"/PATH/TO/LLM_CHECKPOINT" # Mistral-7B is the default model
    }
    run(**app_config)
