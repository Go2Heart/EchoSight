import os
GLD_image_path = "/PATH/TO/GLDv2" 
iNat_image_path = "/PATH/TO/inaturalist"
infoseek_train_path = "/PATH/TO/InfoSeek/train"

def get_image(image_id, dataset_name, iNat_id2name=None):
    """_summary_
        get the image file by image_id. image id are indexed by its first 3 letters in the corresponding folder. e.g. image_id = "abcde" will be stored in "a/b/c/abcde.jpg"
    Args:
        image_id : the image id
    """
    if dataset_name == "inaturalist":
        file_name = iNat_id2name[image_id]
        image_path = os.path.join(iNat_image_path, file_name)
    elif dataset_name == "landmarks":
        image_path = os.path.join(GLD_image_path, image_id[0], image_id[1], image_id[2], image_id + ".jpg")
    elif dataset_name == "infoseek":
        if os.path.exists(os.path.join(infoseek_train_path, image_id + ".jpg")):
            image_path = os.path.join(infoseek_train_path, image_id + ".jpg")
        elif os.path.exists(os.path.join(infoseek_train_path, image_id + ".JPEG")):
            image_path = os.path.join(infoseek_train_path, image_id + ".JPEG")
    else:
        raise NotImplementedError("dataset name not supported")
    return image_path

def reconstruct_wiki_article(knowledge_entry):
    title = knowledge_entry["title"]
    article = "# Wiki Article: " + title + "\n"
    for it, section_title in enumerate(knowledge_entry["section_titles"]):
        article += "\n## Section Title: " + section_title + "\n" + knowledge_entry["section_texts"][it]
    
    return article


def reconstruct_wiki_sections(knowledge_entry, section_index=-1):
    title = knowledge_entry["title"]
    negative_sections = []
    for it, section_title in enumerate(knowledge_entry["section_titles"]):
        if it == int(section_index):
            evidence_section = "# Wiki Article: " + title + "\n" + "## Section Title: " + section_title + "\n" + knowledge_entry["section_texts"][it]
        else:
            negative_sections.append("# Wiki Article: " + title + "\n" + "## Section Title: " + section_title + "\n" + knowledge_entry["section_texts"][it])
    if section_index != -1:
        return evidence_section, negative_sections
    return negative_sections