import os

GLD_image_path = "/PATH/TO/GLDv2"
iNat_image_path = "/PATH/TO/inaturalist"
infoseek_train_path = "/PATH/TO/InfoSeek/train"


def get_image(image_id, dataset_name, iNat_id2name=None):
    """Get the image file by image_id.
    Dataset details:
        iNaturalist:
            image ids are mapped by id2name dict to get the path of the image

        GLDv2/landmarks:
            image ids are indexed by its first 3 letters in the corresponding folder. e.g. image_id = "abcde" will be stored in "a/b/c/abcde.jpg"

        infoseek:
            image ids are either with suffix of .jpg or .JPEG

    Args:
        image_id : the image id
        dataset_name : the dataset name
        iNat_id2name : the dict to map image id to image name for iNaturalist dataset
    """
    if dataset_name == "inaturalist":
        file_name = iNat_id2name[image_id]
        image_path = os.path.join(iNat_image_path, file_name)
    elif dataset_name == "landmarks":
        image_path = os.path.join(
            GLD_image_path, image_id[0], image_id[1], image_id[2], image_id + ".jpg"
        )
    elif dataset_name == "infoseek":
        if os.path.exists(os.path.join(infoseek_train_path, image_id + ".jpg")):
            image_path = os.path.join(infoseek_train_path, image_id + ".jpg")
        elif os.path.exists(os.path.join(infoseek_train_path, image_id + ".JPEG")):
            image_path = os.path.join(infoseek_train_path, image_id + ".JPEG")
    else:
        raise NotImplementedError("dataset name not supported")
    return image_path


def reconstruct_wiki_article_dict(knowledge_entry):
    """Reconstruct the wiki article from the knowledge entry dict.

    Args:
        knowledge_entry : the knowledge entry dict
    """
    title = knowledge_entry["title"]
    article = "# Wiki Article: " + title + "\n"
    for it, section_title in enumerate(knowledge_entry["section_titles"]):
        article += (
            "\n## Section Title: "
            + section_title
            + "\n"
            + knowledge_entry["section_texts"][it]
        )

    return article


def reconstruct_wiki_sections_dict(knowledge_entry, section_index=-1):
    """Reconstruct the wiki sections from the knowledge entry dict.
    
    Args:
        knowledge_entry : the knowledge entry dict
    """
    title = knowledge_entry["title"]
    negative_sections = []
    for it, section_title in enumerate(knowledge_entry["section_titles"]):
        if it == int(section_index):
            evidence_section = (
                "# Wiki Article: "
                + title
                + "\n"
                + "## Section Title: "
                + section_title
                + "\n"
                + knowledge_entry["section_texts"][it]
            )
        else:
            negative_sections.append(
                "# Wiki Article: "
                + title
                + "\n"
                + "## Section Title: "
                + section_title
                + "\n"
                + knowledge_entry["section_texts"][it]
            )
    if section_index != -1:
        return evidence_section, negative_sections
    return negative_sections
