import csv
import os
GLD_image_path = "/PATH/TO/GLDv2" 
iNat_image_path = "/PATH/TO/inaturalist"
infoseek_test_path = "/PATH/TO/InfoSeek/val"

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
        if os.path.exists(os.path.join(infoseek_test_path, image_id + ".jpg")):
            image_path = os.path.join(infoseek_test_path, image_id + ".jpg")
        elif os.path.exists(os.path.join(infoseek_test_path, image_id + ".JPEG")):
            image_path = os.path.join(infoseek_test_path, image_id + ".JPEG")
    else:
        raise NotImplementedError("dataset name not supported")
    return image_path

def load_csv_data(test_file):
    test_list = []
    with open(test_file, "r") as f:
        reader = csv.reader(f)
        test_header = next(reader)
        for row in reader:
            try: 
                if (row[test_header.index("question_type")] == "automatic" or row[test_header.index("question_type")] == "templated" or row[test_header.index("question_type")] == "multi_answer" or row[test_header.index("question_type")] == "infoseek"): 
                    test_list.append(row)
            except:
                # print row and line number
                print(row, reader.line_num)
                raise ValueError("Error in loading csv data")
    return test_list, test_header


def get_test_question(preview_index, test_list, test_header):
    return {test_header[i]: test_list[preview_index][i] for i in range(len(test_header))}

def remove_list_duplicates(test_list):
    # remove duplicates
    seen = set()
    return [x for x in test_list if not (x in seen or seen.add(x))]
    