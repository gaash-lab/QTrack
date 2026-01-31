import json
import copy
import re

def get_frame_number_from_path(path):
    """
    Extracts frame number from image path like:
    .../000947.jpg  -> 947
    """
    return int(re.findall(r"(\d+)\.jpg$", path)[0])


def split_queries(input_json_path, output_json_path, images_per_query=2):
    with open(input_json_path, "r") as f:
        data = json.load(f)

    # Handle both list and {"data": [...]} formats
    if isinstance(data, dict) and "data" in data:
        queries = data["data"]
        wrap_back = True
    else:
        queries = data
        wrap_back = False

    new_queries = []

    for item in queries:
        input_images = item["input_images"]
        num_images = len(input_images)

        # Extract frame numbers from input image paths
        frame_numbers = [get_frame_number_from_path(p) for p in input_images]

        # Number of splits
        num_splits = (num_images + images_per_query - 1) // images_per_query

        # Load expected output
        expected_output = item["expected_output"]

        # MULTI-OBJECT case (dict as string)
        is_multi_object = expected_output.strip().startswith("{")

        if is_multi_object:
            expected_output = json.loads(expected_output)
        else:
            # SINGLE OBJECT case (list as string)
            expected_output = json.loads(expected_output)

        for i in range(num_splits):
            start = i * images_per_query
            end = start + images_per_query

            split_images = input_images[start:end]
            split_frames = frame_numbers[start:end]

            new_item = copy.deepcopy(item)
            new_item["input_images"] = split_images
            new_item["num_input_images"] = len(split_images)
            new_item["id"] = f"{item['id']}_part{i+1}"

            # --------- SPLIT EXPECTED OUTPUT ---------
            if is_multi_object:
                # expected_output = {object_1: {track_id, trajectory:[...]}, ...}
                new_expected = {}
                for obj_key, obj_data in expected_output.items():
                    new_traj = [
                        t for t in obj_data["trajectory"]
                        if t["frame"] in split_frames
                    ]
                    new_expected[obj_key] = {
                        "track_id": obj_data["track_id"],
                        "trajectory": new_traj
                    }
                new_item["expected_output"] = json.dumps(new_expected)

            else:
                # expected_output = [{frame, bbox}, ...]
                new_expected = [
                    t for t in expected_output
                    if t["frame"] in split_frames
                ]
                new_item["expected_output"] = json.dumps(new_expected)

            new_queries.append(new_item)

    if wrap_back:
        out_data = {"data": new_queries}
    else:
        out_data = new_queries

    with open(output_json_path, "w") as f:
        json.dump(out_data, f, indent=2)

    print(f"Original queries: {len(queries)}")
    print(f"New queries after splitting: {len(new_queries)}")
    print(f"Saved to: {output_json_path}")


# if __name__ == "__main__":
#     input_json = "dataset.json"
#     output_json = "dataset_split.json"
#     split_queries(input_json, output_json, images_per_query=2)


if __name__ == "__main__":
    input_json = "/home/gaash/Wasif/Tawheed/MOT_grounding_Dataset/train/annotations.json"          
    output_json = "/home/gaash/Wasif/Tawheed/MOT_grounding_Dataset/train/annotations_splited_1.json"   
    split_queries(input_json, output_json, images_per_query=1)
