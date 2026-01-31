import json
import re

INPUT_JSONL = "/home/gaash/Wasif/Tawheed/MOT_grounding_Dataset/train/train.jsonl"
OUTPUT_JSONL = "/home/gaash/Wasif/Tawheed/MOT_grounding_Dataset/train/train_split_1.jsonl"
FRAMES_PER_QUERY = 1


def extract_frame_from_path(path):
    # e.g. .../000007.jpg -> 7
    return int(re.findall(r"(\d+)\.jpg$", path)[0])


with open(INPUT_JSONL, "r") as fin, open(OUTPUT_JSONL, "w") as fout:
    for line in fin:
        item = json.loads(line)

        images = item["images"]
        ref_image = images[0]
        input_images = images[1:]

        # frame numbers from filenames
        frame_numbers = [extract_frame_from_path(p) for p in input_images]

        # parse answer
        answer = json.loads(item["answer"])

        num_splits = (len(input_images) + FRAMES_PER_QUERY - 1) // FRAMES_PER_QUERY

        for i in range(num_splits):
            start = i * FRAMES_PER_QUERY
            end = start + FRAMES_PER_QUERY

            split_imgs = input_images[start:end]
            split_frames = frame_numbers[start:end]

            # new images list: [ref, img1, img2]
            new_images = [ref_image] + split_imgs

            # slice answer
            new_answer = [
                a for a in answer if a["frame"] in split_frames
            ]

            new_item = {
                "id": f"{item['id']}_part{i+1}",
                "images": new_images,
                "prompt": item["prompt"],
                "answer": json.dumps(new_answer)
            }

            fout.write(json.dumps(new_item) + "\n")

print("Done! Split dataset saved to:", OUTPUT_JSONL)
