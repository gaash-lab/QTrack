import os
import subprocess

JSON_FOLDER = "/home/gaash/Tawheed/Reasoning/Output_json"
SCRIPT_PATH = "/home/gaash/Tawheed/Reasoning/Evaluate/calculate_MOT_metrics.py"
OUTPUT_ROOT = "/home/gaash/Tawheed/Reasoning/MCP_changed"
DATASET_ROOT = "/home/gaash/Tawheed/Reasoning/MOT_grounding_Dataset/test"
PYTHON_BIN = "python3"


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for filename in os.listdir(JSON_FOLDER):
        if not filename.endswith(".json"):
            continue

        input_json = os.path.join(JSON_FOLDER, filename)

        base_name = os.path.splitext(filename)[0]
        output_dir = os.path.join(OUTPUT_ROOT, base_name)

        logs_dir = os.path.join(output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        result_file = os.path.join(logs_dir, "result.txt")

        print(f"Processing: {input_json}")
        print(f"Output dir: {output_dir}")
        print(f"Result log: {result_file}")

        cmd = [
            PYTHON_BIN,
            SCRIPT_PATH,
            "--input_json", input_json,
            "--out", output_dir,
            "--dataset-root", DATASET_ROOT
        ]

        with open(result_file, "w") as f:
            subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False
            )

    print("All JSON files processed.")


if __name__ == "__main__":
    main()
