# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not- use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification, AutoModelForVision2Seq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", required=True, type=str, help="The path for your saved model checkpoint directory (e.g., .../global_step_62/actor/)")
    parser.add_argument("--hf_upload_path", default=None, type=str, help="The path of the Hugging Face repo to upload to (e.g., YourUsername/YourModelName)")
    args = parser.parse_args()

    print("--- Starting Simplified Model Merge for Single-GPU Checkpoint ---")
    print(f"  चेकपॉइंट डायरेक्टरी (Checkpoint Directory): {args.local_dir}")

    # --- 1. Define and Validate Paths ---
    checkpoint_file = os.path.join(args.local_dir, "model_world_size_1_rank_0.pt")
    hf_path = os.path.join(args.local_dir, "huggingface")

    if not os.path.exists(checkpoint_file):
        print(f"\nERROR: Checkpoint file not found!")
        print(f"Looked for: {checkpoint_file}")
        print("Please ensure your --local_dir path is correct.")
        return

    if not os.path.exists(hf_path):
        print(f"\nERROR: 'huggingface' subdirectory not found!")
        print(f"Looked for: {hf_path}")
        print("This directory should contain the model's config.json and other metadata.")
        return

    print("Found all necessary files and directories.")

    # --- 2. Load the Model Weights ---
    print(f"Loading state dictionary from: {checkpoint_file}")
    state_dict = torch.load(checkpoint_file, map_location="cpu")
    print("State dictionary loaded successfully.")
    
    # --- 3. Load Model Config and Structure ---
    print(f"Loading model configuration from: {hf_path}")
    config = AutoConfig.from_pretrained(hf_path)

    # Determine the correct AutoModel class from the config
    if "ForTokenClassification" in config.architectures[0]:
        auto_model = AutoModelForTokenClassification
    elif "ForCausalLM" in config.architectures[0]:
        auto_model = AutoModelForCausalLM
    elif "ForConditionalGeneration" in config.architectures[0]:
        auto_model = AutoModelForVision2Seq
    else:
        raise NotImplementedError(f"Unknown architecture {config.architectures}")

    print(f"Identified model architecture: {config.architectures[0]}")

    # --- THIS IS THE CORRECTED SECTION ---
    print("Creating empty model structure on CPU...")
    # We removed the 'with torch.device("meta"):' line to avoid the .item() error
    model = auto_model.from_config(config, torch_dtype=torch.bfloat16)
    print("Empty model created.")
    
    # --- 4. Save the Final Merged Model ---
    print(f"Saving final merged model to: {hf_path}")
    # The save_pretrained method correctly handles loading a state_dict into the model
    model.save_pretrained(hf_path, state_dict=state_dict)
    print("\n--- Merge Complete! ---")
    print(f"Your final, usable Hugging Face model is now saved in: {hf_path}")

    # --- 5. (Optional) Upload to Hugging Face Hub ---
    if args.hf_upload_path:
        print(f"\nUploading model to Hugging Face Hub at: {args.hf_upload_path}")
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(repo_id=args.hf_upload_path, private=False, exist_ok=True)
            api.upload_folder(
                folder_path=hf_path,
                repo_id=args.hf_upload_path,
                repo_type="model",
            )
            print("Upload complete!")
        except ImportError:
            print("Could not upload: `huggingface_hub` library is not installed. Please run `pip install huggingface_hub`.")
        except Exception as e:
            print(f"An error occurred during upload: {e}")

if __name__ == "__main__":
    main()