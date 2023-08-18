import os
import json

def merge_json_files(directory):
    merged_data = []

    # Loop through each subdirectory and file in the main directory
    for subdir, _, files in os.walk(directory): # Check if the subdirectory starts with 'abc'
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(subdir, file), 'r') as f:
                    data = json.load(f)
                    merged_data.append(data)

    # Save merged data to a new JSON file
    with open("output/medium_harm/eval_mid_harm.json", "w") as f:
        json.dump(merged_data, f, indent=4)

if __name__ == "__main__":
    main_directory = './harm_results_mid'
    merge_json_files(main_directory)
