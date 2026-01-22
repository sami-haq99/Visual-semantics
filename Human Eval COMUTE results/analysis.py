import csv


def process_human_eval_data(csv_file_path):
    # Path to your CSV file
    csv_file = csv_file_path

    # Dictionary to store data system-wise
    data_dict = {}

    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            # Extract relevant columns
            campaign = row[0]  # engdeu0f01, etc.
            system = row[1]  # opusmt, zeromt, aya, etc.
            seg_id = int(row[2])
            src_lang = row[4]
            tgt_lang = row[5]
            human_score = int(row[6])
            image_file = row[7].split('#')[-1]  # get the actual image filename
            if campaign.endswith('f01') or campaign.endswith('f02'):
                use_of_image = False
            else:
                use_of_image = True

            # Initialize the system key if not present
            if system not in data_dict:
                data_dict[system] = []

            # Append the entry
            data_dict[system].append({
                "segment_id": seg_id,
                "human_score": human_score,
                "image_file": image_file,
                "lang_pair": f"{src_lang}-{tgt_lang}",
                "image_used": use_of_image  # Placeholder, adjust as needed
            })

    # Example function to process the data_dict further if needed
    return data_dict



csv_file_path = '/home/sami/mmt-eval/doc-mte/MDPI Experiments/Human Eval COMUTE results/human_eval_appraise.CSV'
human_eval_results = process_human_eval_data(csv_file_path)

print(human_eval_results)

