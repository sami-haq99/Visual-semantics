# extract_english.py

def extract_english_sentences(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                english_sentence = parts[2]
                outfile.write(english_sentence + "\n")

if __name__ == "__main__":
    # Change these file names as needed
    input_file = "enfr_test2017mscoco.txt"
    output_file = "enfrtest2017.en"
    extract_english_sentences(input_file, output_file)
