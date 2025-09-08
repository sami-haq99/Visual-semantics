import requests
from PIL import Image
import torch
from zerommt import create_model
import sys


def load_model():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(model_path="/home/shaq/zero-mt/matthieufp/ZeroMMT-3.3B",cache_dir=".",
                        enable_cfg=False).to(device)
    model.eval()
    
    return model
    

tgt_lang = "deu_Latn"

def generate_translation(model, src_file, image_paths, image_dir, src_lang, tgt_lang, output_dir):
    
    src_texts = []
    with open(src_file, "r") as src_file:
        src_texts = [line.strip() for line in src_file if line.strip()]
    
    file_names = []
    with open(image_paths, "r") as file:
        file_names = [line.strip() for line in file if line.strip()]
    
    src_file_name = src_file.split("/")[-1].split(".")[0]
    with open(output_dir + src_file_name + ".{tgt_lang[:2]}", "w") as output_file:
        for file_name, src_text in zip(file_names, src_texts):

            image = Image.open(image_dir+file_name)
            with torch.inference_mode():
                generated = model.generate(imgs=[image],
                                        src_text=[src_text],
                                        src_lang=src_lang,
                                        tgt_lang=tgt_lang,
                                        beam_size=4)
            translation = model.tokenizer.batch_decode(generated, skip_special_tokens=True)
            output_file.write(f"{translation[0]}\n")

if __name__ == "__main__":
    
    if len(sys.argv) != 6:
        print("Usage: python ic-gemma3b.py <image_name_file> <output_file> <language_code> <image_dir> <source_file>")
        sys.exit(1)
    src_lang = "eng_Latn"
    image_name_file = sys.argv[1]
    output_file = sys.argv[2]
    tgt_lang = sys.argv[3]
    image_dir = sys.argv[4]
    source_file = sys.argv[5]

    print(f"Generating translations for language: {tgt_lang}")
    print(f"Image directory: {image_dir}")
    print(f"Source file: {source_file}")
    print(f"Output file: {output_file}/zeromt.{tgt_lang}")

    
    model = load_model()
    generate_translation(model, source_file, image_name_file, image_dir, src_lang, tgt_lang, output_file)