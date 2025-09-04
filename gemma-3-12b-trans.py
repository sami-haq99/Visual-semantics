from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import requests
import torch
import gc
import sys

torch.set_float32_matmul_precision('high')

model_id = "/home/shaq/spinning-storage/shaq/gemma-3-12b-it"

def load_model():
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto"
    ).eval()

    processor = AutoProcessor.from_pretrained(model_id, use_fat=True)

    return model, processor


#dictionary with language and corresponding prompt
language_prompts = {
    "en": "Translate the following sentence into English, using the image as context. Output only the translation, nothing else.",
    "cs": "Translate the following sentence into Czech, using the image as context. Output only the translation, nothing else.."
}


# Directory containing your images
#read below from command line args

def generate_translation(language, image_dir, source_file, output_file, image_name_file):
    
    file_names = []
    with open(image_name_file, "r") as file:
        file_names = [line.strip() for line in file if line.strip()]
        
    source_text = []
    with open(source_file, "r") as file:
        source_text = [line.strip() for line in file if line.strip()]

    # Supported image extensions
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

    image_paths = [image_dir + file_name for file_name in file_names]
    
    model, processor = load_model()
    
    with open(output_file + f"/gemma-3.{language}", "w", encoding="utf-8") as f:
        for filename, text in zip(image_paths, source_text):
            if filename.lower().endswith(image_extensions):
                try:

                    image = Image.open(filename).convert("RGB")
                    max_size = (512, 512)

                    # Only downscale if needed
                    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                        image.thumbnail(max_size)

                    
                    messages = [
                                {
                                    "role": "system",
                                    "content": [{"type": "text", "text": "You are a helpful assistant."}]
                                },
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "image", "image": image},  # if using PIL.Image object
                                        {"type": "text", "text": language_prompts[language]},
                                        {"type": "text", "text": text}
                                    ]
                                }
                            ]

                    
                    inputs = processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True,
                        return_dict=True, return_tensors="pt"
                    ).to(model.device, dtype=torch.bfloat16)

                    input_len = inputs["input_ids"].shape[-1]

                    with torch.inference_mode():
                        generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
                        generation = generation[0][input_len:]

                    decoded = processor.decode(generation, skip_special_tokens=True)

                    f.write(f"{decoded.strip()}\n")
                    
                    del inputs, generation
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                except Exception as e:
                    print(f"Failed to process {filename}: {e}")

if __name__ == "__main__":

    if len(sys.argv) != 6:
        print("Usage: python ic-gemma3b.py <image_name_file> <output_file> <language_code> <image_dir> <source_file>")
        sys.exit(1)
    image_name_file = sys.argv[1]
    output_file = sys.argv[2]
    language = sys.argv[3]
    image_dir = sys.argv[4]
    source_file = sys.argv[5]

    if language not in language_prompts:
        print(f"Unsupported language code: {language}")
        sys.exit(1)

    generate_translation(language, image_dir, source_file, output_file, image_name_file)


