from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import requests
import torch
import gc
import sys

torch.set_float32_matmul_precision('high')

model_id = "/home/support/llm/aya-23-8B"

def load_model():
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    return model, tokenizer


#dictionary with language and corresponding prompt
language_prompts = {
    "en": "Translate the following sentence into English,  output only the translation, nothing else.",
    "cs": "Translate the following sentence into Czech, output only the translation, nothing else.",
    "de": "Translate the following sentence into German, output only the translation, nothing else.",
    "fr": "Translate the following sentence into French, output only the translation, nothing else.",
}


# Directory containing your images
#read below from command line args

def generate_translation(language, source_file, output_file, model, tokenizer):

    source_text = []
    
    try:
        with open(source_file, "r") as file:
            source_text = [line.strip() for line in file if line.strip()]

        source_text = source_text[:100]  # Limit to first 100 lines for testing   
        name_src = source_file.split("/")[-1].split(".")[0]
        with open(output_file+f"/{name_src}_aya-23.{language}", "w") as f:
            for text in source_text:

                messages = [
                    {
                        "role": "user",
                        "content": f"{language_prompts[language]} Sentence: {text}"
                    }
                ]

                input_ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )

                # Generate translation
                gen_tokens = model.generate(
                    input_ids,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.0,
                )

                # Decode and clean the output
                gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
                
                f.write(f"{gen_text.strip()}\n")
                
                del gen_text, gen_tokens
                torch.cuda.empty_cache()
                gc.collect()
                
    except Exception as e:
        print(f"Failed to process {source_file}: {e}")

if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Usage: python ic-gemma3b.py <output_file> <language_code> <source_file>")
        sys.exit(1)
    output_file = sys.argv[1]
    language = sys.argv[2]
    source_file = sys.argv[3]

    if language not in language_prompts:
        print(f"Unsupported language code: {language}")
        sys.exit(1)

    print(f"Generating translations for language: {language}")
    print(f"Source file: {source_file}")
    print(f"Output file: {output_file}/aya-23.{language}")
    model, tokenizer = load_model()
    
    print("Model loaded successfully.")
    
    generate_translation(language, source_file, output_file, model, tokenizer)

    print("Translation generation completed.")
    
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
