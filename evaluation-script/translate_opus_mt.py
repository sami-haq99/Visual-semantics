from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def generate_translation(src_lang, tgt_lang, texts):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Tokenize a whole list at once
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)

    # Decode all at once
    translations = [tokenizer.decode(t, skip_special_tokens=True) for t in outputs]
    return translations


if __name__ == "__main__":
    # Example usage
    
    input_dir = "/home/sami/mmt-eval/doc-mte/MDPI Experiments/data/comute" 
    src_lang = "en"
    tgt_lang = ["de", "fr", "cs"]

    for lang in tgt_lang:
        source_file = f"{input_dir}/{lang}_src.{src_lang}"
        with open(source_file, "r", encoding="utf-8") as f:
            source_texts = [line.strip() for line in f.readlines()]

        translated_texts = generate_translation(src_lang, lang, source_texts)
        output_file = f"{input_dir}/opusmt.{lang}"
        with open(output_file, "w", encoding="utf-8") as f:
            for line in translated_texts:
                f.write(line + "\n")    
        print(f"Saved translations to {output_file}")