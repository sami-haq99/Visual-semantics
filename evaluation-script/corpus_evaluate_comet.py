from comet import download_model, load_from_checkpoint
import os
import json
import sys

def add_context_to_text(text, context, sep=" </s> "):
    return f"{context}{sep}{text}"

def read_input_files(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]

def load_model(model_name = "Unbabel/wmt22-comet-da"):
    # Load COMET model (QE model can skip ref field if desired)
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)

    return model


def calculate_comet_score(src_text, hyp_text, ref_text, ctx_text, model, context=False):
    
    if context:
        #src_text = [add_context_to_text(s, c) for s, c in zip([], ctx_text)]
        hyp_text = [add_context_to_text(h, c) for h, c in zip(hyp_text, ctx_text)]
        ref_text = [add_context_to_text(r, c) for r, c in zip(ref_text, ctx_text)]

    data = []
    for src, hyp, ref in zip(src_text, hyp_text, ref_text):
        example = {
            "src": f"{src}",
            "mt":  f"{hyp}",
            "ref": f"{ref}"
        }
        data.append(example)
    

    if context:
        model.enable_context()
    # Score with context
    output = model.predict(
        data, batch_size=8, gpus=1, progress_bar=True)
    return output


def evaluate_folder(base_dir, model,  systems=("aya", "gemma", "zeromt", "opusmt"), 
                    langs=("de", "fr", "cs"), doc=True):

    results = {}  # nested dict to hold everything

    for testset in os.listdir(base_dir):
        test_path = os.path.join(base_dir, testset)
        if not os.path.isdir(test_path):
            continue

        print(f"Processing {testset}...")
        results[testset] = {}

        # Reference file (always English here)

        for lang in langs:
            results[testset][lang] = {}

            for sys_name in systems:
                cand_file = os.path.join(test_path, f"{sys_name}.{lang}")
                if not os.path.exists(cand_file):
                    continue
                
                ref_file = os.path.join(test_path, f"ref.{lang}")
                refs = read_input_files(ref_file)

                cands = read_input_files(cand_file)
                if doc:
                    ctx_file = os.path.join(test_path, f"cap.{lang}")
                    ctx = read_input_files(ctx_file) if os.path.exists(ctx_file) else None
                else:
                    ctx = None
                if testset == "comute" or testset == "mlt":
                    src_file = os.path.join(test_path, f"{lang}_src.en")
                else:
                    src_file = os.path.join(test_path, f"src.en")
                srcs = read_input_files(src_file)
                output = calculate_comet_score(srcs, cands, refs, ctx, model, context=doc)
                results[testset][lang][sys_name] = {
                    "system_score": output.system_score,
                    "segment_scores": output.scores    
                }
                # elif metric == "bertscore": --- IGNORE ---
    # save everything into one JSON
    out_file = os.path.join(base_dir, f"comet_doc_{str(doc)}_scores.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    
    #export PYTHONPATH=$PYTHONPATH:/home/sami/mmt-eval/doc-mte/doc-bert/doc-mt-metrics/bert_score/
    #environment source /home/sami/mmt-eval/doc-mte/comet-env/bin/activate
    base_directory = "/home/sami/mmt-eval/doc-mte/MDPI Experiments/data/"  # adjust as needed
    model = load_model("Unbabel/wmt22-comet-da")
    evaluate_folder(base_directory, model=model, doc=True)
    
    evaluate_folder(base_directory, model=model, doc=False)
