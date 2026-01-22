from evaluate_bert import get_bert_score, read_input_files
import os
import json
import sys


def evaluate_folder(base_dir,  systems=("aya", "gemma", "zeromt", "opusmt"), 
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
                
                ref_file = os.path.join(test_path, f"ref_bad.{lang}") #comute bad referece
                refs = read_input_files(ref_file)

                cands = read_input_files(cand_file)
                if doc:
                    ctx_file = os.path.join(test_path, f"cap.{lang}")
                    ctx = read_input_files(ctx_file) if os.path.exists(ctx_file) else None
                else:
                    ctx = None
                

                seg_scores = get_bert_score(
                    cands, refs, ctx, lang=lang, doc=doc
                )

                results[testset][lang][sys_name] = {
                    "segment_scores": seg_scores.tolist(),   # JSON can’t handle numpy arrays
                    "system_score": float(seg_scores.mean()) # system-level mean
                }

    # save everything into one JSON
    out_file = os.path.join(base_dir, f"bert_comute_bad_doc_{str(doc)}_scores.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    
    #export PYTHONPATH=$PYTHONPATH:/home/sami/mmt-eval/doc-mte/doc-bert/doc-mt-metrics/bert_score/
    #environment source doc-bert-env/bin/activate
    base_directory = "/home/sami/mmt-eval/doc-mte/MDPI Experiments/data/comute"  # adjust as needed
    evaluate_folder(base_directory, doc=True) #bad referece files
    
    evaluate_folder(base_directory, doc=False) #bad referece files
    