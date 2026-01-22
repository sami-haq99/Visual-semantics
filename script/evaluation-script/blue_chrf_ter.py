from sacrebleu.metrics import BLEU, CHRF, TER
import os
import json
import sys


def read_input_files(file_path: str):
    with open(file_path) as f:
        return [line.strip() for line in f]
    
def generate_score(hyp, refs, metric):
    
    if metric == "bleu":
        method = BLEU(effective_order=True)
    elif metric == "chrf":
        method = CHRF()
    elif metric == "ter":
        method = TER()
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # System-level score
    sys_score = method.corpus_score(hyp, refs).score

    # Segment-level scores
    seg_scores = []
    for i, hyp in enumerate(hyp):
        
        if isinstance(refs, list):
            # If there's only one reference, use it directly
            seg_score = method.sentence_score(hyp, [refs[0][i]]).score
        else:
            seg_score = method.sentence_score(hyp, refs[i]).score
        seg_scores.append(seg_score)
    
    
    return sys_score, seg_scores



def evaluate_folder(base_dir,  systems=("aya", "gemma", "zeromt", "opusmt"), 
                    langs=("de", "fr", "cs"), metric="bleu"):

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
                
                ref_file = os.path.join(test_path, f"ref_bad.{lang}")
                refs = read_input_files(ref_file)
                refs = [refs]  # sacrebleu expects a list of reference lists

                hyp = read_input_files(cand_file)


                sys_score, seg_scores = generate_score(hyp, refs, metric)

                results[testset][lang][sys_name] = {
                    "segment_scores": seg_scores,   # JSON can’t handle numpy arrays
                    "system_score": sys_score # system-level mean
                }

    # save everything into one JSON
    out_file = os.path.join(base_dir, f"{metric}_bad_scores.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    
    #export PYTHONPATH=$PYTHONPATH:/home/sami/mmt-eval/doc-mte/doc-bert/doc-mt-metrics/bert_score/
    base_directory = "/home/sami/mmt-eval/doc-mte/MDPI Experiments/data/comute"  # adjust as needed
    evaluate_folder(base_directory, metric="chrf") #bad
    evaluate_folder(base_directory, metric="bleu") #bad
    evaluate_folder(base_directory, metric="ter") #bad
    