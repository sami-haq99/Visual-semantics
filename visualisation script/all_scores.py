
import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def read_json_score_files(input_json_path):
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = {}

    for testset, languages in data.items():
        results[testset] = {}
        
        for lang, systems in languages.items():
            results[testset][lang] = {}
            
            for system, scores in systems.items():
                seg_scores = scores.get("segment_scores", [])
                sys_score = scores.get("system_score", None)
                
                results[testset][lang][system] = {
                    "system_score": sys_score,
                    "segment_scores": seg_scores
                }
    return results

def to_dataframe(score_dict):
    rows = []
    for metric, corpora in score_dict.items():
        for corpus, langs in corpora.items():
            for lang, systems in langs.items():
                for system, scores in systems.items():
                    rows.append({
                        "metric": metric,
                        "corpus": corpus,
                        "language": lang,
                        "system": system,
                        "score": scores["system_score"]
                    })
    return pd.DataFrame(rows)

def to_dataframe_segments(score_dict):
    rows = []
    for metric, corpora in score_dict.items():
        for corpus, langs in corpora.items():
            for lang, systems in langs.items():
                for system, scores in systems.items():
                    rows.append({
                        "metric": metric,
                        "corpus": corpus,
                        "language": lang,
                        "system": system,
                        "score": scores["segment_scores"]
                    })
    return pd.DataFrame(rows)


def normalize_scores(score_dict):
    # metrics to normalize
    metrics_to_norm = ["bleu", "ter", "chrf"]
    
    # find min and max per metric across all corpora/langs/systems
    minmax = {}
    for metric in metrics_to_norm:
        all_scores = []
        for corpus, langs in score_dict[metric].items():
            for lang, systems in langs.items():
                for system, scores in systems.items():
                    if scores["system_score"] is not None:
                        all_scores.append(scores["system_score"])
                    all_scores.extend(scores["segment_scores"])
        if all_scores:
            minmax[metric] = (min(all_scores), max(all_scores))
    
    # normalize
    for metric in metrics_to_norm:
        min_val, max_val = minmax[metric]
        rng = max_val - min_val if max_val > min_val else 1.0
        
        for corpus, langs in score_dict[metric].items():
            for lang, systems in langs.items():
                for system, scores in systems.items():
                    # system score
                    if scores["system_score"] is not None:
                        scores["system_score"] = (scores["system_score"] - min_val) / rng
                    # segment scores
                    scores["segment_scores"] = [
                        (s - min_val) / rng for s in scores["segment_scores"]
                    ]
    return score_dict

if __name__ == "__main__":
    
    base_dir = "/home/sami/mmt-eval/doc-mte/MDPI Experiments/data/corpus results/"
    
    json_comet = f"{base_dir}/comet_doc_False_scores.json"
    json_comet_doc = f"{base_dir}/comet_doc_True_scores.json"

    json_bert = f"{base_dir}/bert_doc_False_scores.json"
    json_bert_doc = f"{base_dir}/bert_doc_True_scores.json"
    
    json_chrf = f"{base_dir}/chrf_scores.json"
    json_bleu = f"{base_dir}/bleu_scores.json"
    json_ter = f"{base_dir}/ter_scores.json"

    score_dict = {
        "comet": read_json_score_files(json_comet),
        "comet_doc": read_json_score_files(json_comet_doc),
        "bert": read_json_score_files(json_bert),
        "bert_doc": read_json_score_files(json_bert_doc),
        "chrf": read_json_score_files(json_chrf),
        "bleu": read_json_score_files(json_bleu),
        "ter": read_json_score_files(json_ter),
    }
    

    #normalise the ter, blue and chrf scores to range 0 and 1. 
    score_dict = normalize_scores(score_dict)
    
    df = to_dataframe(score_dict)


    