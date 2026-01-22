
import json
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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


def extract_system_scores(score_dict):
    system_scores = {}
    for metric, corpora in score_dict.items():
        system_scores[metric] = {}
        for corpus, langs in corpora.items():
            system_scores[metric][corpus] = {}
            for lang, systems in langs.items():
                system_scores[metric][corpus][lang] = {}
                for system, scores in systems.items():
                    system_scores[metric][corpus][lang][system] = scores["system_score"]
                    # Create a DataFrame for the system scores
                    df = pd.DataFrame({
                        "segment_id": range(len(scores["segment_scores"])),
                        "system_score": scores["segment_scores"]
                    })
                    # Save to CSV
                    #csv_file = f"/home/sami/mmt-eval/doc-mte/MDPI Experiments/data/corpus results/{metric}_{corpus}_{lang}_{system}.csv"
                    #df.to_csv(csv_file, index=False)
                    print("System Score:", f"{metric}_{corpus}_{lang}_{system}", scores["system_score"])
    
    rows = []
    for metric, corpora in system_scores.items():
            for corpus, langs in corpora.items():
                for lang, systems in langs.items():
                    for system, scores in systems.items():
                        rows.append({
                            "metric": metric,
                            "corpus": corpus,
                            "language": lang,
                            "system": system,
                            "score": scores
                        })
    record_df = pd.DataFrame(rows)
    record_df.to_csv(f"/home/sami/mmt-eval/doc-mte/MDPI Experiments/data/corpus results/all_system_scores.csv", index=False)
    
    # calculate average score per langauge for a corpora combining the system scores for that language
    avg_rows = []
    for metric, corpora in system_scores.items():
            for corpus, langs in corpora.items():
                for lang, systems in langs.items():
                    scores = list(systems.values())
                    avg_score = sum(scores) / len(scores) if scores else None
                    avg_rows.append({
                        "metric": metric,
                        "corpus": corpus,
                        "language": lang,
                        "average_score": avg_score
                    })
    avg_df = pd.DataFrame(avg_rows)
    avg_df.to_csv(f"/home/sami/mmt-eval/doc-mte/MDPI Experiments/data/corpus results/average_system_scores_per_language.csv", index=False)
    return system_scores


    
    

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
    
    extract_system_scores(score_dict)
    

    #system_scores_df.to_csv(f"/home/sami/mmt-eval
    #normalise the ter, blue and chrf scores to range 0 and 1.
    #score_dict = normalize_scores(score_dict)
    
    #df = to_dataframe(score_dict)


    csv_file_path = '/home/sami/mmt-eval/doc-mte/MDPI Experiments/Human Eval COMUTE results/human_eval_appraise.CSV'
    human_eval_results = process_human_eval_data(csv_file_path)
    
    #from score_dict extract only 'comute' dataset for all metrics
    comute_scores = {metric: corpora.get("comute", {}) for metric, corpora in score_dict.items()}
    
    #within comute_scores, for each metric extract only 'de' scores 
    comute_scores = {metric: langs.get("de", {}) for metric, langs in comute_scores.items()}
    
    #for each metric and system, get the segment scores where the segment index matches the segment ids in human_eval_results
    matched_scores = {}
    for metric, systems in comute_scores.items():
        matched_scores[metric] = {}
        for system, scores in systems.items():
            segment_scores = scores["segment_scores"]
            if system in human_eval_results:
                matched_segment_scores = []
                for entry in human_eval_results[system]:
                    seg_id = entry["segment_id"]
                    if seg_id < len(segment_scores):
                        matched_segment_scores.append({
                            "segment_id": seg_id,
                            "human_score": entry["human_score"],
                            "model_score": segment_scores[seg_id],
                            "image_file": entry["image_file"],
                            "lang_pair": entry["lang_pair"],
                            "image_used": entry["image_used"]
                        })
                matched_scores[metric][system] = matched_segment_scores

    

    #save matched scores to a csv file, where each row contains: segment_id, human_score, comet_score, bert_score, bert_doc, comet_doc, chrf_score, bleu_score, ter_score, image_file, lang_pair, image_used    
    output_rows = []
    for system in human_eval_results.keys():
        for entry in human_eval_results[system]:
            seg_id = entry["segment_id"]
            row = {
                "system": system,
                "segment_id": seg_id,
                "human_score": entry["human_score"],
                "image_file": entry["image_file"],
                "lang_pair": entry["lang_pair"],
                "image_used": entry["image_used"]
            }
            for metric in matched_scores.keys():
                metric_scores = matched_scores[metric].get(system, [])
                score_entry = next((s for s in metric_scores if s["segment_id"] == seg_id), None)
                if score_entry:
                    row[f"{metric}_score"] = score_entry["model_score"]
                else:
                    row[f"{metric}_score"] = None
            output_rows.append(row)
    df = pd.DataFrame(output_rows)
    #df.to_csv(f"/home/sami/mmt-eval/doc-mte/MDPI Experiments/Human Eval COMUTE results/human_metric_matched_scores.csv", index=False)

    #draw human metric pairplot using seaborn for scores in df, excluding the 'ter' metric
    import numpy as np
    import matplotlib.pyplot as plt
    #rename 'comet' to 'comute' in column names
    #make the plot longer in height instead of width
    df.rename(columns={"human_score": "Human", "comet_score": "COMET", "comet_doc_score": "Doc-COMET", "bert_score": "BERTScore", "bert_doc_score": "Doc-BERT", "chrf_score": "CHRF", "bleu_score": "BLEU"}, inplace=True)
    #show x and y label ticks in smaller font size
    plt.figure(figsize=(10, 12))
    sns.pairplot(df, vars=["Human", "COMET", "Doc-COMET", "BERTScore", "Doc-BERT", "CHRF", "BLEU"], hue="image_used", height=1.4, aspect=1, diag_kind="kde", plot_kws={
            "s": 10,                 # ✅ Smaller points      # ✅ Transparency to show overlap
            #"linewidth": 0.8,
            #"edgecolor": "grey",
            "marker": "o"
        })
    plt.suptitle("Human Evaluation vs Metric Scores (Image Used vs Not Used)", y=1.02)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.show()
    #plt.savefig(f"/home/sami/mmt-eval/doc-mte/MDPI Experiments/Human Eval COMUTE results/human_metric_pairplot.png", dpi=300)
    plt.close()


    # Rename columns
    df.rename(columns={
        "human_score": "Human",
        "comet_score": "COMET",
        "comet_doc_score": "Doc-COMET",
        "bert_score": "BERTScore",
        "bert_doc_score": "Doc-BERT",
        "chrf_score": "CHRF",
        "bleu_score": "BLEU"
    }, inplace=True)

    metrics = ["Human", "COMET", "Doc-COMET", "BERTScore", "Doc-BERT", "CHRF", "BLEU"]

    # ===========================
    # FIXED COLORS + LESS OVERLAP
    # ===========================
    CUSTOM_PALETTE = {
        0: "#1f77b4",   # true matplotlib default BLUE
        1: "#ff7f0e"    # true matplotlib default ORANGE
    }

    sns.set_style("white")

    g = sns.pairplot(
        df,
        vars=metrics,
        hue="image_used",
        palette=CUSTOM_PALETTE,     # ✅ Exact blue & orange
        height=1.4,
        aspect=1,
        diag_kind="kde",
        plot_kws={
            "s": 6,                 # ✅ Smaller points
            "alpha": 0.5,         # ✅ Transparency to show overlap
            "linewidth": 0,
            "edgecolor": None,
            "marker": "o"
        },
        diag_kws={
            "fill": True,
            "alpha": 0.5
        }
    )

    # Tight layout
    g.figure.set_size_inches(10, 13)
    g.figure.subplots_adjust(
        left=0.05,
        right=0.99,
        bottom=0.05,
        top=0.95,
        wspace=0.05,
        hspace=0.05
    )

    # Smaller ticks
    for ax in g.axes.flat:
        if ax:
            ax.tick_params(labelsize=7)

    # Title
    g.figure.suptitle(
        "Human Evaluation vs Metric Scores (Image Used vs Not Used)",
        fontsize=12,
        y=0.995
    )

    plt.show()

    # For LaTeX export:
    # plt.savefig("human_metric_pairplot.pdf", dpi=300, bbox_inches="tight")
    plt.close()





    #Calculate average scores per system for each metric and human score, for image used and not used separately
    avg_scores_used = df[df["image_used"] == True].groupby("system").mean().reset_index()
    avg_scores_not_used = df[df["image_used"] == False].groupby("system").mean().reset_index()
    


    #calculate system wise pearson correlation between human scores and each metric, for image used and not used separately
    from scipy.stats import pearsonr

    for metric in matched_scores.keys():
        # For image used
        used_human_scores = avg_scores_used[f"{metric}_score"].values
        used_model_scores = avg_scores_used["human_score"].values
        if len(used_human_scores) > 0 and len(used_model_scores) > 0:
            corr, _ = pearsonr(used_human_scores, used_model_scores)
            print(f"Pearson correlation for {metric} (image used): {corr}")
            #add this corr score to avg_scores_used dataframe
            avg_scores_used.loc[avg_scores_used.index, f"{metric}_corr"] = corr
            
        # For image not used
        not_used_human_scores = avg_scores_not_used[f"{metric}_score"].values
        not_used_model_scores = avg_scores_not_used["human_score"].values
        if len(not_used_human_scores) > 0 and len(not_used_model_scores) > 0:
            corr, _ = pearsonr(not_used_human_scores, not_used_model_scores)
            print(f"Pearson correlation for {metric} (image not used): {corr}")
            #add this corr score to avg_scores_not_used dataframe
            avg_scores_not_used.loc[avg_scores_not_used.index, f"{metric}_corr"] = corr

    print(avg_scores_used)
    print(avg_scores_not_used)
    
    #avg_scores_used.to_csv(f"/home/sami/mmt-eval/doc-mte/MDPI Experiments/Human Eval COMUTE results/human_metric_avg_scores_used.csv", index=False)
    #avg_scores_not_used.to_csv(f"/home/sami/mmt-eval/doc-mte/MDPI Experiments/Human Eval COMUTE results/human_metric_avg_scores_not_used.csv", index=False)


    #calculate the overall pearson correlation between human scores and each metric, for image used and not used separately
    corr_results = {}
    for metric in matched_scores.keys():
        # For image used
        used_df = df[df["image_used"] == True]
        if not used_df.empty:
            corr, _ = pearsonr(used_df[f"{metric}_score"].values, used_df["human_score"].values)
            print(f"Overall Pearson correlation for {metric} (image used): {corr}")
            corr_results[f"{metric}_used"] = corr
        # For image not used
        not_used_df = df[df["image_used"] == False]
        if not not_used_df.empty:
            corr, _ = pearsonr(not_used_df[f"{metric}_score"].values, not_used_df["human_score"].values)
            print(f"Overall Pearson correlation for {metric} (image not used): {corr}")
            corr_results[f"{metric}_not_used"] = corr
    
    #save the results to csv file for each metric and image used/not used
    corr_df = pd.DataFrame(list(corr_results.items()), columns=["metric_image_used", "pearson_correlation"])
  #  corr_df.to_csv(f"/home/sami/mmt-eval/doc-mte/MDPI Experiments/Human Eval COMUTE results/human_metric_pearson_correlation.csv", index=False)

