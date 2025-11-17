
import json

from scipy.stats import pearsonr, spearmanr
from matplotlib import pyplot as plt
from scipy.stats import kendalltau
import numpy as np
import pandas as pd
import seaborn as sns

def draw_scatter_plot(human_scores, bert_scores):
    # Calculate correlations
    pearson_corr, _ = pearsonr(human_scores, bert_scores)
    spearman_corr, _ = spearmanr(human_scores, bert_scores)

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(human_scores, bert_scores, color='blue', alpha=0.7)

    # Line of best fit
    m, b = np.polyfit(human_scores, bert_scores, 1)
    plt.plot(human_scores, m*np.array(human_scores)+b, color='red', linestyle='--', label='Best fit line')

    # Add correlation text
    plt.text(min(human_scores), max(bert_scores), 
            f'Pearson: {pearson_corr:.2f}\nSpearman: {spearman_corr:.2f}', 
            fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    # Titles and labels
    plt.title('Human Scores vs. BERT Scores')
    plt.xlabel('Human Scores')
    plt.ylabel('BERT Scores')
    plt.grid(True)
    plt.legend()
    plt.show()

def draw_histogram(human_scores, bert_scores):
    
    plt.figure(figsize=(10, 6))

    plt.hist(human_scores, bins=5, alpha=0.5, label='Human Scores', color='blue', edgecolor='black')
    plt.hist(bert_scores, bins=5, alpha=0.5, label='BERT Scores', color='green', edgecolor='black')

    # Titles and labels
    plt.title('Score Distributions')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.show()


def compute_correlations_score(human_scores, bert_scores):
    
    list_human_scores = []
    list_bert_scores = []
    for key in human_scores.keys():
        if key in bert_scores:
            list_human_scores.append(human_scores[key])
            list_bert_scores.append(bert_scores[key])
        else:
            print(f"Key {key} not found in BERT scores.")
    
    
    
    human_scores_normalized = [(x - 1) / 4 for x in list_human_scores]

    pearson_corr, _ = pearsonr(human_scores_normalized, list_bert_scores)
    spearman_corr, _ = spearmanr(human_scores_normalized, list_bert_scores)
    print(f"Pearson correlation: {pearson_corr}")
    print(f"Spearman correlation: {spearman_corr}")

    #calculate kendall correlation
    kendall_corr, _ = kendalltau(human_scores_normalized, list_bert_scores)
    print(f"Kendall correlation: {kendall_corr}")
    
    return pearson_corr, spearman_corr, kendall_corr

    draw_scatter_plot(human_scores_normalized, list_bert_scores)
    
def compute_correlations_bleu_score(human_scores, bleu_scores):
    
    list_human_scores = []
    list_bleu_scores = []
    for key in human_scores.keys():
        if key in bleu_scores:
            list_human_scores.append(human_scores[key])
            list_bleu_scores.append(bleu_scores[key])
        else:
            print(f"Key {key} not found in BLEU scores.")



    #calculate pearson correlation
    #normalize the human scores as bertscore is between 0 and 1
    human_scores_normalized = [(x - 1) / 4 for x in list_human_scores]
    bleu_scores_normalized = [x / 100 for x in list_bleu_scores]

    #print 10 examples of human and bleu scores
    #for i in range(10):
    #    print(f"Human score: {human_scores_normalized[i]}, BLEU score: {bleu_scores_normalized[i]}")

    pearson_corr, _ = pearsonr(human_scores_normalized, bleu_scores_normalized)
    spearman_corr, _ = spearmanr(human_scores_normalized, bleu_scores_normalized)
    print(f"Pearson correlation: {pearson_corr}")
    print(f"Spearman correlation: {spearman_corr}")

    #calculate kendall correlation
    kendall_corr, _ = kendalltau(human_scores_normalized, bleu_scores_normalized)
    print(f"Kendall correlation: {kendall_corr}")
    
    return pearson_corr, spearman_corr, kendall_corr

    draw_scatter_plot(human_scores_normalized, bleu_scores_normalized)

def draw_corr_heatmap(scores_dict, base_dir):
    data = {
    "Metric": ["COMET doc-level", "COMET", "BERT doc-level", "BERT", "BLEU"],
    "Pearson": [scores_dict["comet_doc"][0], scores_dict["comet"][0], scores_dict["bert_doc"][0], scores_dict["bert"][0], scores_dict["bleu"][0]],
    "Spearman": [scores_dict["comet_doc"][1], scores_dict["comet"][1], scores_dict["bert_doc"][1], scores_dict["bert"][1], scores_dict["bleu"][1]],
    "Kendall": [scores_dict["comet_doc"][2], scores_dict["comet"][2], scores_dict["bert_doc"][2], scores_dict["bert"][2], scores_dict["bleu"][2]]
}

    # Create DataFrame
    df = pd.DataFrame(data)
    df.set_index("Metric", inplace=True)

    # Plot heatmap
    plt.figure(figsize=(10, 4))
    sns.heatmap(df, annot=True, cmap="Blues", fmt=".3f", cbar_kws={'label': 'Correlation'})
    plt.title("Correlation Scores of Metrics", fontsize=14, pad=12)
    plt.yticks(rotation=0)
    plt.show()
    #plt.savefig(f"{base_dir}/correlation_heatmap.pdf", bbox_inches='tight')

if __name__ == "__main__":
    
    base_dir = "/home/sami/mmt-eval/doc-mte/MDPI Experiments/data/thumb/"
    json_human = f"{base_dir}/mscoco_THumB-1.0.jsonl"
    
    json_comet = f"{base_dir}/thumb_cometscore_doc_False_scores.json"
    json_comet_doc = f"{base_dir}/thumb_cometscore_doc_True_scores.json"
    
    json_bert = f"{base_dir}/thumb_bertscore_doc_False_scores.json"
    json_bert_doc = f"{base_dir}/thumb_bertscore_doc_True_scores.json"
        
    json_blue_chrf_ter = f"{base_dir}/thumb_bleu_ter_charf_scores.json"


    human_scores = {}
    corr_scores = {}
    with open(json_human, "r") as f:
        for line in f:
            record = json.loads(line)
            score = record["human_score"]
            seg_id = record["seg_id"]
            sys_id = record["SYS"]
            unique_id = f"{seg_id}_{sys_id}"
            human_scores[unique_id] = score

    scores_dict = {}
    bleu_scores = {}
    ter_scores = {}
    charf_scores = {}
    with open(json_blue_chrf_ter, "r") as f:
        record = json.load(f)

        scores_dict["bleu"] = record["bleu"]["segment_scores"]
        #scores_dict["ter"] = record["ter"]["segment_scores"]
        scores_dict["chrf"] = record["chrf"]["segment_scores"]

    with open(json_bert, "r") as f:
        record = json.load(f)
        

        scores_dict["bert"] = record["bertscore"]["segment_scores"]
    with open(json_bert_doc, "r") as f:
        record = json.load(f)

        scores_dict["bert_doc"] = record["bertscore"]["segment_scores"]
    with open(json_comet, "r") as f:
        record = json.load(f)

        scores_dict["comet"] = record["cometscore"]["segment_scores"]

    with open(json_comet_doc, "r") as f:
        record = json.load(f)

        scores_dict["comet_doc"] = record["cometscore"]["segment_scores"]
        
    for key in scores_dict.keys():
        print(f"computing correlations for {key}...")
        p, sp, k = compute_correlations_score(human_scores, scores_dict[key])
        corr_scores[key] = (p, sp, k)

    
    #draw_corr_heatmap(corr_scores, base_dir)
    
    #visualize the distributions of human and bert scores for bert_doc
    list_human_scores = []
    aligned_scores_dict = {}
    dict_keys = sorted(human_scores.keys())
    
    for metric in scores_dict.keys():
        aligned_scores_dict[metric] = []
        
        for seg_id in dict_keys:
            if seg_id in scores_dict[metric]:
                aligned_scores_dict[metric].append(scores_dict[metric][seg_id])

    for key in dict_keys:
        list_human_scores.append(human_scores[key])
    
    human_scores_normalized = [(x - 1) / 4 for x in list_human_scores]
    
    #draw_histogram(human_scores_normalized, aligned_scores_dict["bert_doc"])
   
    df = pd.DataFrame(aligned_scores_dict)
    df['human'] = human_scores_normalized

    df['bleu'] = df['bleu'] / 100
    df['chrf'] = df['chrf'] / 100

    means = df.mean()

    corrs = df.corr(method='spearman')['human'].drop('human')

    metrics = corrs.index.tolist()
    values = corrs.values

    # Close loop
    values = np.concatenate((values, [values[0]]))
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2, linestyle='solid', label="Correlation with Human")
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)  # correlation range
    plt.title("Correlation of Metrics with Human Scores", size=14, y=1.1)
    plt.legend(loc='upper right')
    plt.show()