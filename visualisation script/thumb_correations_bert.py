
import json

from scipy.stats import pearsonr, spearmanr
from matplotlib import pyplot as plt
from scipy.stats import kendalltau
import numpy as np
import pandas as pd
import seaborn as sns
import itertools
import random

def plot_perm_both_heatmap(results, alpha=0.05):

    # Extract unique metrics
    metrics = sorted({m for mpair in results.keys() for m in mpair})
    M = pd.DataFrame(0, index=metrics, columns=metrics, dtype=int)

    # Fill matrix according to PERM-BOTH paper definition
    for (metricA, metricB), vals in results.items():

        # auto-detect p-value and accuracies
        if isinstance(vals, dict):
            p = vals.get("p", vals.get("p-value", vals.get("p_value")))
            accA = vals.get("accA", None)
            accB = vals.get("accB", None)
        else:
            # assume tuple: (accA, accB, diff, p)
            accA, accB, diff, p = vals

        # WMT21 rule: row metric > col metric AND significant
        if p < alpha and accA > accB:
            M.loc[metricA, metricB] = 1
        if p < alpha and accB > accA:
            M.loc[metricB, metricA] = 1

    # Make diagonal light
    np.fill_diagonal(M.values, 0)

    # Plot
    plt.figure(figsize=(8, 8))
    plt.imshow(M.values, cmap="Greys", interpolation="nearest")
    plt.xticks(np.arange(len(metrics)), metrics, rotation=90)
    plt.yticks(np.arange(len(metrics)), metrics)
    plt.title("PERM-BOTH Directional Significance Heatmap", fontsize=14)
    plt.tight_layout()
    plt.show()

    return M


def create_pairwise_df(human_scores, metric_scores):
    """
    human_scores: dict {system: human_average_score}
    metric_scores: dict of dicts {metric: {system: score}}
    """

    systems = list(human_scores.keys())
    pairs = list(itertools.combinations(systems, 2))
    
    rows = []

    for A, B in pairs:
        row = {
            "systemA": A,
            "systemB": B,
            "human": np.sign(human_scores[A] - human_scores[B])
        }

        for metric_name, metric_dict in metric_scores.items():
            #ignore metric_name 'human'
            if metric_name == 'human':
                continue    
            row[metric_name] = np.sign(metric_dict[A] - metric_dict[B])

        rows.append(row)

    return pd.DataFrame(rows)

def perm_both(df, metricA, metricB, N=10000):
    df = df[df["human"] != 0]   # ignore human ties
    
    human = df["human"].values
    A = df[metricA].values
    B = df[metricB].values
    
    # actual difference
    accA = (human == A).mean()
    accB = (human == B).mean()
    diff_real = accA - accB

    count = 0

    for _ in range(N):
        # random swap A/B for each pair
        swap = np.random.rand(len(A)) < 0.5
        A_perm = np.where(swap, B, A)
        B_perm = np.where(swap, A, B)

        accA_perm = (human == A_perm).mean()
        accB_perm = (human == B_perm).mean()

        if accA_perm - accB_perm >= diff_real:
            count += 1

    p = count / N

    return {
        "accA": accA,
        "accB": accB,
        "difference": diff_real,
        "p_value": p
    }


def system_level_pairwise_accuracy(df, metric, exclude_systems=None):
    # df must contain: system, human_mean, metric_mean columns
    
    systems = df["system"].unique()
    correct = 0
    total = 0

    if exclude_systems is not None:
        print(f"Excluding systems: {exclude_systems}")
        systems = [sys for sys in systems if sys not in exclude_systems]

    
    for sysA, sysB in itertools.combinations(systems, 2):
        hA = df[df.system == sysA]["human_mean"].item()
        hB = df[df.system == sysB]["human_mean"].item()

        # skip ties as WMT does
        if hA == hB:
            continue
        
        mA = df[df.system == sysA][metric].item()
        mB = df[df.system == sysB][metric].item()
        
        human_delta = np.sign(hA - hB)
        metric_delta = np.sign(mA - mB)
        
        if human_delta == metric_delta:
            correct += 1
        
        total += 1
    
    return correct / total


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


def compute_correlations_score(human_scores, scores):
    
    list_human_scores = []
    list_scores = []
    for key in human_scores.keys():
        if key in scores:
            list_human_scores.append(human_scores[key])
            list_scores.append(scores[key])
        else:
            print(f"Key {key} not found in BERT scores.")
    

    pearson_corr, _ = pearsonr(list_human_scores, list_scores)
    spearman_corr, _ = spearmanr(list_human_scores, list_scores)
    print(f"Pearson correlation: {pearson_corr}")
    print(f"Spearman correlation: {spearman_corr}")

    #calculate kendall correlation
    kendall_corr, _ = kendalltau(list_human_scores, list_scores)
    print(f"Kendall correlation: {kendall_corr}")
    
    return pearson_corr, spearman_corr, kendall_corr



def draw_corr_heatmap(scores_dict, base_dir = None):
    data = {
    "Metric": ["Doc-COMET", "COMET", "Doc-BERT", "BERTScore", "CHRF", "BLEU"],
    "Pearson": [scores_dict["comet_doc"][0], scores_dict["comet"][0], scores_dict["bert_doc"][0], scores_dict["bert"][0], scores_dict["chrf"][0], scores_dict["bleu"][0]],
    "Spearman": [scores_dict["comet_doc"][1], scores_dict["comet"][1], scores_dict["bert_doc"][1], scores_dict["bert"][1], scores_dict["chrf"][1], scores_dict["bleu"][1]],
    "Kendall": [scores_dict["comet_doc"][2], scores_dict["comet"][2], scores_dict["bert_doc"][2], scores_dict["bert"][2], scores_dict["chrf"][2], scores_dict["bleu"][2]]
}

    # Create DataFrame
    df = pd.DataFrame(data)
    df.set_index("Metric", inplace=True)

    # Plot heatmap
    plt.figure(figsize=(10, 4))
    sns.heatmap(df, annot=True, cmap="Blues", fmt=".3f", cbar_kws={'label': 'Correlation'})
    #plt.title("Correlation Scores of Metrics", fontsize=14, pad=12)
    plt.yticks(rotation=0)
    plt.show()
    #plt.savefig(f"{base_dir}/correlation_heatmap.pdf", bbox_inches='tight')



def plot_significance_heatmap(sig_df, title="Metric Significance (PERM-BOTH)"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(sig_df, annot=True, cmap="viridis_r", linewidths=.5,
                cbar_kws={'label': 'p-value'})
    plt.title(title)
    plt.show()


def compute_system_means(df):
    """
    df index = metrics
    df columns = systems
    df cell = dict {seg_id: score}
    """
    means = {}

    for metric in df.index:
        means[metric] = {}
        for system in df.columns:
            seg_scores = list(df.loc[metric, system].values())
            means[metric][system] = np.mean(seg_scores)

    return means

def compute_significance_matrix(df, R=1000):
    metrics = [m for m in df.index if m != "human"]
    sig = pd.DataFrame(index=metrics, columns=metrics, dtype=float)

    for m1 in metrics:
        for m2 in metrics:
            if m1 == m2:
                sig.loc[m1, m2] = 0.0
            else:
                sig.loc[m1, m2] = perm_both_test(df, m1, m2, R=R)

    return sig

def sign(x):
    if x > 0: return 1
    if x < 0: return -1
    return 0

def pairwise_accuracy(metric_values, human_values):
    """
    metric_values: dict {system → mean score}
    human_values:  dict {system → human mean score}
    """
    systems = list(metric_values.keys())
    correct = 0
    total = 0

    for s1, s2 in itertools.combinations(systems, 2):
        m_delta = metric_values[s1] - metric_values[s2]
        h_delta = human_values[s1] - human_values[s2]

        if sign(m_delta) == sign(h_delta):
            correct += 1
        total += 1

    return correct / total

def perm_both_test(df, metric_a, metric_b, human_metric="human", R=1000):
    """
    Returns p-value comparing metric_a vs metric_b
    using PERM-BOTH (Deutsch et al. 2021)
    """
    system_means = compute_system_means(df)

    human_val = system_means[human_metric]

    true_acc_a = pairwise_accuracy(system_means[metric_a], human_val)
    true_acc_b = pairwise_accuracy(system_means[metric_b], human_val)
    
    true_diff = true_acc_a - true_acc_b

    # Prepare arrays of segment-level scores per system
    orig_A = {sys: list(df.loc[metric_a, sys].values()) for sys in df.columns}
    orig_B = {sys: list(df.loc[metric_b, sys].values()) for sys in df.columns}

    count = 0

    for _ in range(R):
        # Shuffle segment scores independently for both metrics
        perm_A = {sys: random.sample(orig_A[sys], len(orig_A[sys])) for sys in df.columns}
        perm_B = {sys: random.sample(orig_B[sys], len(orig_B[sys])) for sys in df.columns}

        # Convert back to system-level means
        perm_means_A = {sys: np.mean(scores) for sys, scores in perm_A.items()}
        perm_means_B = {sys: np.mean(scores) for sys, scores in perm_B.items()}

        # Compute permuted accuracy values
        perm_acc_A = pairwise_accuracy(perm_means_A, human_val)
        perm_acc_B = pairwise_accuracy(perm_means_B, human_val)
        perm_diff = perm_acc_A - perm_acc_B

        # Count permutations where difference ≥ observed difference
        if abs(perm_diff) >= abs(true_diff):
            count += 1

    p_value = count / R
    return p_value

if __name__ == "__main__":
    
    base_dir = "/home/sami/mmt-eval/doc-mte/MDPI Experiments/MDPI code-base/corpus results/thumb/"
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

    
    draw_corr_heatmap(corr_scores) #just to show not save
    
    system_level_scores = {}
    system_level_scores["human"] = {}
    for key in human_scores.keys():
        sys_id = key.split("_")[1]
        seg_id = key.split("_")[0]
        #sys_id wise human scores
        if sys_id not in system_level_scores["human"]:
            system_level_scores["human"][sys_id] = {}
        system_level_scores["human"][sys_id][seg_id] = human_scores[key]

    for metric_key in scores_dict.keys():
        system_level_scores[metric_key] = {}
        for key in scores_dict[metric_key].keys():
            sys_id = key.split("_")[1]
            seg_id = key.split("_")[0]
            if sys_id not in system_level_scores[metric_key]:
                system_level_scores[metric_key][sys_id] = {}
            system_level_scores[metric_key][sys_id][seg_id] = scores_dict[metric_key][key]
    
    df = pd.DataFrame.from_dict(system_level_scores, orient="index")
    
    #find cases where human scores are high and comet_doc is high as well, print scores for other metrics for thoses systems
    high_human_threshold = 4.5
    high_comet_doc_threshold = 0.85

    for sys_id in system_level_scores["human"].keys():
        if system_level_scores["human"][sys_id]["mean"] > high_human_threshold and system_level_scores["comet_doc"][sys_id]["mean"] > high_comet_doc_threshold:
            print(f"System: {sys_id}")
            for metric in system_level_scores.keys():
                if metric != "human":
                    print(f"  {metric}: {system_level_scores[metric][sys_id]['mean']}")


    df_rows = {}
    #normalise the segment scores to min-max 0-1 range
    for metric_key in system_level_scores.keys():
        for sys_id in system_level_scores[metric_key].keys():
            seg_scores = list(system_level_scores[metric_key][sys_id].values())
            min_score = min(seg_scores) if seg_scores else 0
            max_score = max(seg_scores) if seg_scores else 1
            for seg_id in system_level_scores[metric_key][sys_id].keys():
                orig_score = system_level_scores[metric_key][sys_id][seg_id]
                norm_score = (orig_score - min_score) / (max_score - min_score) if max_score > min_score else 0.0
                system_level_scores[metric_key][sys_id][seg_id] = norm_score


    
    