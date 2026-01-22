
import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def read_json_comute_score_files(input_json_path):
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = {}

    for testset, languages in data.items():
        #results[testset] = {}
        if testset =='comute':
            for lang, systems in languages.items():
                results[lang] = {}
                
                for system, scores in systems.items():
                    seg_scores = scores.get("segment_scores", [])
                    sys_score = scores.get("system_score", None)
                    
                    results[lang][system] = {
                        "system_score": sys_score,
                        "segment_scores": seg_scores
                    }
    return results

def read_json_score_files(input_json_path):
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = {}

    for testset, languages in data.items():
        #results[testset] = {}
        
        for lang, systems in languages.items():
            if lang in testset:
                results[lang] = {}
                for system, scores in systems.items():
                    seg_scores = scores.get("segment_scores", [])
                    sys_score = scores.get("system_score", None)
                    
                    results[lang][system] = {
                        "system_score": sys_score,
                        "segment_scores": seg_scores
                    }
    return results

def normalize_ter_scores(score_dict):
    #normlise based on min max normalisation
    for langs, systems in score_dict["ter"].items():
        for system, scores in systems.items():
            # system score
            if scores["system_score"] is not None:
                scores["system_score"] = 1 - (scores["system_score"] / 100.0)
            # segment scores
            scores["segment_scores"] = [
                1 - (s / 100.0) for s in scores["segment_scores"]
            ]
    return score_dict

def normalize_scores(score_dict):
    # metrics to normalize
    metrics_to_norm = ["bleu", "chrf"] # do not normalize the TER scores
    # normalize
    for metric in metrics_to_norm:
        
        for langs, systems in score_dict[metric].items():
            for system, scores in systems.items():
                # system score
                if scores["system_score"] is not None:
                    scores["system_score"] = scores["system_score"] / 100.0
                # segment scores
                scores["segment_scores"] = [
                    s / 100.0 for s in scores["segment_scores"]
                ]

    return score_dict

def to_segment_df(score_dict):
    records = []
    for metric, langs in score_dict.items():
        if metric != "ter":
            for lang, systems in langs.items():
                for system, vals in systems.items():
                    for seg in vals["segment_scores"]:
                        records.append({
                            "metric": metric,
                            "language": lang,
                            "system": system,
                            "segment_score": seg
                        })

    seg_df = pd.DataFrame(records)
    return seg_df
    
def to_system_df(score_dict):
    records = []
    for metric, langs in score_dict.items():
        for lang, systems in langs.items():
            for system, vals in systems.items():
                records.append({
                    "metric": metric,
                    "language": lang,
                    "system": system,
                    "system_score": vals["system_score"]
                })

    df = pd.DataFrame(records)
    return df

def segment_score_distribution(score_dict):

    seg_df = to_segment_df(score_dict)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=seg_df, x="system", y="segment_score", hue="metric")
    plt.title("Distribution of Segment Scores per System and Metric")
    plt.show()

def bar_chart_system_scores(score_dict, title):
    
    df = to_system_df(score_dict)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="system", y="system_score", hue="metric")
    plt.title(title)
    #plt.show()
    plt.savefig(f"{title}.pdf", bbox_inches='tight')


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




def flatten_scores(score_dict, run_name):
    records = []
    for metric, langs in score_dict.items():
        for lang, systems in langs.items():
            for system, vals in systems.items():
                records.append({
                    "metric": metric,
                    "language": lang,
                    "system": system,
                    "system_score": vals["system_score"],
                    "run": run_name
                })
    return pd.DataFrame(records)


#difference between orignal and bad_Ref scores
def difference_heat_map(df_org, df_rerun):

    # Merge original and rerun scores
    diff_df = (
        df_rerun.merge(df_org, on=["metric", "language", "system"], suffixes=("_rerun", "_org"))
    )
    diff_df["diff"] = diff_df["system_score_rerun"] - diff_df["system_score_org"]

    # Average differences across languages
    avg_diff = diff_df.groupby(["system", "metric"], as_index=False)["diff"].mean()

    #rename systems for better visualization
    system_rename = {
        "gemma": "Gemma-3",
        "zeromt": "ZeroMMT",
        "aya": "Aya-23",
        "opusmt": "Opus-MT"
    }
    #rename metrics for better visualization
    metric_rename = {
        "chrf": "CHRF",
        "bleu": "BLEU",
        "ter": "TER",
        "bert": "BERTScore",
        "bert_doc": "Doc-BERT",
        "comet": "COMET",
        "comet_doc": "Doc-COMET"
    }
    avg_diff["system"] = avg_diff["system"].map(system_rename)
    avg_diff["metric"] = avg_diff["metric"].map(metric_rename)
    # Pivot for heatmap
    pivot = avg_diff.pivot(index="system", columns="metric", values="diff")

    system_order = ["Gemma-3", "ZeroMMT", "Aya-23", "Opus-MT"]
    metric_order = ["CHRF", "BLEU", "TER", "BERTScore", "Doc-BERT", "COMET", "Doc-COMET"]

    # Reindex pivot to follow this order
    pivot = pivot.reindex(index=system_order, columns=metric_order)
    
    plt.figure(figsize=(9, 5))
    sns.heatmap(pivot, annot=True, center=0, cmap="RdBu", fmt=".3f")
    #plt.title("Average Score Differences Across Languages (Original − Bad Reference)")
    #plt.ylabel("Systems")
    #plt.xlabel("Metrics")
    #remove x and y labels
    plt.ylabel("")
    plt.xlabel("")
    #label the heatmap bar on the right side
    cbar = plt.gcf().axes[-1]
    cbar.set_ylabel('Average Score Difference', rotation=90, labelpad=15)
    
    plt.show()
    #plt.savefig("heatmap_avg_diff_org_vs_bad_ref.pdf", bbox_inches='tight')
    
    

def scatter_plot_org_ref(df_org, df_rerun):
    
    diff_df = (
        df_rerun.merge(df_org, on=["metric", "language", "system"], suffixes=("_rerun", "_org"))
    )
    diff_df["diff"] = diff_df["system_score_rerun"] - diff_df["system_score_org"]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=diff_df, x="system_score_org", y="system_score_rerun",
                    hue="metric", style="language", s=100)

    
        # Add regression/trend line across all data
    sns.regplot(
        data=diff_df,
        x="system_score_org",
        y="system_score_rerun",
        scatter=False,        # don’t add extra scatter points
        ci=None,              # no confidence interval
        color="red",          # regression line color
        line_kws={"label":"Trend line"}
    )
    
     #plt.plot([0, 1], [0, 1], "k--", label="y=x (perfect match)")

     #plt.text(0.05, 0.9, "Perfect agreement (y=x)",  transform=plt.gca().transAxes,  fontsize=10, color="black", rotation=30)
    plt.plot([], [], color="red", label="Trend line")
    plt.xlabel("Original Score")
    plt.ylabel("Bad_reference Score")
    plt.legend()
    plt.title("Original vs Bad_reference Scores")
    plt.show()
    #plt.savefig("scatter_plot_org_vs_bad_reference.pdf", bbox_inches='tight')


from scipy.stats import pearsonr

def scatter_plot_org_ref_corr(df_org, df_rerun):
    
    # Merge original and rerun scores
    diff_df = (
        df_rerun.merge(df_org, on=["metric", "language", "system"], suffixes=("_rerun", "_org"))
    )
    diff_df["diff"] = diff_df["system_score_rerun"] - diff_df["system_score_org"]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=diff_df, x="system_score_org", y="system_score_rerun",
                    hue="metric", style="language", s=100)

    # Add regression line across all points
    sns.regplot(
        data=diff_df,
        x="system_score_org",
        y="system_score_rerun",
        scatter=False,
        ci=None,
        color="red",
        line_kws={"label":"Trend line"}
    )

    # Reference y=x line
    plt.plot([0, 1], [0, 1], "k--", label="y=x (perfect match)")
    #plt.text(0.05, 0.9, "Perfect agreement (y=x)", 
          #   transform=plt.gca().transAxes, 
          #   fontsize=10, color="black", rotation=30)

    # ---- Compute Pearson correlation for each metric ----
    metrics = diff_df["metric"].unique()
    corr_texts = []
    for metric in metrics:
        subset = diff_df[diff_df["metric"] == metric]
        r, _ = pearsonr(subset["system_score_org"], subset["system_score_rerun"])
        corr_texts.append(f"{metric}: r={r:.2f}")

    # Place correlation results as text box
    plt.gca().text(1.02, 0.5, "\n".join(corr_texts),
                   transform=plt.gca().transAxes,
                   fontsize=9, va="center")

    plt.xlabel("Original Score")
    plt.ylabel("Bad_reference Score")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("Original vs Bad_reference Scores")

    plt.tight_layout()
    plt.show()
    #plt.savefig("scatter_plot_org_vs_bad_reference.pdf", bbox_inches='tight')



def flatten_segments(score_dict, run_name):
        records = []
        for metric, langs in score_dict.items():
            for lang, systems in langs.items():
                for system, vals in systems.items():
                    for seg in vals["segment_scores"]:
                        records.append({
                            "metric": metric,
                            "language": lang,
                            "system": system,
                            "segment_score": seg,
                            "run": run_name
                        })
        return pd.DataFrame(records)
    
def good_bad_score_distribution(df_org, df_rerun):
        
   

    seg_org = flatten_segments(score_dict_orig, "original")
    seg_rerun = flatten_segments(score_dict_bad_ref, "bad_ref")
    seg_df = pd.concat([seg_org, seg_rerun])

    plt.figure(figsize=(12, 6))
    sns.violinplot(data=seg_df, x="system", y="segment_score", hue="run", split=True)
    plt.title("Distribution of Segment Scores: Original vs Bad Reference")
    #plt.show()
    plt.savefig("segment_score_distribution_bad_ref_vs_org.pdf", bbox_inches='tight')

def plot_radar_for_system(system, avg_df):
    metrics = avg_df["metric"].unique()
    N = len(metrics)
    
    # Angles for radar axes
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)

    for run in ["Original", "Rerun"]:
        run_data = avg_df[(avg_df["system"] == system) & (avg_df["run"] == run)]
        values = run_data["segment_score"].tolist()
        values += values[:1]  # close the loop
        ax.plot(angles, values, marker="o", label=run)
        ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title(f"Radar Plot: {system}")
    ax.legend(loc="upper right")
    plt.show()



def line_comparison_plot(df_org, df_rerun):
    # Merge data
    df_combined = df_org.merge(
        df_rerun, on=["metric", "language", "system"], suffixes=("_org", "_rerun")
    )

    # Melt into long form for easier plotting
    df_long = df_combined.melt(
        id_vars=["metric", "language", "system"],
        value_vars=["system_score_org", "system_score_rerun"],
        var_name="run_type",
        value_name="score"
    )

    # Rename run_type for better readability
    df_long["run_type"] = df_long["run_type"].map({
        "system_score_org": "Original",
        "system_score_rerun": "Bad_reference"
    })

    # Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_long,
        x="system",
        y="score",
        hue="run_type",
        style="metric",
        markers=True,
        dashes=False,
        linewidth=2
    )

    plt.xticks(rotation=45, ha='right')
    plt.title("Comparison of Original vs Bad_reference MT Scores")
    plt.xlabel("System")
    plt.ylabel("Score")
    plt.legend(title="Run Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()



def plot_difference_chart(df_org, df_rerun):
    # Merge data
    df_combined = df_org.merge(
        df_rerun, on=["metric", "language", "system"], suffixes=("_org", "_rerun")
    )

    # Compute mean per metric and language
    df_avg = (
        df_combined.groupby(["metric", "language"])
        [["system_score_org", "system_score_rerun"]]
        .mean()
        .reset_index()
    )
    
    #print(df_avg)
    print(df_avg.to_string())

    # Melt into long form for plotting
    df_long = df_avg.melt(
        id_vars=["metric", "language"],
        value_vars=["system_score_org", "system_score_rerun"],
        var_name="run_type",
        value_name="score"
    )

    df_long["run_type"] = df_long["run_type"].map({
        "system_score_org": "Original",
        "system_score_rerun": "Bad_reference"
    })

    #print df_long
    print(df_long.to_string())
    
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df_long,
        x="metric",
        y="score",
        hue="run_type",
        ci=None
    )
    plt.title("Average Original vs Bad_reference MT Scores (per Metric)")
    plt.xlabel("Metric")
    plt.ylabel("Average Score")
    plt.legend(title="Run Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    
    base_dir = "/home/sami/mmt-eval/doc-mte/MDPI Experiments/data/corpus results/comute_bad_ref"
    #bad ref scores 
    json_comet_bad_ref = f"{base_dir}/comet_comute_bad_doc_False_scores.json"
    json_comet_doc_bad_ref = f"{base_dir}/comet_comute_bad_doc_True_scores.json"

    json_bert_bad_ref = f"{base_dir}/bert_comute_bad_doc_False_scores.json"
    json_bert_doc_bad_ref = f"{base_dir}/bert_comute_bad_doc_True_scores.json"
    
    json_chrf_bad_ref = f"{base_dir}/chrf_comute_bad_scores.json"
    json_bleu_bad_ref = f"{base_dir}/bleu_comute_bad_scores.json"
    json_ter_bad_ref = f"{base_dir}/ter_comute_bad_scores.json"
    
    #origianl ref scores 
    base_dir_orig = "/home/sami/mmt-eval/doc-mte/MDPI Experiments/data/corpus results"
    json_comet = f"{base_dir_orig}/comet_doc_False_scores.json"
    json_comet_doc = f"{base_dir_orig}/comet_doc_True_scores.json"

    json_bert = f"{base_dir_orig}/bert_doc_False_scores.json"
    json_bert_doc = f"{base_dir_orig}/bert_doc_True_scores.json"

    json_chrf = f"{base_dir_orig}/chrf_scores.json"
    json_bleu = f"{base_dir_orig}/bleu_scores.json"
    json_ter = f"{base_dir_orig}/ter_scores.json"

    score_dict_bad_ref = {
        "comet": read_json_score_files(json_comet_bad_ref),
        "comet_doc": read_json_score_files(json_comet_doc_bad_ref),
        "bert": read_json_score_files(json_bert_bad_ref),
        "bert_doc": read_json_score_files(json_bert_doc_bad_ref),
        "chrf": read_json_score_files(json_chrf_bad_ref),
        "bleu": read_json_score_files(json_bleu_bad_ref),
        "ter": read_json_score_files(json_ter_bad_ref),  #commet for other score analysis 
    }
    
    score_dict_orig = {
        "comet": read_json_comute_score_files(json_comet),
        "comet_doc": read_json_comute_score_files(json_comet_doc),
        "bert": read_json_comute_score_files(json_bert),
        "bert_doc": read_json_comute_score_files(json_bert_doc),
        "chrf": read_json_comute_score_files(json_chrf),
        "bleu": read_json_comute_score_files(json_bleu),
        "ter": read_json_comute_score_files(json_ter),#commet for other score analysis 
    }
    

    #normalise the ter, blue and chrf scores to range 0 and 1. 
    score_dict_bad_ref = normalize_scores(score_dict_bad_ref)
    score_dict_orig = normalize_scores(score_dict_orig)
    score_dict_bad_ref = normalize_ter_scores(score_dict_bad_ref)
    score_dict_orig = normalize_ter_scores(score_dict_orig)
    
    #plot the distribution of segment scores for each system and metric

    #segment_score_distribution(score_dict_bad_ref)
    
    #segment_score_distribution(score_dict_orig)

    #bar_chart_system_scores(score_dict_bad_ref, "COMUTE testet (bad reference scores)")
    #bar_chart_system_scores(score_dict_orig, "COMUTE testet system scores)")
    
    df_org = flatten_scores(score_dict_orig, "original")
    df_rerun = flatten_scores(score_dict_bad_ref, "bad_ref")

    #line_comparison_plot(df_org, df_rerun)
    plot_difference_chart(df_org, df_rerun)
    
    
    difference_heat_map(df_org, df_rerun)
    
    #scatter_plot_org_ref_corr(df_org, df_rerun)
    
    #good_bad_score_distribution(score_dict_orig, score_dict_bad_ref)

    