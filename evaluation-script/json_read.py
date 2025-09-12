import json
import matplotlib.pyplot as plt
import numpy as np
 # clean background with grid
import seaborn as sns
sns.set_style("whitegrid")

# Load JSON file
with open("/home/sami/mmt-eval/doc-mte/MDPI Experiments/data/chrf_scores.json", "r", encoding="utf-8") as f:
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



for testset, languages in results.items():
    # Get all systems across all languages
    all_systems = set()
    for lang_systems in languages.values():
        all_systems.update(lang_systems.keys())
    systems = sorted(all_systems)
    langs = list(languages.keys())

    # Prepare data
    scores = []
    for lang in langs:
        lang_scores = [languages[lang].get(sys, {}).get("system_score", np.nan) for sys in systems]
        scores.append(lang_scores)
    scores = np.array(scores)

    x = np.arange(len(systems))
    width = 0.8 / len(langs)
    
    # Color palette
    colors = plt.cm.Set2.colors  # soft pastel colors
    
    fig, ax = plt.subplots(figsize=(12,6))
    
    for i, lang_scores in enumerate(scores):
        bars = ax.bar(
            x + i*width, lang_scores, width,
            label=langs[i],
            color=colors[i % len(colors)],
            edgecolor='black',
            linewidth=0.7
        )
        # Add value labels on top
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(
                    bar.get_x() + bar.get_width()/2, height + 0.5,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9
                )

    ax.set_xlabel("Systems", fontsize=12)
    ax.set_ylabel("BLEU Score", fontsize=12)
    ax.set_title(f"Testset: {testset}", fontsize=14, fontweight='bold')
    ax.set_xticks(x + width*(len(langs)-1)/2)
    ax.set_xticklabels(systems, rotation=45, ha='right', fontsize=10)
    ax.legend(title="Language")
    
    plt.tight_layout()
    #plt.show()
    #save the figures in pdf format
    fig.savefig(f"/home/sami/mmt-eval/doc-mte/MDPI Experiments/data/chrf_scores_{testset}.pdf", format='pdf')