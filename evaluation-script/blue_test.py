from sacrebleu.metrics import BLEU, CHRF, TER
import os
import json
import sys
import sacrebleu


def read_input_files(file_path: str):
    with open(file_path) as f:
        return [line.strip() for line in f]
    
def generate_bleu_score(hyp, refs):
    bleu = BLEU()
    # System-level score
    sys_score = bleu.corpus_score(hyp, refs).score
    
    # Segment-level scores
    seg_scores = [bleu.sentence_score(h, refs).score for h in hyp]
    
    return sys_score, seg_scores

if __name__ == "__main__":
    
    base_directory = "/home/sami/mmt-eval/doc-mte/MDPI Experiments/data/"  # adjust as needed
    hyp = read_input_files("/home/sami/mmt-eval/doc-mte/MDPI Experiments/data/comute/gemma.de")
    refs = read_input_files("/home/sami/mmt-eval/doc-mte/MDPI Experiments/data/comute/ref.de")
    
    sys_score, seg_score = generate_bleu_score(hyp, refs)
   
    
    print(f"BLEU Segment Scores: {seg_score}")
    print(f"BLEU System Score: {sys_score}")
    print(len(hyp), len(refs))
    print("mean:", sum(seg_score)/len(seg_score))
    
   