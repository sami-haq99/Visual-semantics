import sacrebleu
from sacrebleu.metrics import BLEU, CHRF, TER

def generate_bleu_score(hyp, refs):
    bleu = BLEU(smooth_method="exp")
    # System-level score
    sys_score = bleu.corpus_score(hyp, refs).score
    
    # Segment-level scores
    #seg_scores = [bleu.sentence_score(h, r).score for h, r in zip(hyp, refs)]
    seg_scores = []
    for i, hyp in enumerate(hyp):
        seg_score = bleu.sentence_score(hyp, [refs[0][i]]).score
        seg_scores.append(seg_score)

    return sys_score, seg_scores

# One hypothesis per sentence
sys = [ "this is a long sentence another sentence", "this is a test sentence same as the reference" ]

# One reference set (could be multiple if you want)
refs = [ "this is a long sentence another sentence", "this is a test sentence same as the reference"]
refs = [refs]  # sacrebleu expects a list of reference sets

sys_score, seg_scores = generate_bleu_score(sys, refs)

# Corpus BLEU
#sys_score = sacrebleu.corpus_bleu(hyps, refs).score

# Segment BLEU
#seg_scores = [sacrebleu.sentence_bleu(h, [r]).score for h, r in zip(hyps, refs[0])]

print(f"BLEU System Score: {sys_score}")
print(f"BLEU Segment Scores: {seg_scores}")