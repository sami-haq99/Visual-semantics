from bert_score import score
from add_context import add_context
import sys

def read_input_files(file_path: str):
    with open(file_path) as f:
        return [line.strip() for line in f]

# add contexts to reference and hypothesis texts
#BERT: [SEP]
#RoBERTa: </s>
#XLM-RoBERTa: </s>
# set doc=True to evaluate at the document level
# best layer = 9, correlation score =  0.762062080371671, Rank 4 on WMT 16

def get_bert_score(cands, refs, ctx, lang="de", model_type="bert-base-multilingual-cased", num_layers=9, doc=True, sep_token="[SEP]"):

    if doc == True:
        cands = add_context(orig_txt=cands, context=refs, doc_ids=ctx, sep_token=sep_token)
        refs = add_context(orig_txt=refs, context=refs, doc_ids=ctx, sep_token=sep_token)

    P, R, F1 = score(cands, refs, lang=lang, verbose=True, doc=doc, model_type=model_type, num_layers=num_layers,rescale_with_baseline=True)
    return F1


if __name__ == "__main__":
    #system arguments, ref, cand ctx files paths
    if len(sys.argv) == 6:
        refs = read_input_files(sys.argv[1])
        cands = read_input_files(sys.argv[2])
        ctx = read_input_files(sys.argv[3])
        if (sys.argv[4].lower() == 'true'):
            doc = True
        else:
            doc = False
        lang = sys.argv[5]
        print(f"Evaluating with doc={doc}, lang={lang}")
        
        P, R, F1 = get_bert_score(cands, refs, ctx, lang=lang, model_type="bert-base-multilingual-cased", num_layers=9, doc=doc, sep_token="[SEP]")
        seg_score = F1.cpu().numpy()
        sys_score = seg_score.mean()
        print("System-level BERTScore: {:.4f}".format(sys_score))
        print("Segment-level BERTScore: ", seg_score)
        
    else:
        print("Example: python evaluate-bert.py data/wmt17-de-en-refs.txt data/wmt17-de-en-hyps.txt data/wmt17-de-en-doc-ids.txt")
        exit(1)
