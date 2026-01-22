#from evaluate_bert import get_bert_score
from corpus_evaluate_comet import calculate_comet_score, load_model
from blue_chrf_ter import generate_score
import os
import shutil
import json

def generate_comet(cands, refs, ctx, seg_ids, out_dir, model, doc = False):
    results = {}
    srcs = refs  # Using references as source text for COMET
    output = calculate_comet_score(srcs, cands, refs, ctx, model, context=doc)
    seg_scores = output.scores
    sys_score = output.system_score
    seg_score = {seg_id: score for seg_id, score in zip(seg_ids, seg_scores)}
    results['cometscore'] = {
        "system_score": sys_score,
        "segment_scores": seg_score
    }
    
        # save everything into one JSON

    out_file = os.path.join(out_dir, f"thumb_cometscore_doc_{doc}_scores.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved results to {out_file}")
    
    
def generate_bert(cands, refs, ctx, seg_ids, out_dir, doc = True):

    results = {}  # nested dict to hold everything
    seg_scores = get_bert_score(cands, refs, ctx, lang="en", model_type="roberta-large", num_layers=9, doc=doc, sep_token="</s>")
    
    sys_score = seg_scores.mean()
    seg_score = seg_scores.tolist()
    seg_score = {seg_id: score for seg_id, score in zip(seg_ids, seg_score)}

    results['bertscore'] = {
        "segment_scores": seg_score,   # JSON can’t handle numpy arrays
        "system_score": float(sys_score) # system-level mean
    }

    # save everything into one JSON

    out_file = os.path.join(out_dir, f"thumb_bertscore_doc_{doc}_scores.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved results to {out_file}")


def generate_blue_ter_chrf(cands, refs, out_dir, seg_ids):

    results = {}  # nested dict to hold everything
    sys_score, seg_scores = generate_score(cands, refs, 'bleu')

    seg_scores = {seg_id: score for seg_id, score in zip(seg_ids, seg_scores)}
    
    results['bleu'] = {
        "segment_scores": seg_scores,   # JSON can’t handle numpy arrays
        "system_score": sys_score # system-level mean
    }



    sys_score, seg_scores = generate_score(cands, refs, 'ter')
    seg_scores = {seg_id: score for seg_id, score in zip(seg_ids, seg_scores)}
    results['ter'] = {
        "segment_scores": seg_scores,   # JSON can’t handle numpy arrays
        "system_score": sys_score # system-level mean
    }
    

    sys_score, seg_scores = generate_score(cands, refs, 'chrf')
    seg_scores = {seg_id: score for seg_id, score in zip(seg_ids, seg_scores)}
    results['chrf'] = {
        "segment_scores": seg_scores,   # JSON can’t handle numpy arrays
        "system_score": sys_score # system-level mean
    }


    # save everything into one JSON

    out_file = os.path.join(out_dir, f"thumb_bleu_ter_charf_scores.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved results to {out_file}")








if __name__ == "__main__":
    
    base_dir = "/home/sami/mmt-eval/doc-mte/MDPI Experiments/data/thumb/"
    json_candidates = f"{base_dir}/mscoco_THumB-1.0.jsonl"
    json_references = f"{base_dir}/mscoco_references.json"
    captions_file = f"{base_dir}/cap.en"
    candidates = {}
    image_ids = {}
    captions = {}
    
    #for bert context
    with open(captions_file, "r") as f:
        for line in f:
            name, caption = line.split(':', 1)  # Split at the first colon only
            captions[name.strip()] = caption.strip()
    with open(json_candidates, "r") as f:
        for line in f:
            record = json.loads(line)
            cand = record["hyp"]
            seg_id = record["seg_id"]
            sys_id = record["SYS"]
            image_id = record["image"]
            image_id = image_id.split('_')[-1]  # Extract the numeric part after the underscore
            unique_id = f"{seg_id}_{sys_id}"
            candidates[unique_id] = cand
            image_ids[unique_id] = image_id #to get respective image captions for bert context

    references = {}
    with open(json_references, "r") as f:
        for line in f:
            record = json.loads(line)
            ref = record["refs"]
            seg_id = record["seg_id"]
            #sys_id = record["SYS"]
            #unique_id = f"{seg_id}_{sys_id}"
            references[seg_id] = ref

    # Sort by seg_id to ensure alignment
    sorted_cand_ids = sorted(candidates.keys())
    sorted_cands = [candidates[seg_id] for seg_id in sorted_cand_ids]
    sorted_image_ids = [image_ids[seg_id] for seg_id in sorted_cand_ids]
    
    context_captions = []
    for img_id in sorted_image_ids:
        context_captions.append(captions[img_id]) #image ids start from 1
    #sorted_refs = [references[seg_id] for seg_id in sorted_seg_ids] 
    #example: print(sorted_cands[0], sorted_refs[0])
    #print(sorted_cands[0], sorted_refs[0])
    
    references_list = []
    for i in sorted_cand_ids:
        seg_id = i.split("_")[0]
        references_list.append(references.get(seg_id, ""))

    
    #generate_blue_ter_chrf(sorted_cands, references_list, base_dir, sorted_cand_ids)

    #generate_bert(sorted_cands, references_list, context_captions, sorted_cand_ids, base_dir, doc=True)
    
    model = load_model("Unbabel/wmt22-comet-da")
    generate_comet(sorted_cands, references_list, context_captions, sorted_cand_ids, base_dir, model, doc=True)

