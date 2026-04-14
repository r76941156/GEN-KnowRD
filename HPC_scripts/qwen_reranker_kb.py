import os
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------- LOAD QWEN3 RERANKER MODEL --------
def load_reranker_model():
    print("🔄 Loading Qwen3-Reranker-8B with multi-GPU support...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-8B", padding_side='left',use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-Reranker-8B",
        device_map="auto",               # Multi-GPU support
        
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2" # Reduce memory, faster
    ).eval()

    
    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    token_false_id = tokenizer.convert_tokens_to_ids("no")

    return model, tokenizer, prefix_tokens, suffix_tokens, token_true_id, token_false_id

# -------- FORMAT INPUT FOR RERANKING --------
def format_instruction(instruction, query, doc):
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

def process_inputs_old(pairs, tokenizer, prefix_tokens, suffix_tokens, max_length=8192):
    inputs = tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)  # ✅ Use model.device (multi-GPU safe)
    return inputs

def process_inputs(pairs, tokenizer, prefix_tokens, suffix_tokens, max_length=8192):
    prefix_ids = torch.tensor(prefix_tokens, dtype=torch.long)
    suffix_ids = torch.tensor(suffix_tokens, dtype=torch.long)

    encoded = tokenizer(
        pairs,
        padding="longest",                 
        truncation=True,
        return_tensors="pt",
        max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )

    
    input_ids = []
    for seq in encoded.input_ids:
        new_seq = torch.cat([prefix_ids, seq, suffix_ids])
        input_ids.append(new_seq[:max_length])  # truncate if needed
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

    inputs = {"input_ids": input_ids.to(model.device)}
    return inputs



@torch.no_grad()
def compute_rerank_scores(inputs, model, token_true_id, token_false_id):
    logits = model(**inputs).logits[:, -1, :]
    true_logits = logits[:, token_true_id]
    false_logits = logits[:, token_false_id]
    combined = torch.stack([false_logits, true_logits], dim=1)
    log_probs = torch.nn.functional.log_softmax(combined, dim=1)
    return log_probs[:, 1].exp().tolist()  # probability for "yes"

# -------- RERANK DISEASES --------
def rerank_by_qwen(case_text, diseases, model, tokenizer, prefix_tokens, suffix_tokens, token_true_id, token_false_id):
    instruction = "Given a patient case description, determine whether the document describes a disease that matches or explains the patient's condition."


    field_name = "probability_all_combined"


    doc_fn = lambda d: (
       f"{d.get('disease', '')}\n"
       f"### Clinical Presentation:\n{d.get('clinical_presentation_section', '')}\n"
       f"### Diagnostic Evaluation:\n{d.get('diagnostic_evaluation_section', '')}\n"
       f"### Subtype Variant:\n{d.get('subtype_variant_section', '')}\n"
       f"### Management Therapy:\n{d.get('management_therapy_section', '')}"
    ).strip()

    pairs = [
        format_instruction(instruction, case_text, doc_fn(d))
        for d in diseases
    ]
    inputs = process_inputs(pairs, tokenizer, prefix_tokens, suffix_tokens)
    scores = compute_rerank_scores(inputs, model, token_true_id, token_false_id)

    for d, s in zip(diseases, scores):
        d[field_name] = round(s, 4)

    return [{"disease": d["disease"], field_name: d[field_name]} for d in diseases]

# -------- PROCESS JSON FILE --------
def process_json_file(json_path, model, tokenizer, prefix_tokens, suffix_tokens, token_true_id, token_false_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    output_txt = os.path.join(output_dir, f"{base_name}_reranked.txt")

  
    if os.path.exists(output_txt) and os.path.getsize(output_txt) > 0:
        print(f"⏭️ Skipping {json_path} (already processed)")
        return

    if "pubmed" not in base_name.lower():
       print(f"⏭️  Skipping {json_path} (pubmed file)")
       return

    with open(json_path, "r", encoding="utf-8") as f:
        patient_entry = json.load(f)

    try:
        patient_id = patient_entry["patient_id"]
        case_text = patient_entry["case_data"]
        diseases = patient_entry["diseases"]

        detailed_scores = rerank_by_qwen(case_text, diseases, model, tokenizer, prefix_tokens, suffix_tokens, token_true_id, token_false_id)

        with open(output_txt, "w", encoding="utf-8") as out_f:
            json.dump({"patient_id": patient_id, "detailed_scores": detailed_scores}, out_f, indent=2)

        print(f"✅ Reranked: {output_txt}")
    except Exception as e:
        print(f"❌ Error in {json_path}: {e}")
        with open(output_txt, "w", encoding="utf-8") as out_f:
            json.dump({"error": str(e)}, out_f)

# -------- MAIN --------
if __name__ == "__main__":
    model_name = "gemini"
    input_root = f"patient_stage1_fusion_json_eval_{model_name}_top20_kb"
    output_root = f"query_{model_name}_KB_output"

    model, tokenizer, prefix_tokens, suffix_tokens, token_true_id, token_false_id = load_reranker_model()
    all_json_files = sorted([f for f in os.listdir(input_root) if f.endswith(".json")])

    for json_file in tqdm(all_json_files, desc="📊 Reranking with Qwen3", unit="file"):
        json_path = os.path.join(input_root, json_file)
        process_json_file(json_path, model, tokenizer, prefix_tokens, suffix_tokens, token_true_id, token_false_id, output_root)

