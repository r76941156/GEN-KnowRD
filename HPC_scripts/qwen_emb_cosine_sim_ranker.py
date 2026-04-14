import os
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

# === 🔧 Config ===
model = "claude"
checkpoint_dir = "/data/wei_lab/sander/output_qwen_emb8b_kb_claude/v0-20251020-161045/checkpoint-30"

input_root = f"patient_stage1_dataset_json"
output_root = f"dataset_json_output_{model}"
os.makedirs(output_root, exist_ok=True)

task_instruction = (
    "Given a patient case description, retrieve relevant rare diseases that explain the clinical findings"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load tokenizer and model ===
print("🔄 Loading Qwen3-Embedding-8B model ...")
model_name = "Qwen/Qwen3-Embedding-8B"  # HuggingFace model repo

#Load original model
#tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
#model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

#Load from local checkpoint (if available)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, padding_side="left")
model = AutoModel.from_pretrained(checkpoint_dir, torch_dtype=torch.float16).to(device)
model.eval()


# === Pooling strategy: last token pooling ===
def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        return last_hidden_states[torch.arange(last_hidden_states.size(0), device=last_hidden_states.device), sequence_lengths]


# === Embedding function ===
def encode_texts(texts, is_query: bool) -> Tensor:
    if is_query:
        texts = [f"Instruct: {task_instruction}\nQuery: {t}" for t in texts]

    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**tokens)
        pooled = last_token_pool(outputs.last_hidden_state, tokens["attention_mask"])
        embeddings = F.normalize(pooled, p=2, dim=1)
    return embeddings


# === Disease embedding cache ===
disease_cache_path = os.path.join(output_root, "cached_disease_embeddings.npy")
disease_name_cache_path = os.path.join(output_root, "cached_disease_names.json")

global_cached_embeds = None
global_cached_diseases = None

def build_or_load_disease_cache(diseases):
    global global_cached_embeds, global_cached_diseases

    if global_cached_embeds is None:
        # === 🧩 Combine disease name + 4 section texts ===
        disease_texts = []
        disease_names = []
        for d in diseases:
            name = d.get("disease", "").strip()
            full_text = (
                f"{name}\n" #comment below lines for ablation without disease knowledge
                f"### Clinical Presentation:\n{d.get('clinical_presentation', '')}\n"
                f"### Diagnostic Evaluation:\n{d.get('diagnostic_evaluation', '')}\n"
                f"### Subtype Variant:\n{d.get('subtype_variant', '')}\n"
                f"### Management Therapy:\n{d.get('management_therapy', '')}"
            ).strip()
            disease_texts.append(full_text)
            disease_names.append(name)

        global_cached_diseases = disease_names

        # === 🧠 Load or build embeddings ===
        if os.path.exists(disease_cache_path) and os.path.exists(disease_name_cache_path):
            print("📦 Loading cached disease embeddings ...")
            global_cached_embeds = torch.tensor(np.load(disease_cache_path)).to(device)
        else:
            print("⚙️ Building disease embedding cache (with 4 sections) ...")
            doc_embs = encode_texts(disease_texts, is_query=False)
            np.save(disease_cache_path, doc_embs.cpu().numpy())
            with open(disease_name_cache_path, "w", encoding="utf-8") as f:
                json.dump(disease_names, f, ensure_ascii=False, indent=2)
            global_cached_embeds = doc_embs

    return global_cached_embeds, global_cached_diseases


# === Reranking logic ===
def rerank_by_embedding(case_text, diseases, top_k=None):
    if not diseases or not case_text.strip():
        return []

    doc_embs, disease_names = build_or_load_disease_cache(diseases)
    query_emb = encode_texts([case_text], is_query=True)  # shape: [1, dim]

    scores = torch.matmul(query_emb, doc_embs.T).squeeze(0).cpu().numpy()

    ranked = [
        {"disease": disease_names[i], "match_score": round(float(s), 4)}
        for i, s in enumerate(scores) if s > 0
    ]
    ranked.sort(key=lambda x: -x["match_score"])

    for i, entry in enumerate(ranked, start=1):
        entry["rerank_position"] = i

    return ranked[:top_k] if top_k else ranked


# === Process one patient JSON file ===
def process_json_file(json_path):
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    output_txt = os.path.join(output_root, f"{base_name}_cosine_reranked.txt")
    if os.path.exists(output_txt) and os.path.getsize(output_txt) > 0:
        print(f"⏭️ Skipping {json_path} (already processed)")
        return

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            entry = json.load(f)
        patient_id = entry.get("patient_id", base_name)
        case_text = entry.get("case_data", "")
        diseases = entry.get("diseases", [])

        reranked = rerank_by_embedding(case_text, diseases)

        with open(output_txt, "w", encoding="utf-8") as out_f:
            json.dump({"patient_id": patient_id, "reranked": reranked}, out_f, indent=2)

        print(f"✅ Reranked: {output_txt} ({len(reranked)} kept, {len(diseases)} total)")

    except Exception as e:
        print(f"❌ Error in {json_path}: {e}")
        with open(output_txt, "w", encoding="utf-8") as out_f:
            json.dump({"error": str(e)}, out_f)


# === Main driver ===
if __name__ == "__main__":
    all_json_files = sorted([f for f in os.listdir(input_root) if f.endswith(".json")])
    print(f"📂 Found {len(all_json_files)} JSON files in {input_root}")

    for json_file in tqdm(all_json_files, desc="📊 Reranking with Instruction", unit="file"):
        json_path = os.path.join(input_root, json_file)
        process_json_file(json_path)

    print(f"\n🎉 All done! Results saved to: {output_root}")

