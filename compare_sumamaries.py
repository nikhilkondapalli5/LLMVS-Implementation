import argparse
from pathlib import Path

from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer, util
import torch


def load_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return p.read_text(encoding="utf-8").strip()

# function to compute ROUGE
def compute_rouge(ref: str, pred: str):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(ref, pred)
    return scores

# function to compute BERTScore
def compute_bertscore(ref: str, pred: str, lang: str = "en"):
    P, R, F1 = bert_score([pred], [ref], lang=lang, verbose=False)
    return {
        "precision": float(P.mean().item()),
        "recall": float(R.mean().item()),
        "f1": float(F1.mean().item()),
    }

# function to compute embedding cosine similarity
def compute_embedding_cosine(ref: str, pred: str, model_name: str = "all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    emb_ref = model.encode(ref, convert_to_tensor=True)
    emb_pred = model.encode(pred, convert_to_tensor=True)

    sim = util.cos_sim(emb_ref, emb_pred)
    # sim is 1x1 tensor
    return float(sim.item())


def main():
    parser = argparse.ArgumentParser(
        description="Compare summaries of long and short video."
    )
    parser.add_argument(
        "--long_summary",
        type=str,
        required=True,
        help="Path to text file with summary of the long/original video.",
    )
    parser.add_argument(
        "--short_summary",
        type=str,
        required=True,
        help="Path to text file with summary of the short/30s video.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Language code for BERTScore (default: en).",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name for cosine similarity.",
    )

    args = parser.parse_args()

    long_summary = load_text(args.long_summary)
    short_summary = load_text(args.short_summary)

    print("=== Long summary ===")
    print(long_summary)
    print("\n=== Short summary ===")
    print(short_summary)
    print("\n======================\n")

    # ROUGE
    rouge = compute_rouge(ref=long_summary, pred=short_summary)
    print("ROUGE scores (short vs long):")
    for k, v in rouge.items():
        print(
            f"  {k}: P={v.precision:.4f}, R={v.recall:.4f}, F1={v.fmeasure:.4f}"
        )

    # BERTScore
    bert = compute_bertscore(ref=long_summary, pred=short_summary, lang=args.lang)
    print("\nBERTScore (short vs long):")
    print(f"  Precision: {bert['precision']:.4f}")
    print(f"  Recall:    {bert['recall']:.4f}")
    print(f"  F1:        {bert['f1']:.4f}")

    # Embedding cosine similarity
    print("\nComputing embedding cosine similarity...")
    cosine_sim = compute_embedding_cosine(
        ref=long_summary, pred=short_summary, model_name=args.embedding_model
    )
    print(f"Embedding cosine similarity: {cosine_sim:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    # Make sure torch uses CPU if GPU isn't needed
    torch.set_num_threads(4)
    main()