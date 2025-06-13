import json
from typing import List, Tuple
import string
import re

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punctuation(text):
        return text.translate(str.maketrans('', '', string.punctuation))

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punctuation(lower(s))))

def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    common = set(pred_tokens) & set(truth_tokens)
    num_same = sum(min(pred_tokens.count(token), truth_tokens.count(token)) for token in common)

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)

def evaluate(predictions: List[str], references: List[str]) -> float:
    """Compute average F1 score across all QnA pairs."""
    assert len(predictions) == len(references), "Prediction and reference counts must match."
    scores = [f1_score(p, r) for p, r in zip(predictions, references)]
    return sum(scores) / len(scores)

# Example Usage:
if __name__ == "__main__":
    # Replace these with your own outputs and references
    with open('llama_8b.json', 'r') as f:
        data = json.load(f)
    with open('dpo_ready_data_test.json', 'r') as f:
        ground_truths = json.load(f)
    ground_truths = [item["chosen"] for item in ground_truths]
    # for temp in ['./dpo-llama-output/checkpoint-104', "./dpo-llama-output/checkpoint-208", "./dpo-llama-output/checkpoint-309"]:
    model_outputs = [item['answer'][len(item['prompt']):] for item in data]
    avg_f1 = evaluate(model_outputs, ground_truths)
    print(f"Average F1 Score: {avg_f1:.4f} model: ")