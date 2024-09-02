from typing import Dict, List

from torch.nn import CrossEntropyLoss


def cross_entropy_loss(lm_logits, labels, reduction="sum"):
    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction=reduction)
    loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
    return loss


def multi_references_acc(predictions: List[int], references: List[List[int]]) -> Dict[str, float]:
    num_total = len(predictions)
    num_correct = sum([1 if predictions[i] in references[i] else 0 for i in range(num_total)])
    accuracy = float(num_correct / num_total)
    return {"acc (em)": accuracy}
