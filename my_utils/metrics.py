import re
from itertools import groupby

import torch

# -------------------------------------------- METRICS:


def levenshtein(a: list[str], b: list[str]) -> int:
    n, m = len(a), len(b)

    if n > m:
        a, b = b, a
        n, m = m, n

    current = range(n + 1)
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]


def compute_ser(y_true: list[list[str]], y_pred: list[list[str]]) -> float:
    ed_acc = 0
    length_acc = 0
    for t, h in zip(y_true, y_pred):
        ed_acc += levenshtein(t, h)
        length_acc += len(t)
    return 100.0 * ed_acc / length_acc


def compute_metrics(
    y_true: list[list[str]], y_pred: list[list[str]]
) -> dict[str, float]:
    metrics = {"ser": compute_ser(y_true, y_pred)}
    return metrics


# -------------------------------------------- CTC DECODERS:


def standard_ctc_greedy_decoder(
    y_pred: torch.Tensor,
    i2w: dict[int, str],
    blank_padding_token: int,
) -> list[str]:
    # y_pred = [seq_len, num_classes]
    # Best path
    y_pred_decoded = torch.argmax(y_pred, dim=1)
    # Merge repeated elements
    y_pred_decoded = torch.unique_consecutive(y_pred_decoded, dim=0).tolist()
    # Convert to string (remove CTC blank token)
    y_pred_decoded = [i2w[i] for i in y_pred_decoded if i != blank_padding_token]
    # Convert to split-sequence encoding to ensure a fair comparison between both encodings
    y_pred_decoded = re.split(r"\s+|:", " ".join(y_pred_decoded))
    return y_pred_decoded


def split_ctc_greedy_decoder(
    y_pred: torch.Tensor,
    i2w: dict[int, str],
    blank_padding_token: int,
) -> list[str]:
    # y_pred = [seq_len, num_classes]

    y_pred_decoded = []

    prev_blank = False
    prev_feat = None
    prev_type = 1  # Force the first decoded token feature (that is not a CTC-blank label) to be of glyph type
    # Iterate over each time step
    for ts in y_pred:
        # Iterate over each category in descending order in each time step
        for cat in torch.argsort(ts, descending=True).tolist():
            # Get decoded token feature for current time step
            cfeat = i2w[cat] if cat != blank_padding_token else "blank"
            # Get token feature type (glyph or position); glyph = 0, position = 1
            ctype = 1 if cfeat[0] == "S" or cfeat[0] == "L" else 0
            # Append token feature
            if cfeat == "blank":
                y_pred_decoded.append(cfeat)
                # prev_feat, prev_type = prev_feat, prev_type
                prev_blank = True
                break
            elif cfeat == prev_feat and ctype == prev_type and not prev_blank:
                y_pred_decoded.append(cfeat)
                # prev_feat, prev_type = prev_feat, prev_type
                # prev_blank = False
                break
            elif cfeat != prev_feat and ctype != prev_type:
                y_pred_decoded.append(cfeat)
                prev_feat, prev_type = cfeat, ctype
                prev_blank = False
                break
    # Merge repeated elements
    y_pred_decoded = [k for k, _ in groupby(y_pred_decoded)]
    # Delete CTC-blank labels
    y_pred_decoded = [sym for sym in y_pred_decoded if sym != "blank"]
    return y_pred_decoded
