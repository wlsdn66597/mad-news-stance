"""분류 지표 (sklearn 없이 자체 구현): accuracy, per-class P/R/F1, macro-F1, confusion matrix."""

LABELS_DEFAULT = ["supportive", "oppositional", "neutral"]


def accuracy(preds, golds):
    if not preds:
        return 0.0
    return sum(1 for p, g in zip(preds, golds) if p == g) / len(preds)


def per_class_prf(preds, golds, labels):
    out = {}
    for c in labels:
        tp = sum(1 for p, g in zip(preds, golds) if p == c and g == c)
        fp = sum(1 for p, g in zip(preds, golds) if p == c and g != c)
        fn = sum(1 for p, g in zip(preds, golds) if g == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[c] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
    return out


def macro_f1(preds, golds, labels):
    prf = per_class_prf(preds, golds, labels)
    return sum(v["f1"] for v in prf.values()) / len(labels)


def confusion(preds, golds, labels):
    """row=gold, col=pred. 마지막 열은 파싱 실패(None/미분류)."""
    li = {l: i for i, l in enumerate(labels)}
    mat = [[0] * (len(labels) + 1) for _ in labels]
    for p, g in zip(preds, golds):
        if g not in li:
            continue
        mat[li[g]][li[p] if p in li else -1] += 1
    return mat


def format_report(preds, golds, labels=LABELS_DEFAULT):
    lines = [f"accuracy={accuracy(preds, golds):.3f}   macro_f1={macro_f1(preds, golds, labels):.3f}"]
    prf = per_class_prf(preds, golds, labels)
    for c in labels:
        v = prf[c]
        lines.append(f"  {c:12s} P={v['precision']:.2f} R={v['recall']:.2f} "
                     f"F1={v['f1']:.2f} (support={v['support']})")
    mat = confusion(preds, golds, labels)
    lines.append("  confusion (row=gold, col=pred, 마지막=none):")
    lines.append("           " + " ".join(f"{l[:4]:>5s}" for l in labels) + "  none")
    for i, c in enumerate(labels):
        row = " ".join(f"{mat[i][j]:5d}" for j in range(len(labels)))
        lines.append(f"  {c[:8]:>8s}  {row} {mat[i][-1]:5d}")
    return "\n".join(lines)
