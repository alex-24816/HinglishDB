"""
HinglishDB — Evaluation Script
Computes BLEU, SQAM, and TSED metrics on model predictions.
"""

import json, re, os, argparse
import sqlglot
from sqlglot import exp
from sacrebleu.metrics import BLEU


# ── BLEU ──────────────────────────────────────────────────────
def compute_bleu(predictions: list) -> float:
    bleu = BLEU(tokenize="none")
    return bleu.corpus_score(
        [p["prediction"] for p in predictions],
        [[p["gold"] for p in predictions]]
    ).score


# ── SQAM ──────────────────────────────────────────────────────
def parse_sql_clauses(sql: str) -> dict:
    sql = sql.strip().rstrip(";")
    patterns = {
        "SELECT":   r"SELECT\s+(.*?)\s+FROM",
        "FROM":     r"FROM\s+(.*?)(?=\s+WHERE|\s+GROUP BY|\s+HAVING|\s+ORDER BY|\s+LIMIT|$)",
        "WHERE":    r"WHERE\s+(.*?)(?=\s+GROUP BY|\s+HAVING|\s+ORDER BY|\s+LIMIT|$)",
        "GROUP BY": r"GROUP BY\s+(.*?)(?=\s+HAVING|\s+ORDER BY|\s+LIMIT|$)",
        "HAVING":   r"HAVING\s+(.*?)(?=\s+ORDER BY|\s+LIMIT|$)",
        "ORDER BY": r"ORDER BY\s+(.*?)(?=\s+LIMIT|$)",
    }
    return {
        clause: (m.group(1).strip() if (m := re.search(p, sql,
                 re.IGNORECASE | re.DOTALL)) else "")
        for clause, p in patterns.items()
    }


def split_where(text: str) -> set:
    if not text:
        return set()
    parts = re.split(r"\s+(?:AND|OR)\s+", text, flags=re.IGNORECASE)
    return set(re.sub(r"\bt\d+\.", "", p.strip().lower()) for p in parts)


def split_clause(text: str) -> set:
    if not text:
        return set()
    return set(re.sub(r"`", "", re.sub(r"\bt\d+\.", "", p.strip().lower()))
               for p in text.split(","))


def compute_sqam(pred: str, gold: str) -> float:
    weights = {"SELECT": 0.30, "FROM": 0.20, "WHERE": 0.25,
               "GROUP BY": 0.10, "HAVING": 0.05, "ORDER BY": 0.10}
    pc, gc = parse_sql_clauses(pred), parse_sql_clauses(gold)
    score = 0.0
    for clause, w in weights.items():
        fn = split_where if clause in ("WHERE", "HAVING") else split_clause
        ps, gs = fn(pc[clause]), fn(gc[clause])
        if not gs and not ps:
            cs = 1.0
        elif not gs or not ps:
            cs = 0.0
        else:
            cs = len(ps & gs) / len(ps | gs)
        score += w * cs
    return score


# ── TSED ──────────────────────────────────────────────────────
def ast_nodes(node) -> list:
    nodes = []
    def walk(n, d=0):
        if n is None:
            return
        r = type(n).__name__
        if hasattr(n, "this") and isinstance(n.this, str):
            r += f":{n.this.lower()}"
        nodes.append((d, r))
        for child in n.args.values():
            if isinstance(child, list):
                for c in child:
                    if isinstance(c, exp.Expression):
                        walk(c, d + 1)
            elif isinstance(child, exp.Expression):
                walk(child, d + 1)
    walk(node)
    return nodes


def tree_edit_distance(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1,
                           dp[i-1][j-1] + cost)
    return dp[m][n]


def compute_tsed(pred: str, gold: str) -> float:
    try:
        pa = sqlglot.parse_one(pred, read="sqlite")
        ga = sqlglot.parse_one(gold, read="sqlite")
    except Exception:
        return 0.0
    pn, gn = ast_nodes(pa), ast_nodes(ga)
    D = tree_edit_distance(pn, gn)
    N = max(len(pn), len(gn))
    return max(0.0, 1.0 - D / N) if N > 0 else 1.0


# ── Main ──────────────────────────────────────────────────────
def evaluate(predictions_path: str) -> dict:
    with open(predictions_path, encoding="utf-8") as f:
        predictions = json.load(f)

    bleu = compute_bleu(predictions)
    sqam_scores = [compute_sqam(p["prediction"], p["gold"])
                   for p in predictions]
    tsed_scores = [compute_tsed(p["prediction"], p["gold"])
                   for p in predictions]

    avg_sqam = sum(sqam_scores) / len(sqam_scores)
    avg_tsed = sum(tsed_scores) / len(tsed_scores)

    results = {
        "num_predictions": len(predictions),
        "bleu":  round(bleu, 2),
        "sqam":  round(avg_sqam, 4),
        "tsed":  round(avg_tsed, 4),
    }

    print("=" * 45)
    print(f"Results: {os.path.basename(predictions_path)}")
    print("=" * 45)
    print(f"  BLEU : {results['bleu']}")
    print(f"  SQAM : {results['sqam']}")
    print(f"  TSED : {results['tsed']}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True,
                        help="Path to predictions JSON file")
    parser.add_argument("--output", default=None,
                        help="Optional path to save metrics JSON")
    args = parser.parse_args()

    metrics = evaluate(args.predictions)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {args.output}")
