import math
from typing import Dict, Iterable, List


def pagerank(
    nodes: Iterable[str],
    edges: Dict[str, List[str]],
    damping: float = 0.85,
    iterations: int = 20,
) -> Dict[str, float]:
    nodes = list(nodes)
    n = len(nodes)
    if n == 0:
        return {}
    score = {node: 1.0 / n for node in nodes}
    for _ in range(iterations):
        new_score = {node: (1.0 - damping) / n for node in nodes}
        sink_mass = 0.0
        for node in nodes:
            outs = edges.get(node, [])
            if not outs:
                sink_mass += score[node]
                continue
            share = score[node] / len(outs)
            for dst in outs:
                new_score[dst] = new_score.get(dst, 0.0) + damping * share
        if sink_mass > 0:
            sink_share = damping * sink_mass / n
            for node in nodes:
                new_score[node] += sink_share
        score = new_score
    return score
