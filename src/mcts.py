import math

from .mcts_node import MCTSNode


def ucb_score(parent, child, c_param=1.41):
    """
    UCB1スコアを計算する関数。

    Args:
        parent (MCTSNode): 親ノード
        child (MCTSNode): 子ノード
        c_param (float): 探索パラメータ。デフォルト1.41。

    Returns:
        float: UCB1スコア
    """
    if child.visit_count == 0:
        return float("inf")
    return (child.value_sum / child.visit_count) + c_param * math.sqrt(
        math.log(parent.visit_count) / child.visit_count
    )


def select_child(node):
    """
    UCBスコアに基づいて子ノードを1つ選択する。

    Args:
        node (MCTSNode): 現在のノード

    Returns:
        MCTSNode: 選択された子ノード
    """
    best_score = -float("inf")
    best_child = None
    for child in node.children:
        score = ucb_score(node, child)
        if score > best_score:
            best_score = score
            best_child = child
    return best_child


def backpropagate(node, value):
    """
    子ノードの評価値を親へと遡って更新する。

    Args:
        node (MCTSNode): 評価を開始するノード
        value (float): 更新する価値(報酬)
    """
    while node is not None:
        node.visit_count += 1
        node.value_sum += value
        node = node.parent


def mcts_search(
    root: MCTSNode,
    llm,
    iterations=5,
    mini_step_size=32,
    expand_threshold=0,
    step_separator_ids=None,
    top_k=5,
):
    """
    MCTS探索をrootノードから指定回数繰り返し、最良と判断される子ノードを返す。

    Args:
        root (MCTSNode): ルートノード
        llm (_MCTSModel): モデル
        iterations (int): MCTSの繰り返し回数。デフォルト5。
        mini_step_size (int): 1ステップでの最大生成トークン数。デフォルト32。
        expand_threshold (int): ノードを拡張するために必要なvisit_countの閾値。デフォルト0。
        step_separator_ids (List[int] | None): ステップ区切りトークンIDのリスト。
        top_k (int): サンプリング時のtop_k値。デフォルト5。

    Returns:
        MCTSNode: 最良の子ノード
    """
    for _ in range(iterations):
        # Selection
        node: MCTSNode = root
        while not node.is_leaf():
            node = select_child(node)

        # Expansion
        if node.visit_count > expand_threshold:
            node.expand(
                llm,
                beam_size=2,
                mini_step_size=mini_step_size,
                step_separator_ids=step_separator_ids,
                top_k=top_k,
            )
        else:
            backpropagate(node, node.reward_score)
            continue

        # Backpropagation
        if len(node.children) == 0:
            backpropagate(node, 0.0)
        else:
            best_child = max(node.children, key=lambda c: c.reward_score)
            backpropagate(best_child, best_child.reward_score)

    if not root.children:
        return root

    return max(
        root.children,
        key=lambda c: (
            c.value_sum / c.visit_count if c.visit_count > 0 else -float("inf")
        ),
    )
