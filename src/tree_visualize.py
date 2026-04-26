from typing import Optional, Set

import graphviz

from .mcts_node import MCTSNode


def get_best_path_node_ids(node: MCTSNode) -> Set[int]:
    """
    終端ノードからrootまで辿り、そのパス上のノードIDをセットで返す。

    Args:
        node (MCTSNode): 終端ノード

    Returns:
        Set[int]: パス上のノードのidを格納したセット
    """
    best_path = []
    n = node
    while n is not None:
        best_path.append(id(n))
        n = n.parent
    return set(best_path)


def get_root_node(node: MCTSNode) -> MCTSNode:
    """
    指定されたノードの最上位の親（ルート）ノードを取得する。

    Args:
        node (MCTSNode): 任意のノード

    Returns:
        MCTSNode: ルートノード
    """
    root_node = node
    while root_node.parent is not None:
        root_node = root_node.parent
    return root_node


def visualize_mcts_tree(
    node: MCTSNode,
    output_file: str = "mcts_tree",
    highlight_node: Optional[MCTSNode] = None,
):
    """
    MCTSの木構造を可視化する。
    指定されたノードの最上位の親（ルート）からツリーを構築する。

    Args:
        node (MCTSNode): 任意のノード（自動的にルートから可視化される）
        output_file (str, optional): 出力ファイル名（拡張子なし）。デフォルトは"mcts_tree"
        highlight_node (Optional[MCTSNode], optional): ハイライトする終端ノード。指定された場合、
            そのノードからルートまでのパスが☆でマークされる。
    """
    # ルートノードを取得
    root_node = get_root_node(node)
    dot = graphviz.Digraph(comment="MCTS Tree")
    dot.attr(rankdir="TB")  # Top to Bottom layout

    # ハイライトするパスを取得
    highlight_path = get_best_path_node_ids(highlight_node) if highlight_node else set()

    def add_nodes_edges(node: MCTSNode, node_id: str = "0"):
        # ノードの情報を作成
        value_avg = node.value_sum / max(1, node.visit_count)
        label = f"Visit Count: {node.visit_count}\nReward: {node.reward_score:.2f}\nValue Sum: {node.value_sum:.2f}\nAvg: {value_avg:.2f}"

        # ノードを追加（ハイライトパス上のノードは赤色で表示）
        attrs = (
            {"fillcolor": "lightpink", "style": "filled"}
            if id(node) in highlight_path
            else {}
        )
        dot.node(node_id, label, **attrs)

        # 子ノードを再帰的に追加
        for i, child in enumerate(node.children):
            child_id = f"{node_id}_{i}"
            add_nodes_edges(child, child_id)
            # エッジを追加（ハイライトパス上のエッジは赤色で表示）
            edge_attrs = (
                {"color": "red", "penwidth": "2.0"}
                if id(child) in highlight_path
                else {}
            )
            dot.edge(node_id, child_id, **edge_attrs)

    # ルートから再帰的にノードとエッジを追加
    add_nodes_edges(root_node)

    # グラフを保存
    dot.render(output_file, view=True)


def visualize_mcts_tree_with_best_path(
    final_node: MCTSNode, tokenizer, output_file: str = "mcts_tree_with_best_path"
):
    """
    終端ノードからルートまでの最良パスをハイライトしてツリーを可視化する。

    Args:
        final_node (MCTSNode): MCTS探索で最後に選ばれたノード
        tokenizer (PreTrainedTokenizer): トークン列を文字列に戻すためのトークナイザ
        output_file (str, optional): 出力ファイル名（拡張子なし）。デフォルトは"mcts_tree_with_best_path"
    """
    # 最上位のルートノードを取得
    root_node = get_root_node(final_node)

    # ツリーを可視化（最良パスをハイライト）
    visualize_mcts_tree_with_tokens(
        root_node, tokenizer, output_file, highlight_node=final_node
    )


def visualize_mcts_tree_with_tokens(
    node: MCTSNode,
    tokenizer,
    output_file: str = "mcts_tree_with_tokens",
    highlight_node: Optional[MCTSNode] = None,
):
    """
    MCTSの木構造をトークン情報付きで可視化する。
    指定されたノードの最上位の親（ルート）からツリーを構築し、オプションで特定のパスをハイライトする。

    Args:
        node (MCTSNode): 任意のノード（自動的にルートから可視化される）
        tokenizer (PreTrainedTokenizer): トークン列を文字列に変換するトークナイザ
        output_file (str, optional): 出力ファイル名（拡張子なし）。デフォルトは"mcts_tree_with_tokens"
    """
    # ルートノードを取得
    root_node = get_root_node(node)
    dot = graphviz.Digraph(comment="MCTS Tree with Tokens")
    dot.attr(rankdir="TB")

    # ハイライトするパスを取得
    highlight_path = get_best_path_node_ids(highlight_node) if highlight_node else set()

    def add_nodes_edges(node: MCTSNode, node_id: str = "0"):
        # ノードの情報を作成（トークンを含む）
        if node.action_tokens is not None:
            raw_text = tokenizer.decode(node.action_tokens, skip_special_tokens=False)
            action_text = raw_text.replace("\n", "\\n")
        else:
            action_text = "[ROOT]"
        value_avg = node.value_sum / max(1, node.visit_count)
        label = f"Action: {action_text}\nVisit Count: {node.visit_count}\nReward: {node.reward_score:.2f}\nValue Sum: {node.value_sum:.2f}\nAvg: {value_avg:.2f}"

        # ノードを追加（ハイライトパス上のノードは赤色で表示）
        attrs = (
            {"fillcolor": "lightpink", "style": "filled"}
            if id(node) in highlight_path
            else {}
        )
        dot.node(node_id, label, **attrs)

        # 子ノードを再帰的に追加
        for i, child in enumerate(node.children):
            child_id = f"{node_id}_{i}"
            add_nodes_edges(child, child_id)
            # エッジを追加（ハイライトパス上のエッジは赤色で表示）
            edge_attrs = (
                {"color": "red", "penwidth": "2.0"}
                if id(child) in highlight_path
                else {}
            )
            dot.edge(node_id, child_id, **edge_attrs)

    # ルートから再帰的にノードとエッジを追加
    add_nodes_edges(root_node)

    # グラフを保存
    dot.render(output_file, view=True)
