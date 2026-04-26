from mlx_lm import load

from src import stream_reasoning_generate
from src.tree_visualize import (
    visualize_mcts_tree,
    visualize_mcts_tree_with_best_path,
    visualize_mcts_tree_with_tokens,
)

MODEL = "mlx-community/gemma-3-1b-it-8bit"
PROMPT = "please calculate 17 x 23 step by step"


def main():
    print(f"Loading model: {MODEL}")
    model, tokenizer = load(MODEL)

    eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    tokenizer.eos_token_ids.add(eot_id)

    messages = [{"role": "user", "content": PROMPT}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    print("Running MCTS reasoning...\n" + "=" * 40)

    final_step = None
    for step in stream_reasoning_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=256,
        top_k=5,
        iterations_per_step=5,
        max_iterations=15,
        mini_step_size=32,
    ):
        print(step.text, end="", flush=True)
        final_step = step

    print("\n" + "=" * 40)

    if final_step is None or final_step.node is None:
        print("No nodes generated.")
        return

    final_node = final_step.node

    # ① ノード統計のみのツリー（シンプル）
    print("Saving mcts_tree.pdf ...")
    visualize_mcts_tree(final_node, output_file="mcts_tree")

    # ② 各ノードの生成テキスト付きツリー
    print("Saving mcts_tree_tokens.pdf ...")
    visualize_mcts_tree_with_tokens(
        final_node, tokenizer, output_file="mcts_tree_tokens"
    )

    # ③ 採択パスをピンクでハイライトしたツリー
    print("Saving mcts_best_path.pdf ...")
    visualize_mcts_tree_with_best_path(
        final_node, tokenizer, output_file="mcts_best_path"
    )

    print("Done. Three PDF files have been saved and opened.")


if __name__ == "__main__":
    main()
