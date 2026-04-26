from mlx_lm import load

from src import reasoning_generate

MODEL = "mlx-community/gemma-3-1b-it-8bit"


def apply_chat_template(tokenizer, prompt: str) -> str:
    if (
        hasattr(tokenizer, "apply_chat_template")
        and tokenizer.chat_template is not None
    ):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt


# ---------------------------------------------------------------------------
# Example 1: reasoning_generate — mlx_lm.generate と同じ使い方
# ---------------------------------------------------------------------------
def example_basic(model, tokenizer):
    print("\n" + "=" * 60)
    print("Example 1: reasoning_generate (basic)")
    print("=" * 60)

    prompt = apply_chat_template(tokenizer, "pleae calculate 17 x 23 step by step")

    output = reasoning_generate(
        model,
        tokenizer,
        prompt=prompt,
        verbose=True,  # 生成中テキストとタイミングを表示
        max_tokens=256,
        # MCTS パラメータ
        top_k=5,  # 探索時の top-k サンプリング幅
        iterations_per_step=5,
        max_iterations=15,
        mini_step_size=32,
    )
    print("Result:", output)


if __name__ == "__main__":
    print(f"Loading model: {MODEL}")
    model, tokenizer = load(MODEL)

    # <end_of_turn> を EOS として追加
    eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    tokenizer.eos_token_ids.add(eot_id)

    example_basic(model, tokenizer)
