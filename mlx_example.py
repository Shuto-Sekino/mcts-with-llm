from mlx_lm import generate, load


def main():
    model, tokenizer = load("mlx-community/gemma-3-1b-it-8bit")

    # <end_of_turn> を EOS として追加
    eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    tokenizer.eos_token_ids.add(eot_id)

    prompt = "pleae calculate 17 x 23 step by step"

    if (
        hasattr(tokenizer, "apply_chat_template")
        and tokenizer.chat_template is not None
    ):
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    generate(model, tokenizer, prompt=prompt, verbose=True, max_tokens=2000)


if __name__ == "__main__":
    main()
