from mlx_lm import generate, load


def main():
    model, tokenizer = load("mlx-community/Qwen3.5-9B-8bit")

    prompt = "hello"

    if (
        hasattr(tokenizer, "apply_chat_template")
        and tokenizer.chat_template is not None
    ):
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    response = generate(model, tokenizer, prompt=prompt, verbose=True, max_tokens=2000)
    print(response)


if __name__ == "__main__":
    main()
