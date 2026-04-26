import time
from dataclasses import dataclass
from typing import Generator, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .mcts import mcts_search
from .mcts_node import MCTSNode
from .reasoning_model import _MCTSModel


@dataclass
class ReasoningStepResponse:
    """
    stream_reasoning_generate が各MCTSステップごとに返すレスポンス。

    Args:
        text (str): このステップで選択された推論テキスト。
        step_tokens (int): このステップで選択されたトークン数。
        total_tokens (int): 生成済みトークンの累計。
        iteration (int): 何番目のMCTSステップか（1始まり）。
        finish_reason (str | None): "stop"（EOS到達）| "length"（max_tokens到達）| None。
    """

    text: str
    step_tokens: int
    total_tokens: int
    iteration: int
    node: Optional["MCTSNode"] = None
    finish_reason: Optional[str] = None


def stream_reasoning_generate(
    model: nn.Module,
    tokenizer,
    prompt: Union[str, List[int]],
    max_tokens: int = 256,
    *,
    top_k: int = 5,
    iterations_per_step: int = 5,
    max_iterations: int = 15,
    mini_step_size: int = 32,
    expand_threshold: int = 0,
    step_separator_ids: Optional[List[int]] = None,
) -> Generator[ReasoningStepResponse, None, None]:
    """
    MCTSによる推論生成を行い、1ステップ選択ごとに ReasoningStepResponse を yield する。

    Args:
        model (nn.Module): mlx-lm でロードしたモデル。
        tokenizer: 対応するトークナイザ。
        prompt (str | List[int]): 入力プロンプト（文字列またはトークンIDのリスト）。
        max_tokens (int): 生成する最大トークン数。デフォルト256。
        top_k (int): MCTS探索時のtop-kサンプリング幅。デフォルト5。
        iterations_per_step (int): 各MCTSステップでの探索反復回数。デフォルト5。
        max_iterations (int): MCTSステップの最大数。デフォルト15。
        mini_step_size (int): 1ステップあたりの最大生成トークン数。デフォルト32。
        expand_threshold (int): ノード拡張に必要なvisit_countの閾値。デフォルト0。
        step_separator_ids (List[int] | None): Step as Action のステップ区切りトークンIDリスト。

    Yields:
        ReasoningStepResponse: MCTSの1ステップ選択ごとのレスポンス。
    """
    if isinstance(prompt, str):
        add_special_tokens = (
            tokenizer.bos_token is None
            or not prompt.startswith(tokenizer.bos_token)
        )
        input_ids = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    elif isinstance(prompt, mx.array):
        input_ids = prompt.tolist()
    else:
        input_ids = list(prompt)

    llm = _MCTSModel(model, tokenizer)
    current_node = MCTSNode(input_ids)
    total_tokens = 0

    for i in range(max_iterations):
        best_node = mcts_search(
            current_node,
            llm,
            iterations=iterations_per_step,
            mini_step_size=mini_step_size,
            expand_threshold=expand_threshold,
            step_separator_ids=step_separator_ids,
            top_k=top_k,
        )

        step_tokens = best_node.action_tokens or []
        total_tokens += len(step_tokens)
        text = tokenizer.decode(step_tokens, skip_special_tokens=True) if step_tokens else ""

        finish_reason = None
        if best_node.is_terminal(llm):
            finish_reason = "stop"
        elif total_tokens >= max_tokens:
            finish_reason = "length"

        yield ReasoningStepResponse(
            text=text,
            step_tokens=len(step_tokens),
            total_tokens=total_tokens,
            iteration=i + 1,
            node=best_node,
            finish_reason=finish_reason,
        )

        current_node = best_node
        if finish_reason is not None:
            break


def reasoning_generate(
    model: nn.Module,
    tokenizer,
    prompt: Union[str, List[int]],
    verbose: bool = False,
    **kwargs,
) -> str:
    """
    MCTSによる推論生成を行い、最終的な生成テキストを返す。
    mlx_lm.generate と同じ使い方ができる。

    Args:
        model (nn.Module): mlx-lm でロードしたモデル。
        tokenizer: 対応するトークナイザ。
        prompt (str | List[int]): 入力プロンプト（文字列またはトークンIDのリスト）。
        verbose (bool): Trueのとき、生成中のテキストとタイミング情報を出力する。デフォルトFalse。
        **kwargs: stream_reasoning_generate に渡す追加の引数。

    Returns:
        str: 生成されたテキスト。
    """
    if verbose:
        print("=" * 10)
        tic = time.perf_counter()

    text = ""
    response = None

    for response in stream_reasoning_generate(model, tokenizer, prompt, **kwargs):
        if verbose:
            print(response.text, end="", flush=True)
        text += response.text

    if verbose:
        elapsed = time.perf_counter() - tic
        print()
        print("=" * 10)
        if response is None:
            print("No text generated for this prompt")
        else:
            tps = response.total_tokens / elapsed if elapsed > 0 else 0.0
            print(
                f"Generation: {response.total_tokens} tokens, "
                f"{tps:.3f} tokens-per-sec "
                f"({response.iteration} MCTS steps)"
            )
            print(f"Peak memory: {mx.get_peak_memory() / 1e9:.3f} GB")

    return text
