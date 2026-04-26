import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler


class _MCTSModel:
    """MCTSのコアロジック（mcts.py / mcts_node.py）が依存する内部ラッパー。"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def contains_eos_id(self, token_list):
        eos_ids = getattr(self.tokenizer, "eos_token_ids", None)
        if eos_ids:
            return any(t in eos_ids for t in token_list)
        eos_id = self.tokenizer.eos_token_id
        if isinstance(eos_id, list):
            return any(t in token_list for t in eos_id)
        return eos_id in token_list

    def generate_single_step(
        self, input_ids, top_k=5, max_new_tokens=32, step_separator_ids=None
    ):
        """
        1ステップ分のテキスト生成を行う。

        Args:
            input_ids (List[int]): 現在までの入力トークン列。
            top_k (int): サンプリング時のtop_k値。デフォルト5。
            max_new_tokens (int): 1ステップで生成する最大トークン数。デフォルト32。
            step_separator_ids (List[int] | None): ステップ区切りトークンIDのリスト。

        Returns:
            Tuple[List[int], float]: 更新後のトークン列と信頼度スコア（平均）。
        """
        step_seps = set(step_separator_ids) if step_separator_ids else set()
        sampler = make_sampler(temp=1.0, top_k=top_k)
        prompt = mx.array(input_ids)

        gen_tokens = []
        confidence_scores = []

        for token_id, logprobs in generate_step(
            prompt, self.model, max_tokens=max_new_tokens, sampler=sampler
        ):
            # 信頼度スコア: 選択トークンのprob / top5のprob合計
            probs = mx.exp(logprobs)
            mx.eval(probs)
            top5_idx = np.argpartition(np.array(probs), -5)[-5:]
            top5_sum = float(np.sum(np.array(probs)[top5_idx]))
            c_i = float(probs[token_id].item()) / top5_sum if top5_sum > 0 else 0.0
            confidence_scores.append(c_i)

            gen_tokens.append(token_id)

            if self.contains_eos_id([token_id]):
                break
            if token_id in step_seps:
                break

        v = float(np.mean(confidence_scores)) if confidence_scores else 0.0
        return input_ids + gen_tokens, v
