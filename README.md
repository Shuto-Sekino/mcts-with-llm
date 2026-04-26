# MCTS with LLM

Monte Carlo Tree Search (MCTS) を用いて LLM の推論品質を向上させるアルゴリズムの実装です。  
[mlx-lm](https://github.com/ml-explore/mlx-lm) をバックエンドとして Apple Silicon 上で動作します。

---

## 概要

通常の LLM テキスト生成は **貪欲サンプリング（greedy decoding）** で進めるため、一度生成した内容を見直すことができません。  
本実装では、各「推論ステップ」の選択に MCTS を挟むことで、複数の候補パスを先読みしてより信頼度の高い推論経路を選び取ります。

---

## モジュール構成

```
src/
├── __init__.py            # 公開 API (reasoning_generate, stream_reasoning_generate)
├── reasoning_generate.py  # メインループ：MCTSステップを繰り返し最終テキストを生成
├── mcts.py                # MCTS コアロジック (UCB1, Selection, Expansion, Backpropagation)
├── mcts_node.py           # ノードクラス（状態・統計値・展開処理）
└── reasoning_model.py     # LLM ラッパー（1ステップ生成＋信頼度スコア計算）
```

---

## アルゴリズム解説

### 全体フロー

```
prompt
  │
  ▼
[ルートノード作成]
  │
  └─ MCTSステップ (max_iterations 回)
       │
       ├─ 1. Selection   … UCB1 で葉ノードまで降下
       ├─ 2. Expansion   … LLM で子ノード (beam_size=2) を生成
       ├─ 3. Simulation  … 子ノードの信頼度スコアを報酬とする
       └─ 4. Backprop    … 報酬を親へ伝播
       │
       └─ 最高平均スコアの子を「現在のノード」に採用
            │
            └─ EOS or max_tokens に達したら終了
```

### 具体例：「17 × 23 を段階的に計算せよ」

以下では `iterations_per_step=5, top_k=5, mini_step_size=32` で動かしたときの動きを追います。

#### ステップ 0：ルートノードの初期化

```
root = MCTSNode(input_ids=[<プロンプトのトークン列>])
# visit_count=0, value_sum=0.0
```

#### ステップ 1：最初の MCTS サイクル（5 回のイテレーション）

**Iteration 1 — Backprop のみ（初回はスキップ）**

`expand_threshold=0` の場合、`visit_count > 0` が展開条件です（`mcts.py:90`）。  
初回はルートが未訪問のため、展開せず `backpropagate(node, node.reward_score)` のみ実行します。

**Iteration 2 — Expansion**

```python
# mcts_node.py:52-62
for _ in range(beam_size=2):
    new_ids, value = llm.generate_single_step(root.input_ids, top_k=5, max_new_tokens=32)
```

LLM が 2 通りの「思考の断片」を生成します。たとえば：

| 子ノード | 生成テキスト | 信頼度スコア |
|---------|-------------|------------|
| child_A | `"17 × 23 = 17 × 20 + 17 × 3"` | 0.72 |
| child_B | `"まず 17 × 23 を分解します。"` | 0.45 |

信頼度スコア（`reward_score`）は各トークンの確率を使って計算します。

```
c_i = P(選択トークン) / sum(top-5 トークンの確率)
reward_score = mean(c_i for i in 生成全トークン)
```

スコアが高い child_A の報酬が `backpropagate` で親に伝播されます。

**Selection（Iteration 3〜5）**

子ノードが増えると、UCB1 スコアで次に探索する子を選びます。

```
UCB1(child) = value_sum/visit_count + 1.41 * sqrt(log(parent.visit_count) / child.visit_count)
```

- 左項：**活用**（これまでの平均スコアが高い子を優先）
- 右項：**探索**（あまり訪問されていない子を優先）

未訪問ノードは `inf` を返すので、新しい子が必ず探索されます（`mcts.py:7-9`）。

#### ステップ 1 終了：最良の子を採用

```python
# mcts.py:112-117
best = max(root.children, key=lambda c: c.value_sum / c.visit_count)
# → child_A を採用
current_node = child_A
```

#### ステップ 2〜N：同じ MCTS サイクルを繰り返す

`current_node` を新たなルートとして同じ処理を繰り返します。  
生成テキストは各ステップの `action_tokens` を連結することで完成します。

```
"17 × 23 = 17 × 20 + 17 × 3"
→ "= 340 + 51"
→ "= 391"
→ (EOS) → 終了
```

---

## 実装の詳細

### 信頼度スコアの計算（`reasoning_model.py:43-62`）

```python
probs = mx.exp(logprobs)
top5_idx = np.argpartition(np.array(probs), -5)[-5:]
top5_sum = float(np.sum(np.array(probs)[top5_idx]))
c_i = float(probs[token_id].item()) / top5_sum
```

`top_k` 個の候補トークンの確率合計に対する選択トークンの比率を信頼度とします。  
1ステップ全トークンの平均が、そのノードの `reward_score` になります。

### UCB1 スコア（`mcts.py:6-22`）

```python
def ucb_score(parent, child, c_param=1.41):
    if child.visit_count == 0:
        return float("inf")
    return (child.value_sum / child.visit_count) + c_param * math.sqrt(
        math.log(parent.visit_count) / child.visit_count
    )
```

`c_param=1.41 ≈ √2` はチェスAI AlphaGo などでも使われる標準的な探索定数です。

### ノードの展開（`mcts_node.py:34-62`）

```python
def expand(self, llm, beam_size=2, mini_step_size=32, ...):
    for _ in range(beam_size):
        new_ids, value = llm.generate_single_step(self.input_ids, ...)
        diff_tokens = new_ids[len(self.input_ids):]   # 差分トークンのみ保持
        child = MCTSNode(new_ids, parent=self, action_tokens=diff_tokens)
        child.reward_score = value
        self.children.append(child)
```

各子ノードは「親の `input_ids` に追加されたトークン」だけを `action_tokens` として持ちます。

---

## 使い方

### セットアップ

```bash
# 依存パッケージのインストール
uv sync

# パッケージを追加する場合
uv add <package>
```

### 基本的な使い方

```python
from mlx_lm import load
from src import reasoning_generate

model, tokenizer = load("mlx-community/gemma-3-1b-it-8bit")

# <end_of_turn> を EOS として追加（Gemma 系モデルで必要）
eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
tokenizer.eos_token_ids.add(eot_id)

messages = [{"role": "user", "content": "17 × 23 を段階的に計算してください"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

result = reasoning_generate(
    model,
    tokenizer,
    prompt=prompt,
    verbose=True,
    max_tokens=256,
    top_k=5,
    iterations_per_step=5,
    max_iterations=15,
    mini_step_size=32,
)
print(result)
```

### ストリーミング API

```python
from src import stream_reasoning_generate

for step in stream_reasoning_generate(model, tokenizer, prompt, max_tokens=256):
    print(f"[Step {step.iteration}] {step.text}", end="", flush=True)
    if step.finish_reason:
        print(f"\n終了理由: {step.finish_reason}")
        break
```

---

## パラメータ一覧

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `max_tokens` | 256 | 生成する最大トークン数 |
| `top_k` | 5 | 各ステップでのサンプリング幅。大きいほど多様な候補を生成 |
| `iterations_per_step` | 5 | 1 MCTSステップあたりのイテレーション数 |
| `max_iterations` | 15 | MCTSステップの上限回数 |
| `mini_step_size` | 32 | 1ステップで生成する最大トークン数（= アクションの粒度） |
| `expand_threshold` | 0 | ノードを展開するために必要な `visit_count` の閾値 |
| `step_separator_ids` | None | ステップ区切りトークン ID（指定するとその位置でステップが分割される） |

---

## ファイル実行例

```bash
# 標準の mlx-lm 生成（比較用）
uv run mlx_example.py

# MCTS を用いた推論生成
uv run mcts_example.py

# ツリー可視化つき推論生成
uv run visualize_example.py
```

---

## 参考

本実装は [Hajime-Y/reasoning-model](https://github.com/Hajime-Y/reasoning-model) を元に、[mlx-lm](https://github.com/ml-explore/mlx-lm) を用いて Apple Silicon 向けに移植・実装したものです。
