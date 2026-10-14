# 手首を傾けた接触操作を VLA に学ばせる（Physical AI 2026 応用編 1 最終課題）

OmniCAMPUS アカウント名: yniwaeiai

引き出し・電子レンジの扉に触れるとき、真下を向いた手では手や前腕が棚にぶつかる。人間は手首を傾けてこれを避ける。
本リポジトリは、(1) 人間デモの手首の傾きの解析、(2) 傾けた姿勢で動くスクリプト expert による教師デモ生成、(3) 公開 VLA **π0.5-LIBERO** の **LoRA 微調整**、(4) 摂動付き閉ループ評価と失敗の構造分析、のコード・記録一式である。

- シミュレータ: MuJoCo（LIBERO-plus、robosuite 1.4.1）、ロボット: Franka Panda（OSC_POSE）
- タスク: ① 中段引き出しを開ける（libero_goal）② 下段引き出しを閉める（libero_90 KITCHEN_SCENE4 単独）③ 電子レンジを閉める（libero_90 KITCHEN_SCENE6 単独）
- AI: 推論（π0.5-LIBERO）と学習（LoRA r=16、α=32、attention+MLP、2400 step）

## 主な結果（摂動つき閉ループ評価、各 48 本、OSMesa 描画）
| タスク | 素の π0.5: 成功 / 衝突なし | LoRA v1: 成功 / 衝突なし | LoRA v2: 成功 / 衝突なし |
|---|---|---|---|
| 中段引き出し | 62% / 62%（24 本） | 85% / 85% | 83% / 81% |
| 下段引き出し | 79% / 0%（24 本） | 94% / 44% | 92% / 31% |
| 電子レンジ | 38% / 38%（24 本） | 98% / 98% | 98% / 98% |

v2 は強い摂動の expert データを追加した再学習で、事前に決めた基準では改善を検出できなかった（否定的結果）。詳細はレポートと `docs/作業履歴_最終課題.md`（判断・試行・失敗と訂正を時系列で記録）を参照。

## 構成
| パス | 内容 |
|---|---|
| `scripts/analyze_*.py`, `extract_*_keyposes.py` | 人間デモの手首の傾きの解析、接触区間、キーポーズの抽出 |
| `scripts/run_drawer_hook.py`, `run_close_push.py`, `run_micro_close.py` | 3 タスクの expert（全姿勢追従の制御器 `HookRunner` を含む）と教師デモの生成 |
| `scripts/kadai_record.py` | 教師デモの HDF5 記録（agentview・wrist・状態・行動） |
| `scripts/train_lora_kadai.py` | openpi の `train_lora_pi05.py` を元にした LoRA 学習（`train_lora_pi05.py` も同梱） |
| `scripts/eval_kadai_closedloop.py`, `serve_policy_noise.py`, `pod_*.sh`, `run_exp2.sh` | 方策サーバ + 並列クライアントの閉ループ評価、摂動試験、動画用ロールアウトの撮影 |
| `scripts/summarize_eval.py`, `compare_variants.py`, `analyze_*` | 集計・対応のある比較・失敗の分類 |
| `scripts/make_failure_sheets.py`, `report/make_figures.py`, `report/make_video.py` | 失敗例のコマ送り、レポートの図、提出動画の組み立て |
| `report/` | レポートの HTML 原稿と図、動画の構成（`video_spec.json`） |
| `results/` | 評価の生データ（`eval_runs_20261013/*.jsonl`）、集計、学習曲線、キーポーズ |
| `figs/` | 素 / LoRA の失敗例のコマ送り画像 |
| `docs/` | 計画、人間デモ解析の結果、作業履歴 |

## 再現に必要なもの（本リポジトリに含まれないもの）
- LIBERO-plus、openpi（PyTorch 版）、LIBERO の人間デモ（libero_goal / libero_90）、π0.5-LIBERO の重み（`gs://openpi-assets/checkpoints/pi05_libero` を PyTorch に変換）。
- スクリプト内の `/path/to/work/...` は環境に合わせて書き換える（環境変数 `LIBERO_ROOT` などで上書き可）。
- 学習データ（HDF5）、LoRA 重み、動画フレームは容量のため含まない。教師デモは `run_*` の expert で再生成できる（seed とパラメータは `results/gen_*.json`）。
- 学習は L40S 1 枚で約 3.5 時間（5.2 s/step）。

## AI 支援の使用について
実装・実験・記録は Claude Code（Anthropic の AI コーディング支援）を用いて行った。判断・方針は受講者が承認した。

## 第三者のコード
openpi（Apache-2.0）、LIBERO / LIBERO-plus、robosuite、MuJoCo を利用している。`scripts/train_lora_pi05.py` は openpi 由来のコードを含む。
