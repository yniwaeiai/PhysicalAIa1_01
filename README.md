# 手首を傾けた接触操作を VLA に学ばせる（Physical AI 2026 応用編 1 最終課題）

OmniCAMPUS アカウント名: ModaleAI

引き出し・電子レンジの扉に触れるとき、真下を向いた手では手や前腕が棚にぶつかる。人間は手首を傾けてこれを避ける。
本リポジトリは、(1) 人間デモの手首の傾きの解析、(2) 傾けた姿勢で動くスクリプト expert による教師デモ生成、(3) 公開 VLA **π0.5-LIBERO** の **LoRA 微調整**、(4) 摂動付き閉ループ評価と失敗の構造分析、(5) 確率的に失敗する原因の追究（trace・同一状態からの復元と分岐・入れ替え介入）と、2 つの対処（回復データの追加学習、成功・失敗の対照学習。いずれも否定的結果）のコード・記録一式である。

- シミュレータ: MuJoCo（LIBERO-plus、robosuite 1.4.1）、ロボット: Franka Panda（OSC_POSE）
- タスク: ① 中段引き出しを開ける（libero_goal）② 下段引き出しを閉める（libero_90 KITCHEN_SCENE4 単独）③ 電子レンジを閉める（libero_90 KITCHEN_SCENE6 単独）
- AI: 推論（π0.5-LIBERO）と学習（LoRA r=16、α=32、attention+MLP、2400 step）

## 主な結果（摂動つき閉ループ評価、各 48 本、OSMesa 描画）
| タスク | 素の π0.5: 成功 / 衝突なし | LoRA v1: 成功 / 衝突なし | LoRA v2: 成功 / 衝突なし |
|---|---|---|---|
| 中段引き出し | 62% / 62%（24 本） | 85% / 85% | 83% / 81% |
| 下段引き出し | 79% / 0%（24 本） | 94% / 44% | 92% / 31% |
| 電子レンジ | 38% / 38%（24 本） | 98% / 98% | 98% / 98% |

v2 は強い摂動の expert データを追加した再学習で、事前に決めた基準では改善を検出できなかった（否定的結果）。

### 追加実験: 確率的失敗の原因追究と 2 つの対処（否定的結果）
同じ初期条件でも成否が変わる失敗（確率的な群）の原因を、停留中の行動記録（trace）、同一状態からの復元と分岐、観測・状態の入れ替え介入で追った。
- **原因（`docs/計画_確率的失敗の原因追究_v2_20261014.md` §18）**: 配置で決まる群は expert 自身がその配置で失敗する（教師の限界）。確率的な群は、接近の最初の数チャンクで教師の経路から 1〜2 cm 外れる共変量シフトで、成否は手先の位置に依存し、接触で停留する（作業仮説）。
- **対処 R-a（v3 = 回復データの追加学習、同 §19）**: 方策の rollout から分岐して expert に引き継いだ回復デモ 858 本（68,081 フレーム）を v1 の教師に加えて再学習（2,000 step + 再開 400 step）。事前登録した線 V1〜V4 のうち V1・V2・V4 は反証（引き出し + 電子レンジの衝突なし率 −0.042、運群の完了 −0.062、中段引き出しは −0.062〜−0.083）、V3 は保留、下段引き出しのみ +0.229。
- **対処 C（成功・失敗の対照学習、`docs/計画_対照学習_成功失敗の増幅_C_20261015.md` §8）**: 結果トークン（success / failure）で条件付けて学習し、推論で負の誘導を掛ける。前提試験 不通過（held-out の AUROC ≈ 0.51、学習窓でも般化せず）、誘導の効果 0、ラベル付き自己 rollout を足すだけで害（衝突なし率 −0.083）。
- **読み**: 同じ方策・同じ expert から 1 周で作れるデータでは確率的な群の分岐は直らない。

詳細はレポート（`report/report_20261016.pdf`）と `docs/作業履歴_最終課題.md`（判断・試行・失敗と訂正を時系列で記録）を参照。

## 構成
| パス | 内容 |
|---|---|
| `scripts/analyze_wrist_tilt.py`, `analyze_single_tilt.py`, `analyze_contact_windows.py`, `extract_*_keyposes.py` | 人間デモの手首の傾きの解析、接触区間、キーポーズの抽出 |
| `scripts/run_drawer_hook.py`, `run_close_push.py`, `run_micro_close.py` | 3 タスクの expert（全姿勢追従の制御器 `HookRunner` を含む）と教師デモの生成 |
| `scripts/kadai_record.py` | 教師デモの HDF5 記録（agentview・wrist・状態・行動） |
| `scripts/train_lora_kadai.py` | openpi の `train_lora_pi05.py` を元にした LoRA 学習（`train_lora_pi05.py` も同梱）。v3 の起動は `pod_launch_train_v3*.sh` |
| `scripts/eval_kadai_closedloop.py`, `serve_policy_noise.py`, `pod_run_eval.sh`, `pod_run_targeted.sh`, `run_exp2.sh` | 方策サーバ + 並列クライアントの閉ループ評価、摂動試験、標的評価（同じ seed の反復） |
| `scripts/summarize_eval.py`, `compare_variants.py`, `analyze_eval_breakdown.py`, `analyze_exp2.py` | 集計・対応のある比較・失敗の分類 |
| `scripts/analyze_trace.py`, `teacher_stillness.py`, `stall_detector_offline.py`, `analyze_state_features*.py`, `entry_geometry*.py` | 診断ハーネス 1: 停留中の行動記録（trace）の解析、教師の静止区間、停留検出器、状態特徴、進入の幾何 |
| `scripts/branch_kadai.py`, `run_branches.py`, `pod_run_branches_i{3,4,5}.sh`, `analyze_branches*.py`, `pod_run_aprime.sh`, `analyze_aprime.py` | 診断ハーネス 2: 同一状態からの復元と分岐（I1〜I3）、観測・状態の入れ替え介入（I4・I5） |
| `scripts/run_expert_eval.py`, `run_wrapper_eval.py`, `analyze_expert_eval.py`, `analyze_wrapper.py`, `lanes_*.py`, `analyze_lanes.py` | expert の同条件評価（I6）、ラッパの評価、同一初期状態からの手先位置の散り（図 6） |
| `scripts/collect_rec.py`, `pod_run_rec_rollout.sh`, `pod_run_rec_branch.sh`, `analyze_rec_collect.py`, `check_dataset_v3.py`, `analyze_rec_mode_conflict.py` | 対処 R-a: 回復データの収集（分岐 → expert 引き継ぎ）、選別、データセットの検査、様式の衝突の解析 |
| `scripts/pod_run_v3_eval.sh`, `pod_run_e1_only.sh`, `pod_run_e_drawer_only.sh`, `analyze_v3_eval.py`, `analyze_v3s2000_*.py` | 対処 R-a: v3 の評価と判定、切り分け（step 2000 の評価） |
| `scripts/build_contrast_dataset.py`, `serve_policy_cfg.py`, `eval_kadai_closedloop_cfg.py`, `check_cfg_server.py`, `premise_auroc.py`, `pod_run_contrast_*.sh`, `pod_contrast_lib.sh`, `summarize_contrast_e1.py` | 対処 C: 結果条件付きデータの構築、結果条件付きの方策サーバ（負の誘導）、前提試験（AUROC）、評価 |
| `scripts/make_failure_sheets.py`, `report/make_figures.py`, `report/make_video.py`, `pod_record_video_*.sh` | 失敗例のコマ送り、レポートの図、提出動画の組み立て |
| `report/` | レポートの HTML 原稿と図、提出した PDF（`report_20261016.pdf`）、動画の構成（`video_spec.json`） |
| `results/` | 評価の生データ（`eval_runs_*/`, `v3_eval_20261016/`, `branches*_20261015/`, `expert_eval_20261016/`, `contrast_runs_20261015/` など）、集計（`*_summary_*.md`）、回復データの収集記録（`rec_collect_20261016/`）、学習曲線、キーポーズ、Pod 側のログ（`pod_logs_*/`、`pod_checks_20261016/`） |
| `figs/` | 素 / LoRA の失敗例のコマ送り画像、分岐実験の図 |
| `docs/` | 計画（最初の計画、原因追究 v1 → v2、対照学習 C）、人間デモ解析の結果、作業履歴、腕 C の実行手順、`gpt6/`（計画の批評を外部 LLM に求めた依頼文と回答） |

## 再現に必要なもの（本リポジトリに含まれないもの）
- LIBERO-plus、openpi（PyTorch 版）、LIBERO の人間デモ（libero_goal / libero_90）、π0.5-LIBERO の重み（`gs://openpi-assets/checkpoints/pi05_libero` を PyTorch に変換）。
- スクリプト・記録内の `/path/to/work/...`（共有の作業ディレクトリ）と `/path/to/kadai_work/...`（本課題の作業ディレクトリ）は環境に合わせて書き換える（環境変数 `LIBERO_ROOT` などで上書き可）。`pod_*.sh` と手順書の `<pod-id>`・`<pod-ip>`・`<port>`・`<ssh-key>` は計算機（RunPod L40S）の接続情報の置き換え。
- 学習データ（HDF5）、LoRA 重み、動画フレームは容量のため含まない。教師デモは `run_*` の expert で再生成できる（seed とパラメータは `results/gen_*.json`）。回復デモは `collect_rec.py` と `results/rec_collect_20261016/` の選別記録から再生成できる。
- 学習は L40S 1 枚で約 3.5 時間（5.2 s/step）。

## AI 支援の使用について
実装・実験・記録は Claude Code（Anthropic の AI コーディング支援）を用いて行った。原因追究の計画の批評には外部の LLM（GPT-6）を 2 回用いた（`docs/gpt6/`）。判断・方針は受講者が承認した。

## 第三者のコード
openpi（Apache-2.0）、LIBERO / LIBERO-plus、robosuite、MuJoCo を利用している。`scripts/train_lora_pi05.py` は openpi 由来のコードを含む。
