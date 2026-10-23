# 進入の幾何 G1〜G3（計画 v2 §9。既存 trace のみ、CPU）

スクリプト: `kadai/scripts/entry_geometry.py`（座標変換）→ `entry_geometry_stats.py`（統計・図）。表: `entry_geometry_20261015.csv`（81 窓 × 3 時点 × {x,y,z,tilt,yaw}、k6 は扉の縁・法線）。数値の全量: `entry_geometry_stats_20261015.json`。図: `kadai/figs/entry_geometry_20261015.png`。

## 座標系と確認
- 固定具座標系 = 基準 body（drawer: wooden_cabinet_1_main、k6: microwave_1_main、k4: white_cabinet_1_main）のローカル系。手先位置 = `R_f^T (ee − p_f)`、姿勢 = `R_f^T R_ee`。tilt = 手先 z 軸の鉛直下向きからの角、yaw = 固定具系での手先 x 軸の方位。
- 固定具の姿勢は `init_episode`（npz の `ref_pos0/ref_quat0` を state に入れて呼ぶ）で env に実際に作らせ、各時点で `sim_state[t+1]`（ee[t] と同期）を `set_state_from_flattened` → `forward` して `body_xpos/xmat` を読んだ（sim 優先）。
- 一致確認: **全 81 窓**で、独立に書いた再現式（`pos0 + [d0,d1,0]`、`yaw ⊗ quat0`、prng = (seed, 20261013)）と model の `body_pos/body_quat` の最大差 0（≤1e-16）、`body_xpos` = `body_pos`（親は world）。
- 窓の開始 s は `aprime_windows_20261014.csv` の `win_start`。時点は s−10 / s / s+20（ep の範囲にクランプ）。n: drawer 失敗 17・偽陽性 17、k6 失敗 35・偽陽性 7、k4 失敗 4・偽陽性 1（計 81）。
- k6 の扉: 扉パネル geom（microwave_1_microdoorroot の box geom）の 8 隅を sim で世界座標へ。縁 = 隅の z 最小。法線 = パネルの薄い軸、取っ手（capsule）側を手前（外向き）とした。**代理は使っていない。**

## 無効にした列（★trace の欠陥）
`tgt_pos` は `target_info` が `np.asarray(sim.data.body_xpos[bid])`（view）を返し、全 step が最終状態の値に化けている（drawer 1014 r0: q は 0→−0.143 なのに `tgt_pos` は全 step で同一の y=−0.141）。対象相対の手先位置（h*列）は成功 ep で「開いた後の取っ手」を引くため AUROC 1.0 になる**人工物**。CSV から削除した。`tgt_dist`・`tgt_uv` は値が毎 step 計算されるので無傷（要確認なら copy 化して再収集）。

## 判定
| # | 結果 | 線での判定 |
|---|---|---|
| G1 drawer 手先位置、最近接重心 LOO AUROC | s−10: **0.796**（中央値差 x +12.9 / y +35.9 / z −8.1 mm）、s: **0.585**（−12.3 / −0.2 / −0.4）、s+20: **0.471**（−13.4 / +0.8 / −3.1） | s−10 は線（>0.8）の直下で保留、s と s+20 は 0.6 未満で反証側。窓の時点では位置の分離なし |
| G2 drawer 失敗窓のずれ vs 摂動 | 失敗 17 本は全て strong（rn 0.2・cs 0.06・cy 10）で level は定数 → **Spearman 定義不能**。実際のヨー角 |yaw| とは −0.51 / −0.56 / −0.39、位置ずれ |shift| とは −0.06 / −0.01 / +0.10（s−10/s/s+20） | 支持（>0.5）の符号が逆。反証側（<0.2 は shift のみ該当）。保留〜反証 |
| G3 k6 手先が縁より下の割合 | 失敗・偽陽性とも **0%**（縁からの手先 z 中央値 +148〜187 mm、全窓で縁より上） | 差 0 → 反証（<+0.1） |
| G3 k6 扉面の裏側（法線の符号 <0）の割合 | 失敗 / 偽陽性: s−10: 0.11 / 0.43、s: 0.11 / 0.29、s+20: 0.26 / 0.29 | 差 −0.32 / −0.18 / −0.03（失敗の方が低い）→ 反証 |
| k4（記述のみ、1 失敗 窓×4） | 手先 (x,y,z) mm は失敗 4・偽陽性 1 とも x 9〜80・y −110〜−235・z 32〜88 の範囲に散る。比較不能 | — |

## 追加の事実（予測にない読み。解釈は付けない）
- k6 s+20: 手首の傾き 失敗 51° vs 偽陽性 27°（AUROC 0.82、s−10/s では 0.38/0.40）。扉面からの法線距離の中央値 19 vs 45 mm、手先 z（縁から）187 vs 160 mm。
- drawer を strong のみ（失敗 17 / 偽陽性 13）にしても AUROC は 0.805 / 0.643 / 0.588（s−10 / s / s+20）で同じ傾向。level の交絡は小さい。
- drawer の手先姿勢（tilt 60° 前後・yaw −90° 前後）は失敗と偽陽性で差なし（AUROC 0.33〜0.61）。
- 留保: 同一 seed の複数 rep が独立でない（特に drawer 偽陽性 13 本は少数 seed）。k6 偽陽性は 7 本。n が小さく AUROC の区間は広い（±0.1 以上）。窓の開始は進捗が止まった時点で、ep ごとに段階が違う。

## 総括（G1 反証〜保留、G2 反証、G3 反証）
計画 §9 の読み: 「G1 反証 → 相対姿勢は同じで結果だけ違う = 接触の確率的な側面、対処は再試行（退避→再接近）」側。ただし drawer の s−10 の y 方向 +36 mm と k6 s+20 の傾き差は、窓に入る前〜入った後の手の動きの差を示す未説明の信号として残る。
