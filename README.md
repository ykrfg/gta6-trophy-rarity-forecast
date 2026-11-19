# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ykrfg/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ykrfg/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast how many GTA VI players earn each trophy in week one: a Python program (standard library only) that predicts the five quantiles (5%, 25%, 50%, 75%, 95%) of every trophy's earn rate on Exophase at the target time, from the recorded history of the trophy page.

## 概要

GTA VI の PS5 トロフィー55種について、目標時刻における Exophase の獲得率の5分位数を予測するプログラムです。獲得者数とトラッキング対象プレイヤー数を別々にモデル化し（新規プレイヤーの流入による「希釈」が獲得率を下げる効果を構造的に取り込む）、ロジット尺度で当てはめ、55トロフィー間でプール・縮小し、履歴内のリフレッシュ間のロジット変化から広がりを較正します。Python 標準ライブラリのみ、オフラインで動作します。

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
