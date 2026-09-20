# sdr_allan_deviation

SDRを用いて発振器のアラン偏差を評価する。
電波干渉系の周波数標準を評価する目的で作成した。

アラン偏差計算の核心部分はallan_toolsに依存しており、このプロジェクトはデータの前処理と可視化を行う。

## ハードウェア構成

アラン偏差の評価は通常ミキサーを用いて2つの発振器の周波数差を測定することで行われる。
ここではSDRの片方の発信機をSDRのクロックとして使用することでSDRに搭載されたミキサーを利用する。

```mermaid
flowchart LR
    A[発振器1] --> B[RF信号源] --> E[ATT] -->|RF入力| C[SDR]
    D[発振器2] -->|クロック| C
```

発振器1と2で同じものを二つもしくは評価したい発振器より安定なものを使用する。1つの発信機を接続した場合はクロックの安定度は評価できないがSDRの位相雑音等を評価できる。

## 使用方法

1. データをwav形式で保存する。

2. ```src/config.py```に適切なパラメータを設定する。

- SOURCE_RATE_HZ: SDRのサンプリングレート
- TARGET_RATE_HZ: データを保存するレート
- REFERENCE_HZ: 周波数標準の周波数
- BIN_SECONDS: データをビンにする時間
- SIGNAL_PLOT_SECONDS: 信号をプロットする時間
- MAX_TAU_SECONDS: アラン偏差を評価する最大時間
- TAU_SECONDS: アラン偏差を評価する時間
- OBSERVATION_FREQUENCY: 周波数標準の周波数
- LOWPASS_HZ: 低域フィルタのカットオフ周波数
- HIGHPASS_HZ: 高域フィルタのカットオフ周波数
- SKIP_SECONDS: データの先頭をスキップする時間

3. 以下のコマンドを実行する。

```bash
uv run main path/to/input.wav
```

4. 結果が```outputs/```に保存される。

5. 結果が妥当かどうか以下の項目を確認する。

- signal_iq.pngを確認して下の段が正弦波に見えるか？
- power_spectrum.pngを確認して鋭いピークが立っているか？
- power_spectrum.pngを確認してピークがHPFとLPFの間にあるか？
- instantaneous_frequency.pngを確認してランダムっぽいノイズに見えるか？
