新しいフォルダに、約0.5BパラメータのDecoder-only Transformerをスクラッチ事前学習するプロジェクトを作成してください。

モデル構造はLlama系を参考にし、RMSNorm、RoPE、Causal Self-Attention、SwiGLU FFNを使用します。学習済みモデルの重みは使わず、すべてランダム初期化してください。

今後、以下の3種類を比較する予定です。

1. 全層が独立した通常モデル
2. ブロック内でAttentionとFFNの両方を共有するモデル
3. ブロック内でAttentionのみ共有し、FFNは層ごとに独立させるモデル

今回は1の通常モデルだけを実装してください。ただし、後からAttentionとFFNを別々に共有できるよう、Attention、FFN、Transformer Block、層の割り当てを独立した構成にしてください。

TokenizerはSentencePiece BPEを使って独自に学習し、語彙数はまず16,000を候補としてください。

以下を実装してください。

* Tokenizerの学習
* テキストデータのトークン化と固定長チャンク化
* 約0.5Bモデルの設定
* 総パラメータ数と内訳の表示
* Causal Language Modelingによる学習
* Cross Entropy Lossとtoken-weighted Perplexityの評価
* bf16、Gradient Accumulation、Gradient Clipping
* Checkpoint保存と再開
* 学習token数の記録
* 小型モデルを使ったsmoke test
* 小規模データに過学習できることの確認

本学習の目標token数はパラメータ数の約20倍とし、0.5Bモデルでは約10B tokensを想定します。ただし、今回は本学習を実行せず、小型設定で全処理が正しく動くところまで完成させてください。

最初にプロジェクト構成と実装計画を提示し、その後に実装してください。完了後、0.5B設定の正確なパラメータ数、実行コマンド、smoke test結果、今後共有モデルを追加する際の変更箇所を報告してください。
