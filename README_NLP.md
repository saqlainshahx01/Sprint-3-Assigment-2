# NLP Assignment: Progressive Text Classification

Text classification on the **20 Newsgroups** dataset. Models go from a simple TF-IDF ANN, through 1D CNNs and RNN variants with n-grams and word embeddings, to a Transformer and BERT, and finally LLM zero/one/few-shot inference.

**Notebook:** `nlp_aasigment.ipynb` (GPU runtime, TensorFlow/Keras + PyTorch/Hugging Face)

---

## 1. Dataset and preparation

| Item | Value |
|---|---|
| Dataset | 20 Newsgroups (`sklearn.datasets.fetch_20newsgroups`, train + test merged) |
| Documents | 18,846 |
| Classes | 20 (alt.atheism, comp.graphics, … talk.religion.misc) |
| Split | 70 / 15 / 15, stratified, `seed=42` |
| After removing empty texts | Train 13,184 · Val 2,825 · Test 2,825 |

**Preprocessing (`clean_text`):** remove email headers (From, Subject, …), URLs and email addresses → lowercase → keep letters only → whitespace tokenization → remove NLTK stopwords.

The same split and cleaned text are used by every model, so the comparison is fair.

**Vectorization**
- **TF-IDF:** 10,000 features, unigrams + bigrams (ANN)
- **Sequences:** Keras `Tokenizer`, vocab 30,000, padded or truncated to 300 tokens (CNN, RNNs, Transformer)
- **BERT:** `bert-base-uncased` tokenizer, max length 128

**Metrics:** accuracy and weighted precision, recall and F1, plus a confusion matrix for every model. All results are collected in `nlp_assignment_results.csv`.

---

## 2. Models implemented

| # | Task | Implementation |
|---|---|---|
| 2 | **ANN baseline** | TF-IDF → Dense(256) → Dropout → Dense(128) → Dropout → softmax(20). Early stopping. |
| 3 | **1D CNN** | Embedding(128) → Conv1D(128 filters) → GlobalMaxPool → Dense(128) → softmax. Kernel sizes **3, 5, 7** compared. |
| 4 | **RNN family** | Embedding(128) → {SimpleRNN, GRU, LSTM, BiLSTM, BiGRU}(128) → Dense → softmax |
| 4 | **N-gram features** | n = 5, 10, 100. Word n-grams are built, then TF-IDF vectorized and fed to an ANN. |
| 5 | **Word2Vec** | Skip-gram, 100-d, trained on the training set. Frozen embedding layer in all 5 RNN models. |
| 5 | **FastText** | 100-d, trained on the training set. Frozen embedding layer in all 5 RNN models. |
| 6 | **Transformer** | Custom encoder: positional embedding → 1 multi-head attention block (4 heads, ff=256) → GlobalAvgPool → Dense. |
| 6 | **BERT** | `bert-base-uncased` fine-tuned, 2 epochs, AdamW lr=2e-5, batch size 16. |
| 7 | **LLM** | `google/flan-t5-base` (~250M params), zero-shot / one-shot (1 example per class) / few-shot (5 per class). Evaluated on the first 500 test documents. |

---

## 3. Results (test set)

### Best to worst, by F1

| Rank | Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 1 | **ANN + TF-IDF** | **0.879** | 0.881 | 0.879 | **0.879** |
| 2 | CNN (kernel=5) | 0.865 | 0.867 | 0.865 | 0.865 |
| 3 | CNN (kernel=7) | 0.860 | 0.865 | 0.860 | 0.861 |
| 4 | CNN (kernel=3) | 0.860 | 0.863 | 0.860 | 0.861 |
| 5 | Transformer Encoder | 0.857 | 0.862 | 0.857 | 0.857 |
| 6 | BERT fine-tuning | 0.815 | 0.821 | 0.815 | 0.811 |
| 7 | BiLSTM | 0.799 | 0.818 | 0.799 | 0.803 |
| 8 | BiGRU | 0.730 | 0.745 | 0.730 | 0.733 |
| 9 | Word2Vec + BiGRU | 0.703 | 0.716 | 0.703 | 0.694 |
| 10 | Word2Vec + BiLSTM | 0.691 | 0.699 | 0.691 | 0.682 |
| 11 | FastText + BiGRU | 0.643 | 0.642 | 0.643 | 0.631 |
| 12 | FastText + BiLSTM | 0.639 | 0.642 | 0.639 | 0.628 |
| 13 | ANN, n-gram = 5 | 0.316 | 0.707 | 0.316 | 0.377 |
| 14 | ANN, n-gram = 10 | 0.242 | 0.811 | 0.242 | 0.303 |
| 15 | FastText + GRU | 0.354 | 0.368 | 0.354 | 0.295 |
| 16 | LLM zero-shot | 0.286 | 0.360 | 0.286 | 0.248 |
| 17 | Word2Vec + RNN | 0.083 | 0.163 | 0.083 | 0.067 |
| 18 | FastText + LSTM | 0.118 | 0.223 | 0.118 | 0.061 |
| 19 | Word2Vec + LSTM | 0.121 | 0.153 | 0.121 | 0.061 |
| 20 | Word2Vec + GRU | 0.072 | 0.157 | 0.072 | 0.038 |
| 21 | FastText + RNN | 0.070 | 0.122 | 0.070 | 0.038 |
| 22 | GRU | 0.067 | 0.060 | 0.067 | 0.030 |
| 23 | SimpleRNN | 0.062 | 0.036 | 0.062 | 0.019 |
| 24 | LSTM | 0.059 | 0.009 | 0.059 | 0.013 |
| 25 | ANN, n-gram = 100 | 0.054 | 0.206 | 0.054 | 0.010 |
| 26 | LLM one-shot | 0.044 | 0.002 | 0.044 | 0.004 |
| 26 | LLM few-shot | 0.044 | 0.002 | 0.044 | 0.004 |

Random guessing on 20 balanced classes is about **5%**. SimpleRNN, GRU, LSTM, the n-gram=100 ANN and the one-shot/few-shot LLM are therefore at chance level.

### Transformer, per class (F1)
Strongest: `talk.politics.mideast` 0.95, `rec.motorcycles` 0.95, `rec.sport.baseball` 0.94, `sci.crypt` 0.93.
Weakest: `talk.religion.misc` 0.65, `comp.graphics` 0.78, `comp.sys.ibm.pc.hardware` 0.78.
Religion and politics groups overlap in vocabulary, and the comp.* groups are similar to each other.

---

## 4. Analysis

- **The simple TF-IDF ANN wins (87.9%).** 20 Newsgroups is largely keyword-driven, so bag-of-words features are very strong. Deeper models did not beat it.
- **CNN kernel size matters little.** Kernels 3, 5 and 7 all land at 86.0–86.5%, and kernel 5 is slightly best.
- **Custom Transformer beats BERT here (85.7% vs 81.5%).** BERT was fine-tuned for only 2 epochs on text truncated to 128 tokens, so it is under-trained. More epochs and a longer max length should help.
- **Bidirectional beats unidirectional by a huge margin.** BiLSTM reaches 79.9%, while plain RNN, GRU and LSTM did not learn at all (about 6%). Long (300-token) padded sequences with post-padding hurt unidirectional RNNs, because the final hidden state mostly reflects the padding.
- **Frozen Word2Vec/FastText embeddings were weaker than learned embeddings.** BiGRU: 73.0% with a learned embedding, 70.3% with Word2Vec and 64.3% with FastText. These embeddings were trained on only about 13K documents, so they are far weaker than large pretrained ones such as GoogleNews Word2Vec or the Wikipedia FastText.
- **N-gram features collapse as n grows.** Long n-grams (5, 10, 100 words) almost never repeat across documents, so the TF-IDF features become extremely sparse and accuracy falls from 31.6% (n=5) to 24.2% (n=10) and 5.4% (n=100). The high precision values for n=5 and n=10 come from very few classes being predicted.
- **The LLM scenarios mostly failed.** Zero-shot flan-t5-base gets 28.6%. One-shot and few-shot score 4.4%, which is chance level. The likely cause is the 512-token input limit: 20 or 100 example documents in the prompt are truncated, so the model never sees the target document or the answer format.

---

## 5. How to run

1. Open `nlp_aasigment.ipynb` in Google Colab or Kaggle and select a **GPU** runtime.
2. Run all cells in order. The first cell installs `nltk gensim transformers sentencepiece accelerate`, and the dataset is downloaded automatically by scikit-learn.
3. Outputs:
   - `nlp_assignment_results.csv`: all metrics
   - `tfidf_vectorizer.pkl`, `tokenizer.pkl`: saved preprocessing objects
   - confusion matrices and comparison plots, shown inline

**Requirements:** Python 3, TensorFlow ≥ 2.x, PyTorch, transformers, scikit-learn, gensim, NLTK, pandas, matplotlib, seaborn.

---

## 6. Known gaps versus the assignment, and suggested fixes

| Gap | Suggested fix |
|---|---|
| **N-gram features were not integrated into the RNNs.** They are only used with the ANN. | Feed n-gram tokens (or n-gram TF-IDF vectors) into the RNN input. |
| **Embeddings were applied only to the RNNs.** The assignment asks for ANN and CNN too. | Add embedding variants of the ANN and CNN. |
| **BERT is used only as a fine-tuned classifier.** It is not used as an embedding inside Tasks 2–4. | Extract frozen BERT embeddings and feed them to ANN, CNN and RNN. |
| **Word2Vec and FastText were trained from scratch on the training data.** They are not true pretrained embeddings. | Load pretrained vectors (e.g. `word2vec-google-news-300`, `fasttext-wiki-news-subwords-300` via `gensim.downloader`). |
| **Unidirectional RNN/GRU/LSTM do not converge.** | Use `padding="pre"`, masking (`mask_zero=True`), gradient clipping, and shorter sequences. |
| **One-shot/few-shot LLM prompts exceed the context.** | Use short example snippets (about 50 tokens each) and fewer classes per prompt. Try a larger model (flan-t5-large, Gemma 1B, Phi). |
| **Only one LLM was evaluated, on 500 test documents.** | Compare at least 2–3 models from the 100M–2B range on the full test set. |
| **`results_df` contains duplicate LLM rows.** Cells were re-run. | Reset `RESULTS = []` before re-running, or drop duplicates. |
| **Cell 24 has a stray, malformed `print(...)` block.** | Remove the extra lines so the cell prints only the Word2Vec vocabulary size. |
| **BERT was trained for only 2 epochs at max length 128.** | Train 3–4 epochs and use max length 256–512, with a raw (uncleaned) text variant. |
