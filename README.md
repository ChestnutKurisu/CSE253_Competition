## [Link to Competition Guidelines](https://docs.google.com/document/d/1crq3eqtnBHs5UyIhst_C0u6e65uQ4bj2miv4SPjycXo/export?format=pdf)

## Final Results (ranks based on the Gradescope leaderboard; 1,272 students participated in the competition)

*   Task 1: Composer Classification (Public): 0.9949748743718593 **(Rank 1)**
*   Task 1: Composer Classification (Private): 0.9789473684210527 **(Rank 1)**
*   Task 2: Next Sequence Prediction (Public): 0.9986936642717178 **(Rank 1)**
*   Task 2: Next Sequence Prediction (Private): 0.99545159194282 **(Rank 1)**
*   Task 3: Music Tagging (Public): 0.5270201135580944 **(Rank 1)**
*   Task 3: Music Tagging (Private): 0.4630985627065306 **(Rank 27)** - *More on this later, a classic case of knowingly overfitting the public leaderboard!*

---

## Task 1: Composer Classification (Symbolic, Multi-Class)

### 1.1. Background and Motivation

Key strategies that I employed in my final solution:
1.  **Extensive Feature Engineering:** Moving beyond simple statistics to a rich set of symbolic features,
2.  **Addition of an Open-Source Dataset:** Incorporating the [ASAP (Aligned Scores and Performances) dataset](https://github.com/fosfrancesco/asap-dataset) to enlarge the training pool,
3.  **Ensemble Modeling:** Combining predictions from multiple strong tabular learners.

Initially, I delved into sophisticated transformer models for symbolic music like [MusicBERT](https://github.com/malcolmsailor/musicbert_hf), [MidiBERT](https://github.com/wazenmai/MIDI-BERT), [PianoBart](https://github.com/RS2002/PianoBart), and [Adversarial MidiBERT](https://github.com/RS2002/Adversarial-MidiBERT). While these models report impressive results in their respective papers (e.g., Adversarial-MidiBERT claims >97% accuracy with pre-training on datasets like ASAP), I found it challenging to replicate these SOTA accuracies by fine-tuning open-source checkpoints on the provided assignment data. For instance, fine-tuning MidiBERT on the assignment's training data took about 3 hours on an RTX 4080 GPU and yielded only ~66% accuracy on the public leaderboard. This led me to pivot towards a feature-rich tabular approach.

### 1.2. Chronology of Experiments & Methodology

1.  **Initial Feature Engineering & XGBoost:**
    *   Started with ~15 symbolic features (pitch range, tempo-based features, polyphony, basic key detection).
    *   An XGBoost model on these features quickly jumped to ~0.63 public accuracy.

2.  **Massive Feature Set (364 Dimensions):**
    *   Expanded to a 364-dimensional feature vector, incorporating advanced tonal, harmonic, chord-based, and rhythmic metrics. Some of the features include:
        *   Detailed pitch, duration, and velocity statistics (mean, std, min, max, median, kurtosis, skewness, autocorrelation).
        *   Pitch-class histograms, entropy, and PCA-based representations.
        *   Polyphony, chord rate, tempo statistics, time signature features.
        *   Key detection (correlation with Krumhansl-Kessler profiles).
        *   Melodic interval statistics, upward/downward motion ratios.
        *   Rhythmic features like nPVI, offbeat ratio, swing ratio, downbeat emphasis.
        *   Naive chord detection, chord progression entropy, borrowed chord frequency.
        *   REMI token-based features (e.g., bar density) using `miditok`.
        *   Markov chain probabilities for intervals and chords.
        *   Features inspired by Hindustani classical music (Raga profile correlations).
        *   Tonal tension metrics, voice-leading smoothness.
    *   This feature set alone (with XGBoost) pushed public leaderboard accuracy to ~0.72.

3.  **Incorporating the ASAP Dataset:**
    *   Following clarifications on Piazza ([@166](https://piazza.com/class/m8rskujtdvsgy/post/166), [@178](https://piazza.com/class/m8rskujtdvsgy/post/178)) about using pre-trained models even though they may have seen some training data during pre-training, I began exploring open-source datasets, and integrated several open-source datasets (Maestro v3, Giant-MIDI, and the [ASAP dataset](https://github.com/fosfrancesco/asap-dataset).)
    *   Filtered to retain renditions for only the eight target composers using each dataset's metadata.
    *   To handle long pieces and create more training instances, I split MIDI files exceeding 65 beats (approximate mean length of the assignment's training data) into multiple segments, and saved each segment as a new temporary MIDI file.
    *   This significantly increased the training data size and helped combat overfitting.

4.  **Multi-Model Ensemble:**
    *   Trained three classifiers on the combined "Student + ASAP" dataset:
        1.  [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier)
        2.  [LightGBM](https://github.com/microsoft/LightGBM)
        3.  [TabPFN](https://github.com/PriorLabs/TabPFN) ([arXiv:2207.01848](https://arxiv.org/pdf/2207.01848.pdf))
    *   The final prediction was a weighted average of the probability distributions from these models. Weights of `xgb_weight=0.3`, `tpf_weight=0.6`, and `lgb_weight=0.1` worked best; I did not hyperopt any of the classifiers' hyperparameters or the weights assigned to each as this configuration was already producing a very high accuracy on the public leaderboard.

### 1.3. Implementation Details

*   **Symbolic Feature Extraction:** Used [miditoolkit](https://github.com/YatingMusic/miditoolkit) and [pretty_midi](https://github.com/craffel/pretty-midi) for parsing. Features were cached to speed up re-runs.
*   **Training:**
    *   Final iterations were trained on nearly all available data.
    *   XGBoost: `tree_method="hist"`, `max_depth=10`, `eta=0.15`, `n_estimators=1320`.
    *   LightGBM: `device_type="gpu"`, `n_estimators=1320`.
    *   TabPFN: `device="cuda"`, `ignore_pretraining_limits=True`.

*   **Feature Importance (XGBoost Example):**
    The top 15 features by XGBoost gain (trained on the combined dataset) highlight what the model found distinctive:

    | Rank | Feature Name                 | Short Description                                                                | XGBoost Gain |
    |-----:|:-----------------------------|:---------------------------------------------------------------------------------|-------------:|
    |  1.  | `global_pitch_range_density` | Ratio of pitch range (max - min) to the full 127-step MIDI range                 |   26.916185  |
    |  2.  | `pitch_range`                | Difference between maximum and minimum pitch                                     |   17.187300  |
    |  3.  | `turn_ornament_density`      | Density of "turn" ornaments (approx. # patterns / total bars)                    |   11.673298  |
    |  4.  | `pitch_max`                  | Maximum pitch value (0–127)                                                      |    7.823633  |
    |  5.  | `direction_change_mad`       | Median absolute deviation of consecutive interval directions                     |    4.803987  |
    |  6.  | `pitch_min`                  | Minimum pitch value (0–127)                                                      |    4.715915  |
    |  7.  | `dur_bin_0`                  | Fraction of notes with duration < 0.125 beats                                    |    4.528508  |
    |  8.  | `bar_pitch_mean_std`         | Standard deviation of bar-level average pitch                                    |    4.043872  |
    |  9.  | `pitch_mean`                 | Mean pitch value (0–127)                                                         |    3.918227  |
    | 10.  | `high_reg_density`           | Ratio of notes with pitch >= 84 (upper registers)                                |    3.886408  |
    | 11.  | `seventh_chord_count`        | Count of recognized seventh chords                                               |    3.733670  |
    | 12.  | `pitch_95_5_range`           | Difference between 95th and 5th percentile pitch                                 |    3.502671  |
    | 13.  | `ic_dist_3`                  | Normalized measure of 3-semitone melodic intervals                               |    3.440810  |
    | 14.  | `vel_min`                    | Minimum velocity (0–127)                                                         |    3.432241  |
    | 15.  | `frac_chromatic_moves`       | Proportion of intervals that are 1 semitone                                      |    3.274476  |

    This suggests that overall pitch usage, specific ornamentation styles, and harmonic complexity were key differentiators.

### 1.4. References
*   NLP-based Music Processing for Composer Classification: Somrudee Deepaisarn et al., _Scientific Reports_, 2023. [PubMedCentral PMC10425398](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10425398/)
*   Composer Style Classification of Piano Sheet-Music Images Using Language-Model Pretraining: TJ Tsai, Kevin Ji, ISMIR 2020. [arXiv:2007.14587](https://arxiv.org/abs/2007.14587)
*   Large-Scale MIDI-Based Composer Classification: Qiuqiang Kong, Keunwoo Choi, Yuxuan Wang, 2020. [arXiv:2010.14805](https://arxiv.org/abs/2010.14805)
*   [ASAP Dataset](https://github.com/fosfrancesco/asap-dataset)
*   Other datasets explored: [MAESTRO](https://magenta.tensorflow.org/datasets/maestro), [GiantMIDI-Piano](https://huggingface.co/datasets/roszcz/giant-midi-base-v2), [Metacreation.net](https://www.metacreation.net/dataset)

---

## Task 2: Next Sequence Prediction (Symbolic, Binary)

### 2.1. Overview of Approaches

I primarily focused on a [**Siamese Byte-CNN Classifier**](https://en.wikipedia.org/wiki/Siamese_neural_network), which ultimately gave the best results. An alternative feature-based XGBoost/LightGBM classifier using the same features that I'd implemented for task 1 also performed really well (0.9960 vs 0.9987 accuracy on the public leaderboard) but was slightly edged out.

**1. Siamese Byte-CNN Classifier (Final Approach):**
This method processes the raw byte content of the MIDI files.
*   **Input:**
    *   For bar A (predecessor): The last `SEG_LEN` bytes (tail).
    *   For bar B (successor): The first `SEG_LEN` bytes (head).
    *   I chose `SEG_LEN = 4096` bytes. This was based on an analysis of MIDI file sizes in the dataset: min=86B, median≈549B, 99th percentile≈1702B, max=16009B. 4096 bytes covers >99% of bars while being GPU-friendly. Shorter files are padded with a special token (256); longer ones are truncated.
*   **Architecture:**
    1.  **ByteEncoder:**
        *   `nn.Embedding`: Maps bytes (0-255) and a pad token (256) to `EMBED_DIM=256` vectors.
        *   `nn.Conv1d`: A bank of 1D CNNs (kernel sizes 3, 5, 7; 128 channels each) processes the embedded sequence.
        *   `nn.AdaptiveMaxPool1d`: Pools features from each CNN.
        *   Concatenated CNN outputs form a 384-D vector (3 * 128).
    2.  **PairClassifier (Siamese setup):**
        *   Encodes bar A and bar B using the *same* ByteEncoder instance.
        *   Combines these representations: `concat(enc(A), enc(B), |enc(A)−enc(B)|, enc(A)*enc(B))`.
        *   This combined vector is fed through a small feed-forward network (Linear $$\to$$ ReLU $$\to$$ Dropout $$\to$$ Linear) to produce a single logit.
*   **One-Neighbor Constraint (Critical Post-processing):**
    *   After predicting probabilities for all test pairs, a greedy selection process is applied:
        *   Sort pairs by descending probability.
        *   Iterate through sorted pairs. Mark a pair (A, B) as "True" *only if* bar A has not yet been used as a tail and bar B has not yet been used as a head in a previously accepted "True" edge.
    *   This step enforces that each bar segment connects to at most one predecessor and one successor, mimicking real musical structure. This constraint dramatically boosted public leaderboard accuracy from ~0.975 to **0.9987**.

### 2.2. Implementation Details
*   **Data Loading:** Raw bytes are read, padded/truncated to `SEG_LEN`, and cached as NumPy arrays.
*   **Training (Siamese CNN):**
    *   `nn.BCEWithLogitsLoss`.
    *   AdamW optimizer (`lr=3e-4`).
    *   Trained for `EPOCHS=10` with `BATCH_SIZE=256`.
    *   Monitored training accuracy and AUC.

---

## Task 3: Music Tagging (Continuous, Multi-Label, Multi-Class)

### 3.1. Overview and Motivation
My strategy was to leverage SOTA pre-trained audio tagging models and fine-tune them on the assignment data. After many experiments, I focused on the **M2D (Masked Modeling Duo)** model, combined with a phased fine-tuning approach.

### 3.2. Chronology of Experiments & Methodology

1.  **Initial Trials (Hugging Face AST):**
    *   Started with [Audio Spectrogram Transformer (AST) models from MIT](https://huggingface.co/MIT) (e.g., [`ast-finetuned-audioset-10-10-0.4593`](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593), [`killynguyen/ast-finetuned-audioset-10-10-0.4593-finetuned-gtzan`](https://huggingface.co/killynguyen/ast-finetuned-audioset-10-10-0.4593-finetuned-gtzan)).
    *   Achieved ~0.45-0.46 mAP on public, a good improvement but I aimed higher.
    *   Once I'd moved on to M2D, I shared the AST approach here in @340, but it apparently created a rift between students with and without GPU access.

2.  **Survey of SOTA Music Tagging Models:**
    *   Explored models from [Papers with Code](https://paperswithcode.com/sota/music-tagging-on-magnatagatune) like: [PaSST](https://github.com/kkoutini/passt), [CAV-MAE](https://github.com/yuangongnd/cav-mae), [MATPAC](https://github.com/aurianworld/matpac), [EfficientAT](https://github.com/fschmid56/EfficientAT), [M2D](https://github.com/nttcslab/m2d)
    *   Most single fine-tuned models hovered around 0.42-0.46 mAP. I attempted to ensemble the above with AST and reached 0.478 public accuracy.

3.  **M2D as the Core Model:**
    *   I settled on [M2D (Masked Modeling Duo)](https://github.com/nttcslab/m2d) due to its strong AudioSet pre-training and flexible fine-tuning.
    *   Used the "M2D-CLAP ViT-Base" encoder with weights from [`m2d_clap_vit_base-80x1001p16x16-240128_AS-FT_enconly`](https://github.com/nttcslab/m2d/releases/download/v0.3.0/m2d_clap_vit_base-80x1001p16x16-240128_AS-FT_enconly.zip).
    *   Added a custom classification head (LayerNorm + Linear) for the 10 tags.

4.  **Training Strategy for M2D:**
    *   **Input:** Audio clips loaded with `librosa` at 16kHz, converted to log-Mel spectrograms using [`nnAudio`](https://github.com/KinWaiCheuk/nnAudio).
    *   **Phased Fine-Tuning:** This was crucial to prevent catastrophic forgetting and adapt M2D effectively.
        1.  **Epochs 1 (Head Only):** Freeze M2D backbone, train only the new classification head (LR `1e-3`).
        2.  **Epochs 2 (Partial Backbone):** Unfreeze the last 4 Transformer blocks of M2D, train with a smaller LR (`5e-5`) and Cosine Annealing.
        3.  **Epochs 1 (Full Backbone):** Unfreeze the entire M2D backbone, train with an even smaller LR (`1e-5`).
    *   **Loss Function:** `nn.BCEWithLogitsLoss` with `pos_weight` to handle class imbalance, calculated from tag frequencies in the training set.
    *   **Optimizer:** AdamW with weight decay (`wd=0.18`).

5.  **Threshold Selection:**
    *   For multi-label classification, per-tag thresholds are vital.
    *   After training, I predicted on a local validation split. For each tag, I computed the precision-recall curve and selected the threshold that maximized F1-score. These thresholds generally ranged from 0.4 to 0.9.

6.  **Experiments with External Data (and why they didn't make it to the final cut for this task):**
    *   **MagnaTagATune:** I experimented with integrating the [MagnaTagATune dataset](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset), processing clips into 10s WAVs and even trying source separation with [Demucs v4](https://github.com/facebookresearch/demucs) to remove vocals, strip higher frequencies, and mimic the training/test audio files format. This didn't consistently boost mAP on *this specific assignment's* (synthetically generated) dataset.
    *   **MusicGen:** Tried extending clips using [MusicGen-Large](https://huggingface.co/facebook/musicgen-large). This was very time-consuming and didn't yield measurable mAP gains.
    *   I processed all the training and test audio files using [AudioSR: Versatile Audio Super-resolution at Scale](https://github.com/haoheliu/versatile_audio_super_resolution#audiosr-versatile-audio-super-resolution-at-scale) to address the heavy processing / stripping of higher frequencies performed on the training data, but ultimately this didn't help with mAP.
    *   The professor mentioned in class that the audio files for Task 3 were synthetically generated. This insight made me hesitant to rely too heavily on real-world datasets like MagnaTagATune due to potential distribution shift. I discovered this paper: [Generating Symbolic Music from Natural Language Prompts using an LLM-Enhanced Dataset](https://arxiv.org/abs/2410.02084) and its accompanying GitHub repo ([MetaScore](https://github.com/wx83/MetaScore_Official/tree/main)), considered the possibility that the assignment's dataset had been created using a similar process outlined in the paper, and thought about scraping audio files from MuseScore to gather additional data for training M2D but discarded this idea due to the massive time commitment involved.

### 3.3. Results & The Overfitting Story
*   **Public Leaderboard:** The fine-tuned M2D model with the above strategy achieved **0.5270** mAP.
*   **Private Leaderboard (Rank 27 - 0.46309):** This is where things got interesting. My high public score was likely a result of aggressively tuning based on public leaderboard feedback instead of conducting rigorous cross-validation, and also because in the end, I ranked some of my solutions performing well on the public leaderboard and took an intersection of the predictions on some genres like rock. My final submission for the public board probably reflected these more aggressive (and ultimately overfit) tactics.

### 3.4. References
*   **M2D (Masked Modeling Duo):** [GitHub](https://github.com/nttcslab/m2d) | [D. Niizumi et al., “Masked Modeling Duo: Towards a Universal Audio Pre-training Framework,” *IEEE/ACM Trans. Audio, Speech, Language Processing*, 2024](https://arxiv.org/abs/2404.06095).
*   **Audio Spectrogram Transformer (AST):** [Hugging Face](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)
*   **nnAudio:** [GitHub](https://github.com/KinWaiCheuk/nnAudio)
*   Other models explored: [PaSST](https://github.com/kkoutini/passt), [CAV-MAE](https://github.com/yuangongnd/cav-mae), [MATPAC](https://github.com/aurianworld/matpac), [EfficientAT](https://github.com/fschmid56/EfficientAT)
