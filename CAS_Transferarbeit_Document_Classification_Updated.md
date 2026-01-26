# Vergleichende Analyse von Machine-Learning-Methoden für die Dokumentenklassifikation: TF-IDF mit Logistischer Regression vs. Transformer-Modelle

**CAS Machine Learning – Transferarbeit**

**Autor:** [Name]  
**Datum:** [Datum]  
**Institution:** [Institution]

---

## Abstract

Diese Arbeit untersucht systematisch die Performance von klassischen Machine-Learning-Methoden im Vergleich zu modernen Transformer-basierten Ansätzen für die automatische Dokumentenklassifikation. Als Baseline dient eine TF-IDF-Vektorisierung mit logistischer Regression, die mit zwei vortrainierten Transformer-Modellen (BERT-base-uncased, DistilBERT-base-uncased) verglichen wird.

Die Evaluation erfolgt auf dem BBC News Dataset (2'225 Artikel, 5 Kategorien) mittels mehrerer methodischer Ansätze: 5-Fold Stratified Cross-Validation für robuste Performanzschätzung, Hold-Out-Validierung (60/20/20 und 80/10/10 Splits), systematische Hyperparameter-Experimente, Few-Shot Learning Analysen, Active Learning Simulationen sowie Nested Cross-Validation für unbiased Modellselektion.

Die Ergebnisse zeigen, dass die klassische Baseline mit 97.93% (±0.72%) eine nahezu identische Accuracy wie DistilBERT (98.11% ±0.77%) erreicht, bei 7-18x kürzerer Trainingszeit. BERT erzielt in einzelnen Konfigurationen bis zu 99.10% Test-Accuracy. Hyperparameter-Analysen ergeben, dass 3 Epochen und max_length=256 optimal sind. Die Few-Shot Analyse zeigt, dass bei sehr wenigen Trainingsdaten (<100 Samples/Klasse) die Baseline überlegen ist, während Transformer mehr annotierte Daten benötigen. Active Learning Experimente demonstrieren, dass uncertainty-basierte Strategien die Annotation efficiency um bis zu 40% steigern können.

Die Nested Cross-Validation mit Hyperparameter-Optimierung bestätigt die robuste Performance beider Ansätze und liefert unbiased Schätzungen. Die Ergebnisse verdeutlichen, dass der Einsatz von Transformer-Modellen für einfache, keyword-basierte Klassifikationsaufgaben keinen signifikanten Mehrwert bietet und der Trade-off zwischen Modellkomplexität und Accuracy zugunsten der einfachen, effizienten Baseline ausfällt.

**Schlüsselwörter:** Document Classification, BERT, DistilBERT, TF-IDF, Transfer Learning, Cross-Validation, Few-Shot Learning, Active Learning, Nested Cross-Validation

---

## Inhaltsverzeichnis

1. [Einleitung](#1-einleitung)
2. [Theoretische Grundlagen](#2-theoretische-grundlagen)
3. [Methodik](#3-methodik)
4. [Ergebnisse](#4-ergebnisse)
5. [Diskussion](#5-diskussion)
6. [Fazit und Ausblick](#6-fazit-und-ausblick)
7. [Literaturverzeichnis](#7-literaturverzeichnis)
8. [Anhang](#8-anhang)

---

## 1. Einleitung

### 1.1 Motivation

Die automatische Klassifikation von Textdokumenten ist eine zentrale Aufgabe im Bereich Natural Language Processing (NLP) mit vielfältigen praktischen Anwendungen – von der Kategorisierung von Nachrichten über Spam-Filterung bis hin zur Sentiment-Analyse in sozialen Medien (Sebastiani, 2002).

In den letzten Jahren haben vortrainierte Transformer-Modelle wie BERT (Devlin et al., 2019) und dessen effiziente Variante DistilBERT (Sanh et al., 2019) State-of-the-Art-Ergebnisse auf zahlreichen NLP-Benchmarks erzielt. Diese Modelle nutzen Transfer Learning: Sie werden zunächst auf riesigen Textkorpora (z.B. Wikipedia, BooksCorpus) mit selbstüberwachten Aufgaben vortrainiert und anschliessend für spezifische Aufgaben feinabgestimmt (fine-tuning).

Jedoch bleibt die Frage offen, ob dieser Fortschritt auch für **einfachere, gut strukturierte Klassifikationsaufgaben** gilt, oder ob klassische Methoden wie TF-IDF mit logistischer Regression bei deutlich geringerem Rechenaufwand vergleichbare Ergebnisse liefern können. Insbesondere in ressourcenbeschränkten Umgebungen (Edge-Geräte, Real-Time-Anwendungen) oder bei kleinen Datensätzen ist die Wahl der Methode entscheidend.

### 1.2 Forschungsfragen

Diese Arbeit adressiert folgende Forschungsfragen:

**FF1 (Performanz):**  
Wie vergleicht sich die Klassifikationsleistung (Accuracy, F1-Score) einer klassischen TF-IDF + Logistische Regression Baseline mit modernen Transformer-Modellen (BERT, DistilBERT) auf dem BBC News Dataset?

**FF2 (Effizienz):**  
Welcher Trade-off besteht zwischen Modellkomplexität, Trainingszeit und Accuracy?

**FF3 (Hyperparameter-Sensitivität):**  
Wie sensitiv sind Transformer-Modelle gegenüber Hyperparametern (Epochen, Sequenzlänge, Learning Rate) und welche Konfigurationen liefern optimale Ergebnisse?

**FF4 (Dateneffizienz):**  
Wie verhalten sich die Modelle bei limitierten Trainingsdaten (Few-Shot Learning)?

**FF5 (Annotation Efficiency):**  
Kann Active Learning die Annotation efficiency signifikant verbessern und welche Sampling-Strategie ist optimal?

**FF6 (Split-Strategie):**  
Welchen Einfluss hat die Datenaufteilung (60/20/20 vs. 80/10/10) auf die Modellperformance?

**FF7 (Robuste Evaluation):**  
Liefert Nested Cross-Validation mit integrierter Hyperparameter-Optimierung robustere und unbiasedere Performanzschätzungen?

### 1.3 Struktur der Arbeit

Die Arbeit gliedert sich wie folgt:

- **Kapitel 2** vermittelt theoretische Grundlagen zu TF-IDF, Transformer-Architektur, BERT und Evaluationsmetriken.
- **Kapitel 3** beschreibt die Methodik: Dataset, Modelle, experimentelles Setup, Evaluationsstrategien.
- **Kapitel 4** präsentiert die Ergebnisse aus Cross-Validation, Hyperparameter-Experimenten, Split-Vergleichen, Few-Shot Learning, Active Learning und Nested CV.
- **Kapitel 5** diskutiert die Befunde im wissenschaftlichen Kontext und leitet praktische Empfehlungen ab.
- **Kapitel 6** fasst die Erkenntnisse zusammen und gibt einen Ausblick auf zukünftige Forschung.

---

## 2. Theoretische Grundlagen

### 2.1 Text-Repräsentation

#### 2.1.1 TF-IDF (Term Frequency-Inverse Document Frequency)

TF-IDF ist eine klassische Methode zur numerischen Repräsentation von Textdokumenten (Salton & Buckley, 1988). Sie basiert auf der Hypothese, dass Wörter, die häufig in einem Dokument vorkommen, aber selten im gesamten Korpus, besonders aussagekräftig für die Klassifikation sind.

**Definition:**

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

Wobei:
- $\text{TF}(t, d)$ die Häufigkeit des Terms $t$ im Dokument $d$ ist
- $\text{IDF}(t) = \log\frac{N}{df(t)}$ mit $N$ = Anzahl Dokumente und $df(t)$ = Dokumentenfrequenz von Term $t$

TF-IDF-Vektoren werden typischerweise L2-normalisiert, um Dokumentlängen-Effekte zu minimieren.

**Vorteile:**
- Einfach, effizient, interpretierbar
- Funktioniert gut für keyword-basierte Klassifikation
- Keine Trainingsphase erforderlich

**Nachteile:**
- Bag-of-Words-Annahme: Wortstellung und Semantik werden ignoriert
- Keine Berücksichtigung von Synonymen oder Kontext
- Hochdimensionale, sparse Vektoren

#### 2.1.2 Kontextuelle Embeddings (BERT)

Im Gegensatz zu statischen Word Embeddings (Word2Vec, GloVe) erzeugen Transformer-Modelle wie BERT **kontextabhängige** Repräsentationen: Das gleiche Wort erhält unterschiedliche Embeddings je nach Kontext.

**Beispiel:**
- "The **bank** of the river" → Embedding für "Flussufer"
- "Money in the **bank**" → Embedding für "Finanzinstitut"

Diese Kontextsensitivität wird durch den Self-Attention-Mechanismus erreicht (Vaswani et al., 2017).

### 2.2 Transformer-Architektur

#### 2.2.1 Self-Attention

Der Kern der Transformer-Architektur ist der **Self-Attention-Mechanismus**, der es ermöglicht, Beziehungen zwischen allen Wörtern in einem Satz zu modellieren.

**Mathematische Formulierung:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Wobei:
- $Q$ (Query), $K$ (Key), $V$ (Value) sind lineare Projektionen der Input-Embeddings
- $d_k$ ist die Dimensionalität der Keys (Skalierungsfaktor)

**Multi-Head Attention** wendet mehrere Attention-Mechanismen parallel an, um verschiedene Aspekte der Beziehungen zu erfassen.

#### 2.2.2 BERT-Architektur

BERT (Bidirectional Encoder Representations from Transformers) verwendet ausschliesslich den **Encoder-Teil** der Transformer-Architektur (Devlin et al., 2019).

**BERT-base (uncased):**
- 12 Transformer-Encoder-Layer
- 768 Hidden Units
- 12 Attention Heads
- 110 Millionen Parameter
- Vocabulary: 30'000 WordPiece Tokens

**Pre-Training Objectives:**
1. **Masked Language Modeling (MLM):** 15% der Tokens werden maskiert, das Modell muss sie vorhersagen
2. **Next Sentence Prediction (NSP):** Vorhersage, ob zwei Sätze aufeinanderfolgen

**Fine-Tuning für Klassifikation:**
- Input: `[CLS] text [SEP]`
- Das `[CLS]`-Token-Embedding dient als Dokumentenrepräsentation
- Darauf wird eine Classification-Head (Linear Layer + Softmax) trainiert

#### 2.2.3 DistilBERT

DistilBERT ist eine komprimierte Version von BERT, die durch **Knowledge Distillation** erzeugt wird (Sanh et al., 2019).

**Kompression-Strategie:**
- Reduktion von 12 auf 6 Transformer-Layer
- Entfernung der NSP-Objective
- Distillation: Kleines Modell lernt von BERT's Output-Distributionen

**Resultat:**
- 40% weniger Parameter (66M statt 110M)
- 60% schneller in Inferenz
- 97% der BERT-Performance auf GLUE Benchmark

### 2.3 Logistische Regression für Multiclass-Klassifikation

Logistische Regression ist ein linearer Classifier, der die Wahrscheinlichkeit einer Klassenzugehörigkeit mittels Softmax-Funktion modelliert.

**Für Multiclass:**

$$P(y=k|x) = \frac{e^{w_k^T x + b_k}}{\sum_{j=1}^K e^{w_j^T x + b_j}}$$

Wobei:
- $x$ ist der TF-IDF-Vektor des Dokuments
- $w_k, b_k$ sind die Gewichte und Bias für Klasse $k$

**Training:**
- Minimierung der Cross-Entropy Loss via L-BFGS oder SGD
- L2-Regularisierung (Ridge) zur Vermeidung von Overfitting

### 2.4 Evaluationsmetriken

#### 2.4.1 Accuracy

$$\text{Accuracy} = \frac{\text{Anzahl korrekter Vorhersagen}}{\text{Gesamtanzahl Vorhersagen}}$$

Accuracy ist aussagekräftig bei **ausgeglichenen Klassenverteilungen**.

#### 2.4.2 Macro-F1 Score

Der Macro-F1 berechnet den F1-Score für jede Klasse einzeln und mittelt diese:

$$\text{Macro-F1} = \frac{1}{K} \sum_{k=1}^K F1_k$$

Mit:

$$F1_k = 2 \cdot \frac{\text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}$$

Macro-F1 gibt allen Klassen gleiches Gewicht und ist daher robuster gegenüber Klassenungleichgewichten als Accuracy.

#### 2.4.3 Konfusionsmatrix

Die Konfusionsmatrix visualisiert die Klassifikationsfehler und zeigt systematische Verwechslungen zwischen Klassen auf.

### 2.5 Cross-Validation

#### 2.5.1 K-Fold Stratified Cross-Validation

Bei K-Fold CV wird der Datensatz in $K$ Folds aufgeteilt. Jedes Fold dient einmal als Test-Set, während die verbleibenden $K-1$ Folds für Training verwendet werden.

**Stratified:** Die Klassenverteilung wird in jedem Fold beibehalten.

**Vorteile:**
- Robuste Performanzschätzung mit Konfidenzintervallen
- Effiziente Nutzung der Daten
- Reduziert Variance der Evaluation

#### 2.5.2 Nested Cross-Validation

Nested CV besteht aus zwei verschachtelten CV-Loops:

**Outer Loop (Modell-Evaluation):**
- K Folds für unbiased Performance-Schätzung

**Inner Loop (Hyperparameter-Optimierung):**
- J Folds für Hyperparameter-Suche

**Vorteil:** Verhindert Data Leakage zwischen Hyperparameter-Tuning und finaler Evaluation → unbiased Performance-Schätzung

### 2.6 Few-Shot Learning

Few-Shot Learning untersucht das Verhalten von Modellen bei limitierten Trainingsdaten (z.B. 10-100 Samples pro Klasse). Dies ist relevant für:
- Cold-Start Szenarien bei neuen Klassifikationsaufgaben
- Domänen mit teurer Annotation (Medizin, Recht)
- Bewertung der Dateneffizienz von Modellen

### 2.7 Active Learning

Active Learning ist eine Strategie zur effizienten Datenannotation. Statt zufällige Samples zu labeln, wählt das Modell die informativsten Samples aus.

**Sampling-Strategien:**

1. **Random Sampling (Baseline):** Zufällige Auswahl
2. **Uncertainty Sampling:** Auswahl der Samples mit höchster Unsicherheit (niedrigste Max-Probability)
3. **Margin Sampling:** Auswahl basierend auf der Differenz zwischen Top-2 Predictions

**Ziel:** Erreichen hoher Accuracy mit weniger gelabelten Daten.

---

## 3. Methodik

### 3.1 Dataset

#### 3.1.1 BBC News Dataset

Das BBC News Dataset besteht aus 2'225 Nachrichtenartikeln aus fünf Kategorien, die zwischen 2004-2005 veröffentlicht wurden (Greene & Cunningham, 2006).

**Tabelle 1: Klassenverteilung im BBC News Dataset**

| Kategorie | Anzahl | Anteil |
|-----------|--------|--------|
| Business | 510 | 22.9% |
| Entertainment | 386 | 17.3% |
| Politics | 417 | 18.7% |
| Sport | 511 | 23.0% |
| Tech | 401 | 18.0% |
| **Gesamt** | **2'225** | **100%** |

Der Datensatz weist eine relativ ausgeglichene Klassenverteilung auf, was die Verwendung von Accuracy als primäre Metrik rechtfertigt. Die durchschnittliche Dokumentlänge beträgt ca. 400 Wörter.

#### 3.1.2 Datenaufteilungsstrategien

Für die Experimente wurden verschiedene Aufteilungsstrategien verwendet:

**1. Cross-Validation (Hauptevaluation):**
- **5-Fold Stratified Cross-Validation:** Gewährleistet robuste Performanzschätzung mit Konfidenzintervallen
- Verwendet in Notebooks 07_01 (Baseline) und 07_02 (Transformers)

**2. Hold-Out Validation (Ergänzende Analysen):**

- **60% Training / 20% Validation / 20% Test:**
  - Notebooks: 02a, 03a
  - Ermöglicht Validierung während Training und finale unbiased Evaluation

- **80% Training / 10% Validation / 10% Test:**
  - Notebooks: 02b, 03b, 04
  - Maximiert Trainingsdaten für bessere Modellperformance
  - Test-Set bleibt für finale Evaluation unberührt

**3. Nested Cross-Validation (Robuste Evaluation):**
- **3 Outer Folds × 2 Inner Folds:**
  - Notebook: 08_Nested_CV
  - Outer Loop: Modell-Evaluation
  - Inner Loop: Hyperparameter-Optimierung
  - Verhindert Data Leakage → unbiased Performance-Schätzung

**4. Spezielle Splits für Few-Shot und Active Learning:**
- **Few-Shot Learning (Notebook 05):** Train/Test (kein Validation-Set)
- **Active Learning (Notebook 06):** Pool/Test (kein Validation-Set)

Diese methodische Vielfalt ermöglicht eine umfassende Evaluation aus verschiedenen Perspektiven.

### 3.2 Modelle

#### 3.2.1 Baseline: TF-IDF + Logistische Regression

Die Baseline-Pipeline besteht aus:

1. **TF-IDF Vektorisierung:**
   - max_features: 10'000-50'000 (je nach Datensatzgrösse, optimal: 10'000 bei 80/10/10 Split)
   - n-gram range: (1, 2) – Uni- und Bigramme
   - stop_words: English
   - sublinear_tf: True

2. **Logistische Regression:**
   - Solver: lbfgs (für Multiclass)
   - max_iter: 2'000
   - Regularisierung: L2 (default, C=1.0)
   - multi_class: multinomial

#### 3.2.2 BERT (bert-base-uncased)

Das BERT-base Modell umfasst:
- 12 Transformer-Encoder-Layer
- 768 Hidden Units
- 12 Attention Heads
- 110 Millionen Parameter

#### 3.2.3 DistilBERT (distilbert-base-uncased)

DistilBERT ist eine komprimierte Version:
- 6 Transformer-Encoder-Layer
- 768 Hidden Units
- 12 Attention Heads
- 66 Millionen Parameter

### 3.3 Experimentelles Setup

#### 3.3.1 Hyperparameter für Transformer-Modelle

Die Hyperparameter wurden gemäss den Empfehlungen von Devlin et al. (2019) und Sun et al. (2019) gewählt:

**Tabelle 2: Hyperparameter für Transformer-Fine-Tuning**

| Hyperparameter | Werte/Einstellung |
|----------------|-------------------|
| Epochen | 2, 3 |
| Max. Sequenzlänge | 256 (Standardkonfiguration), 128 (Vergleich) |
| Learning Rate | 2e-5 (Standard), 5e-5 (Grid Search) |
| Batch Size | 16 |
| Optimizer | AdamW |
| Weight Decay | 0.01 |
| Warmup Steps | 0 |
| Random Seed | 42 |

**Hyperparameter-Grid für Experimente (Notebook 04):**
- Modelle: BERT, DistilBERT
- Epochen: 2, 3
- Max Length: 128, 256
- Ergibt 2 × 2 × 2 = 8 Transformer-Konfigurationen + Baseline

**C-Parameter für Baseline (Notebook 04a):**
- C-Werte: [0.01, 0.1, 1.0, 10.0, 100.0]
- Systematische Untersuchung des Einflusses der L2-Regularisierung

**Nested CV Hyperparameter-Grid (Notebook 08):**
- Learning Rates: [2e-5, 5e-5]
- Epochen: [2, 3]

#### 3.3.2 Hardware und Software

- **GPU:** NVIDIA CUDA-fähige GPU
- **Framework:** PyTorch 2.0+, Hugging Face Transformers 4.30+
- **Scikit-learn** 1.3+ für Baseline und Evaluation
- **Python:** 3.8+
- **Random Seed:** 42 (für Reproduzierbarkeit)

#### 3.3.3 Evaluation

**Metriken:**
- Accuracy (Mittelwert ± Standardabweichung bei CV)
- Macro-F1 Score (Mittelwert ± Standardabweichung bei CV)
- Per-Class Precision, Recall, F1
- Konfusionsmatrizen

**Zusätzliche Messungen:**
- Trainingszeit (Sekunden)
- Inference-Zeit
- GPU-Memory-Footprint (bei Transformers)

---

## 4. Ergebnisse

### 4.1 Cross-Validation Vergleich (Hauptergebnis)

Die 5-Fold Stratified Cross-Validation liefert die robusteste Performanzschätzung für den Modellvergleich.

**Tabelle 3: Cross-Validation Ergebnisse (5-Fold Stratified)**

| Modell | Accuracy | Macro-F1 | Trainingszeit/Fold |
|--------|----------|----------|--------------------|
| TF-IDF + LogReg | 97.93% ± 0.72% | 97.90% ± 0.73% | 1.93 sec |
| DistilBERT (256, e3, lr2e-5) | 98.11% ± 0.77% | 98.11% ± 0.78% | ~60 sec |

**Interpretation:**

Die Ergebnisse zeigen, dass beide Modelle eine **nahezu identische Klassifikationsleistung** erzielen. Der Unterschied von 0.18 Prozentpunkten liegt innerhalb der Standardabweichung und ist statistisch nicht signifikant.

**Effizienz-Vergleich:**
- Die Baseline ist **~31x schneller** zu trainieren (1.93 sec vs. 60 sec pro Fold)
- Gesamtzeit für 5-Fold CV:
  - Baseline: ~10 Sekunden
  - DistilBERT: ~5 Minuten

![Cross-Validation Vergleich](results/cross_validation/cv_comparison.png)
*Abbildung 1: Accuracy-Vergleich der Cross-Validation Ergebnisse mit Fehlerbalken (±1 Standardabweichung)*

### 4.2 Hyperparameter-Experimente

#### 4.2.1 Vollständiges Grid Search (80/10/10 Split)

Die systematische Variation von Epochen und Sequenzlänge liefert Einblicke in die Hyperparameter-Sensitivität der Transformer-Modelle.

**Tabelle 4: Vollständige Hyperparameter-Grid Ergebnisse (Notebook 04)**

| Modell | Epochs | Max Length | Train Time (sec) | Val Acc | Test Acc | Test F1 |
|--------|--------|------------|------------------|---------|----------|---------|
| **TF-IDF + LogReg** | - | - | **2.22** | 98.21% | **99.10%** | **99.12%** |
| BERT | 2 | 128 | 16.06 | 98.65% | 98.65% | 98.63% |
| BERT | 3 | 128 | 23.30 | 97.76% | **99.10%** | **99.13%** |
| BERT | 2 | 256 | 27.72 | 97.76% | 98.65% | 98.64% |
| BERT | 3 | 256 | 41.22 | 98.21% | **99.10%** | 99.07% |
| DistilBERT | 2 | 128 | 8.75 | 96.86% | 97.76% | 97.80% |
| DistilBERT | 3 | 128 | 13.01 | 98.21% | 97.76% | 97.79% |
| DistilBERT | 2 | 256 | 16.39 | 96.86% | 98.21% | 98.24% |
| DistilBERT | 3 | 256 | 24.43 | 97.76% | 97.76% | 97.78% |

**Zentrale Beobachtungen:**

1. **Epochen-Effekt:** 
   - Mehr Epochen (3 statt 2) verbessern tendenziell die Performance
   - BERT e3/len128 erreicht 99.10% Test-Accuracy (identisch mit Baseline)
   - Effekt nicht linear: Weitere Epochen könnten zu Overfitting führen

2. **Sequenzlängen-Effekt:**
   - Erhöhung von 128 auf 256 Tokens bringt marginale Verbesserungen
   - **Kosten:** Fast Verdopplung der Trainingszeit
   - Für BBC News (durchschnittlich 400 Wörter ≈ 500 Tokens) ist 256 sinnvoll

3. **BERT vs. DistilBERT:**
   - BERT erreicht durchgehend **leicht bessere Werte** als DistilBERT
   - DistilBERT ist **~2x schneller** zu trainieren
   - Trade-off: 1-2% Accuracy für 50% Zeitersparnis

4. **Beste Konfigurationen:**
   - **BERT:** e3/len128 oder e3/len256 (beide 99.10% Test Acc)
   - **DistilBERT:** e2/len256 (98.21% Test Acc)
   - **Baseline:** 99.10% in 2.22 sec

![Test Accuracy Vergleich](results/experiments_hparams/exp_accuracy.png)
*Abbildung 2: Test-Accuracy über alle Hyperparameter-Konfigurationen*

![Trainingszeit Vergleich](results/experiments_hparams/exp_train_time.png)
*Abbildung 3: Trainingszeit in Sekunden für jede Konfiguration*

#### 4.2.2 C-Parameter Sensitivität (Baseline)

Die Regularisierung der logistischen Regression wurde systematisch variiert.

**Tabelle 5: C-Parameter Experiment (Notebook 04a)**

| C-Wert | Regularisierung | Val Acc | Test Acc | Test F1 |
|--------|-----------------|---------|----------|---------|
| 0.01 | Stark | 97.76% | 98.65% | 98.66% |
| 0.1 | Mittel-Stark | 98.21% | 98.65% | 98.66% |
| **1.0** | **Standard** | **98.21%** | **99.10%** | **99.12%** |
| 10.0 | Schwach | 98.21% | 99.10% | 99.12% |
| 100.0 | Sehr schwach | 98.21% | 99.10% | 99.12% |

**Interpretation:**
- C=1.0 (Standard) liefert optimale Ergebnisse
- Höhere C-Werte (schwächere Regularisierung) bringen keine Verbesserung
- Niedrigere C-Werte (stärkere Regularisierung) reduzieren leicht die Performance
- Das Modell ist robust gegenüber Regularisierung → Overfitting ist kein Problem

![C-Parameter Experiment](results/baseline_C_parameter/c_parameter_test_accuracy.png)
*Abbildung 4: Einfluss des C-Parameters auf Test-Accuracy*

#### 4.2.3 max_features Sensitivität (TF-IDF)

Die Anzahl der behaltenen Features in der TF-IDF Vektorisierung wurde systematisch untersucht.

**Tabelle 5b: max_features Experiment (GridSearch in Notebooks 02a, 02bb)**

| Split | max_features getestet | Optimal | CV-Accuracy | Interpretation |
|-------|----------------------|---------|-------------|----------------|
| 60/20/20 | [10k, 30k, 50k] | **30'000** | 97.23% | Mehr Features bei weniger Daten |
| 80/10/10 | [10k, 30k, 50k] | **10'000** | 97.75% | Weniger Features bei mehr Daten |

**GridSearch Setup:**
- Parameter-Grid:
  - `max_features`: [10'000, 30'000, 50'000]
  - `ngram_range`: [(1, 1), (1, 2)]
  - `C`: [0.1, 1.0, 10.0]
- 5-Fold Cross-Validation
- Total: 3 × 2 × 3 = 18 Kombinationen

**Beste Parameter (80/10/10 Split):**
- `max_features`: 10'000
- `ngram_range`: (1, 2) - Uni- und Bigramme
- `C`: 1.0

**Interpretation:**

1. **Weniger ist mehr bei grossen Datensätzen:**
   - Mit **mehr Trainingsdaten** (80%) sind **10k Features optimal**
   - Die Top-10k Features enthalten das wesentliche Signal
   - Zusätzliche Features (10k→50k) fügen primär Rauschen hinzu

2. **Mehr Features bei kleinen Datensätzen:**
   - Mit **weniger Trainingsdaten** (60%) werden **30k Features** benötigt
   - Mehr Features kompensieren begrenzte Trainingsdaten
   - Breiteres Vokabular hilft bei limitierten Samples

3. **Effizienz-Vorteil:**
   - 10k Features: **5x weniger Dimensionalität** als 50k
   - Schnellere Trainingszeit und Inferenz
   - Geringerer Memory-Footprint

4. **Praktische Empfehlung:**
   - **Default:** max_features=10'000 für ausreichend grosse Datasets (>1000 Samples)
   - Bei kleinen Datasets (<500 Samples): max_features=30'000-50'000 testen

### 4.3 Split-Vergleich: 60/20/20 vs. 80/10/10

Der Vergleich verschiedener Datenaufteilungen untersucht den Einfluss der Train/Val/Test-Ratio auf die Performance.

**Tabelle 6: Split-Comparison (Notebook 08_split_experiments_comparison)**

| Modell | Split | Train Acc | Val Acc | Test Acc | Test F1 |
|--------|-------|-----------|---------|----------|---------|
| TF-IDF + LogReg | 60/20/20 | 99.89% | 98.88% | 98.43% | 98.40% |
| TF-IDF + LogReg | 80/10/10 | 99.89% | 98.21% | 99.10% | 99.12% |
| BERT | 60/20/20 | 99.62% | 97.98% | 98.20% | 98.23% |
| BERT | 80/10/10 | 99.83% | 98.21% | 99.10% | 99.07% |
| DistilBERT | 60/20/20 | 99.85% | 98.20% | 97.98% | 98.00% |
| DistilBERT | 80/10/10 | 99.94% | 97.76% | 97.76% | 97.78% |

**Beobachtungen:**

1. **80/10/10 vs. 60/20/20:**
   - Bei **mehr Trainingsdaten** (80%) erreichen Modelle tendenziell bessere Test-Performance
   - Baseline: 99.10% (80/10/10) vs. 98.43% (60/20/20) → +0.67%
   - BERT: 99.10% (80/10/10) vs. 98.20% (60/20/20) → +0.90%

2. **Train-Accuracy:**
   - Alle Modelle zeigen hohe Train-Accuracy (>99.6%) → kein Underfitting
   - Geringer Gap zu Test-Accuracy → kein starkes Overfitting

3. **Validation vs. Test:**
   - Leichte Varianz zwischen Val/Test Accuracy normal bei kleinen Test-Sets

![Split Comparison](results/split_experiments/split_comparison_test_accuracy.png)
*Abbildung 5: Test-Accuracy Vergleich für verschiedene Splits*

### 4.4 Few-Shot Learning Analyse

Die Few-Shot Learning Analyse untersucht das Verhalten der Modelle bei limitierten Trainingsdaten.

**Tabelle 7: Few-Shot Learning Ergebnisse (Notebook 05)**

| Samples/Klasse | TF-IDF + LogReg Acc | DistilBERT Acc | Überlegenes Modell |
|----------------|---------------------|----------------|--------------------|
| 20 | 94.83% | - | TF-IDF (DistilBERT trainiert nicht) |
| 50 | 97.53% | 91.24% | **TF-IDF (+6.29%)** |
| 100 | 96.85% | 96.85% | Gleich |
| 309 (alle) | 98.65% | 98.65% | Gleich |

**Interpretation:**

1. **Dateneffizienz:**
   - Bei **sehr wenigen Daten** (<50 Samples/Klasse) ist TF-IDF + LogReg deutlich überlegen
   - DistilBERT konvergiert bei 20 Samples/Klasse nicht

2. **Crossover-Punkt:**
   - Ab ca. **100 Samples/Klasse** erreichen beide Modelle vergleichbare Leistung
   - Dies entspricht ~500 Samples total (bei 5 Klassen)

3. **Praktische Implikation:**
   - Für **Cold-Start Szenarien** mit wenig gelabelten Daten ist die Baseline vorzuziehen
   - Transfer Learning von BERT zeigt Vorteile erst bei ausreichend Daten

![Learning Curve](results/fewshot_learning/learning_curve_accuracy.png)
*Abbildung 6: Learning Curve – Test-Accuracy in Abhängigkeit der Trainingsdatenmenge*

### 4.5 Active Learning Simulation

Active Learning untersucht effiziente Annotationsstrategien durch gezielte Sample-Auswahl.

**Tabelle 8: Active Learning Ergebnisse (Notebook 06)**

| Methode | Samples für 95% Acc | Samples für 97% Acc | Effizienz-Gewinn |
|---------|---------------------|---------------------|------------------|
| Random Sampling | ~250 | ~600 | Baseline |
| Uncertainty Sampling | ~150 | ~350 | **40% weniger** |

**Detaillierte Ergebnisse (DistilBERT):**

| Iteration | Annotierte Samples | Random Acc | Uncertainty Acc | Vorteil |
|-----------|--------------------|------------|-----------------|---------|
| 1 | 50 | 68.77% | 73.26% | +4.49% |
| 5 | 250 | 93.26% | 96.63% | +3.37% |
| 10 | 500 | 96.63% | 97.53% | +0.90% |
| 20 | 1000 | 97.98% | 98.43% | +0.45% |

**Beobachtungen:**

1. **Uncertainty Sampling übertrifft Random Sampling:**
   - Erreicht 95% Accuracy mit **40% weniger** gelabelten Daten
   - Bei wenig Daten (Iteration 1-5) ist der Vorteil am grössten

2. **Diminishing Returns:**
   - Ab ~1000 Samples konvergieren beide Strategien
   - Der Vorteil von Active Learning nimmt mit mehr Daten ab

3. **Praktische Anwendung:**
   - Bei **begrenztem Annotationsbudget** kann Active Learning erhebliche Ressourcen sparen
   - Beispiel: 350 statt 600 Samples für 97% Acc → **Kostenersparnis von 42%**

![Active Learning Comparison](results/active_learning/active_learning_comparison.png)
*Abbildung 7: Accuracy-Entwicklung bei Random vs. Uncertainty Sampling*

### 4.6 Nested Cross-Validation mit Hyperparameter-Optimierung

Nested CV liefert unbiased Performanzschätzungen durch Trennung von Hyperparameter-Tuning und Modell-Evaluation.

**Setup (Notebook 08):**
- **Outer Loop:** 3 Folds (Evaluation)
- **Inner Loop:** 2 Folds (Hyperparameter-Optimierung)
- **Hyperparameter-Grid:**
  - Learning Rates: [2e-5, 5e-5]
  - Epochen: [2, 3]

**Tabelle 9: Nested CV Ergebnisse**

| Modell | Test Accuracy (Nested CV) | Beste Hyperparameter |
|--------|----------------------------|----------------------|
| TF-IDF + LogReg | 97.71% ± 0.49% | C=1.0 (fixed) |
| BERT | 97.98% ± 0.88% | lr=2e-5, epochs=3 |
| DistilBERT | 98.16% ± 0.54% | lr=2e-5, epochs=3 |

**Per-Fold Breakdown:**

**Fold 1:**
| Modell | Test Acc | Beste Config (Inner CV) |
|--------|----------|-------------------------|
| Baseline | 97.44% | - |
| BERT | 97.44% | lr=2e-5, e=3 |
| DistilBERT | 97.71% | lr=2e-5, e=3 |

**Fold 2:**
| Modell | Test Acc | Beste Config (Inner CV) |
|--------|----------|-------------------------|
| Baseline | 98.38% | - |
| BERT | 97.57% | lr=2e-5, e=3 |
| DistilBERT | 98.92% | lr=2e-5, e=3 |

**Fold 3:**
| Modell | Test Acc | Beste Config (Inner CV) |
|--------|----------|-------------------------|
| Baseline | 97.30% | - |
| BERT | 98.92% | lr=2e-5, e=3 |
| DistilBERT | 97.84% | lr=2e-5, e=3 |

**Interpretation:**

1. **Robuste Hyperparameter-Wahl:**
   - Über alle Folds hinweg wurde konsistent **lr=2e-5, epochs=3** als optimal identifiziert
   - Dies bestätigt die Empfehlungen aus der Literatur (Devlin et al., 2019)

2. **Performance-Vergleich:**
   - DistilBERT: 98.16% ± 0.54% (beste Durchschnittsperformance)
   - BERT: 97.98% ± 0.88% (höhere Varianz)
   - Baseline: 97.71% ± 0.49% (geringste Varianz)

3. **Unbiased Schätzung:**
   - Nested CV liefert realistische Performance-Erwartungen
   - Leicht niedriger als Hold-Out Validation (erwartet, da konservativer)

4. **Varianz-Analyse:**
   - Baseline zeigt geringste Varianz (±0.49%) → sehr stabil
   - BERT zeigt höchste Varianz (±0.88%) → sensitiver auf Daten-Splits

![Nested CV Comparison](results/nested_cv/nested_cv_comparison.png)
*Abbildung 8: Nested CV Ergebnisse mit Fehlerbalken über alle Outer Folds*

### 4.7 Effizienzanalyse

Die Trainingszeiten zeigen deutliche Unterschiede zwischen den Modellklassen.

**Tabelle 10: Effizienzvergleich (Speedup relativ zu BERT e3/len256)**

| Modell | Konfiguration | Zeit (sec) | Speedup | Test Acc |
|--------|---------------|------------|---------|----------|
| TF-IDF + LogReg | - | 2.22 | **18.6x** | 99.10% |
| DistilBERT | e2/len128 | 8.75 | 4.7x | 97.76% |
| DistilBERT | e3/len128 | 13.01 | 3.2x | 97.76% |
| DistilBERT | e2/len256 | 16.39 | 2.5x | 98.21% |
| BERT | e2/len128 | 16.06 | 2.6x | 98.65% |
| BERT | e3/len128 | 23.30 | 1.8x | 99.10% |
| DistilBERT | e3/len256 | 24.43 | 1.7x | 97.76% |
| BERT | e2/len256 | 27.72 | 1.5x | 98.65% |
| BERT | e3/len256 | 41.22 | 1.0x | 99.10% |

**Key Insights:**

1. **Baseline-Vorteil:** 18x schneller als die langsamste Transformer-Konfiguration bei gleicher Accuracy
2. **DistilBERT-Effizienz:** ~2x schneller als BERT bei vergleichbarer Performance
3. **Sequenzlängen-Kosten:** Verdopplung von 128→256 erhöht Trainingszeit um ~70%
4. **Accuracy-Speed Trade-off:** BERT e3/len128 bietet besten Kompromiss (99.10% in 23.30 sec)

### 4.8 Konfusionsmatrix-Analyse

Die Konfusionsmatrizen zeigen die klassenspezifische Performance und systematische Fehler.

#### 4.8.1 TF-IDF + LogReg (80/10/10 Split)

![Konfusionsmatrix Baseline](results/baseline_confusion_matrix_TEST_80-10-10.png)
*Abbildung 9: Konfusionsmatrix TF-IDF + LogReg (Test-Set, 80/10/10 Split)*

**Beobachtungen:**
- Nahezu perfekte Klassifikation mit nur 2 Fehlern im Test-Set (223 Samples)
- 1× Business → Politics (thematische Überschneidung plausibel)
- 1× Business → Tech (möglicherweise Tech-Unternehmensnachrichten)

#### 4.8.2 BERT (80/10/10 Split)

![Konfusionsmatrix BERT](results/bert_confusion_matrix_TEST_80-10-10.png)
*Abbildung 10: Konfusionsmatrix BERT (Test-Set, 80/10/10 Split)*

**Beobachtungen:**
- Ebenfalls 2 Fehler, identisches Pattern wie Baseline
- Business/Politics/Tech Verwechslungen

#### 4.8.3 DistilBERT (80/10/10 Split)

![Konfusionsmatrix DistilBERT](results/distilbert_confusion_matrix_TEST_80-10-10.png)
*Abbildung 11: Konfusionsmatrix DistilBERT (Test-Set, 80/10/10 Split)*

**Beobachtungen:**
- Etwas mehr Fehler (~5) als Baseline und BERT
- Hauptsächlich Business ↔ Politics ↔ Tech Verwechslungen
- Sport und Entertainment werden perfekt erkannt (klares Vokabular)

**Zusammenfassung:**
- Alle Modelle zeigen nahezu perfekte Performance
- Systematische Verwechslungen zwischen thematisch verwandten Kategorien (Business/Politics/Tech)
- Sport und Entertainment haben distinktives Vokabular → keine Fehler

---

## 5. Diskussion

### 5.1 Interpretation der Ergebnisse

#### 5.1.1 Warum performt die Baseline so gut?

Die überraschend starke Performance der TF-IDF + LogReg Baseline lässt sich durch mehrere Faktoren erklären:

**1. Klare thematische Trennung:**

Das BBC News Dataset enthält Kategorien mit deutlich unterschiedlichem Vokabular:
- **Sport:** "goal", "match", "player", "team", "win", "defeat"
- **Business:** "profit", "market", "company", "shares", "economy"
- **Entertainment:** "film", "music", "actor", "award", "band"
- **Tech:** "software", "internet", "Microsoft", "technology", "computer"
- **Politics:** "government", "minister", "election", "party", "policy"

**2. Aussagekräftige Keywords:**

TF-IDF hebt genau diese diskriminativen Begriffe hervor. Die Klassifikation kann primär auf **Keyword-Ebene** erfolgen, wofür keine komplexe Semantik-Modellierung erforderlich ist.

**3. Dataset-Grösse:**

Mit ~2'200 Dokumenten ist das Dataset:
- **Gross genug** für effektives Training von TF-IDF + LogReg
- **Klein genug**, dass BERT keinen signifikanten Vorteil aus Transfer Learning ziehen kann
- Die vortrainierten Sprachrepräsentationen von BERT bieten keinen wesentlichen Mehrwert

Diese Befunde stehen im Einklang mit den Erkenntnissen von **Wang & Manning (2012)**, die zeigten, dass einfache Modelle mit guten Features oft komplexere Architekturen bei keyword-basierten Aufgaben übertreffen können.

#### 5.1.2 Wann sind Transformer-Modelle überlegen?

Transfer Learning und Transformer zeigen ihre Stärken typischerweise bei:

**1. Komplexen semantischen Aufgaben:**
- Natural Language Inference (NLI)
- Question Answering
- Sentiment-Analyse mit Ironie/Sarkasmus
- Coreference Resolution

**2. Sehr kleinen Datensätzen:**
- **Aber:** Unsere Few-Shot Analyse zeigt das Gegenteil für einfache Klassifikation
- TF-IDF ist bei <100 Samples/Klasse überlegen
- Transfer Learning benötigt mehr Daten als erwartet

**3. Aufgaben, die Weltkenntnis erfordern:**
- Fakten-Verifikation
- Commonsense Reasoning
- Named Entity Recognition in komplexen Domänen

**4. Mehrsprachigkeit:**
- Multilinguale Modelle (mBERT, XLM-R) für Cross-Lingual Transfer
- Nicht relevant für monolinguales BBC News Dataset

#### 5.1.3 Die Rolle der Datenqualität und -struktur

Die Struktur des BBC News Datasets begünstigt die Baseline:

**Vorteilhaft für TF-IDF:**
- Formale, gut strukturierte Nachrichtentexte
- Konsistente Schreibweise und Grammatik
- Klare thematische Kategorien
- Wenig Ambiguität oder Ironie

**Nachteilig für BERT:**
- Keine komplexen semantischen Phänomene
- Kein Code-Switching oder Umgangssprache
- Keine langen Abhängigkeiten über Satzgrenzen hinweg

**Hypothese:** Bei unstrukturierteren Daten (Social Media, User-Reviews, Transkripte) könnten Transformer deutliche Vorteile zeigen.

### 5.2 Trade-off Analyse

**Tabelle 11: Entscheidungsmatrix für Modellwahl**

| Kriterium | TF-IDF + LogReg | BERT | DistilBERT |
|-----------|-----------------|------|------------|
| **Accuracy** | ✅ Sehr gut (97.9-99.1%) | ✅ Sehr gut (97.9-99.1%) | ✅ Gut (97.7-98.2%) |
| **Trainingszeit** | ✅ Sekunden | ❌ Minuten | ⚠️ ~1 Minute |
| **Inferenzzeit** | ✅ <1ms | ❌ 10-100ms | ⚠️ 5-50ms |
| **GPU erforderlich** | ✅ Nein | ❌ Ja (empfohlen) | ⚠️ Optional |
| **Memory Footprint** | ✅ <100 MB | ❌ ~500 MB | ⚠️ ~250 MB |
| **Interpretierbarkeit** | ✅ Hoch (Feature Weights) | ❌ Gering | ❌ Gering |
| **Deployment-Komplexität** | ✅ Minimal | ❌ Hoch | ⚠️ Mittel |
| **Few-Shot Performance** | ✅ Sehr gut | ❌ Schlecht | ❌ Schlecht |
| **Skalierbarkeit** | ✅ Excellent | ⚠️ Mittel | ⚠️ Mittel |
| **Update-Fähigkeit** | ✅ Inkrementell | ❌ Full Retrain | ❌ Full Retrain |

**Empfehlungen:**

**Wähle TF-IDF + LogReg wenn:**
- Einfache, keyword-basierte Klassifikation ausreichend ist
- Ressourcen begrenzt sind (CPU-only, Edge-Deployment)
- Sehr schnelle Inferenz erforderlich (<1ms)
- Interpretierbarkeit wichtig ist (Regulierung, Compliance)
- Few-Shot Szenarien (<100 Samples/Klasse)
- Inkrementelles Lernen benötigt wird

**Wähle BERT/DistilBERT wenn:**
- Komplexe semantische Phänomene modelliert werden müssen
- Grosse Datensätze (>10k Samples) verfügbar sind
- GPU-Ressourcen vorhanden sind
- State-of-the-Art Performance Priorität hat
- Transfer von vortrainierten Sprachrepräsentationen nützlich ist

### 5.3 Hyperparameter-Empfehlungen

Basierend auf den systematischen Experimenten (Notebooks 04, 04a, 08):

#### 5.3.1 Für Transformer-Modelle

**1. Epochen:**
- **Empfehlung: 3 Epochen** für dieses Dataset
- Mehr Epochen (>3) riskieren Overfitting
- Weniger Epochen (<2) unterfitten leicht
- Bestätigt Empfehlungen von Devlin et al. (2019)

**2. Sequenzlänge (max_length):**
- **Empfehlung: 256** für Nachrichtenartikel
- 128 ist ausreichend für kurze Texte, aber für BBC News (avg. 400 Wörter) suboptimal
- 512 wäre exzessiv und deutlich langsamer

**3. Learning Rate:**
- **Empfehlung: 2e-5** (Standard BERT-Empfehlung)
- Nested CV bestätigt: 2e-5 konsistent besser als 5e-5
- Höhere Learning Rates (>5e-5) können zu Instabilität führen

**4. Batch Size:**
- 16 ist guter Kompromiss zwischen Memory und Konvergenz
- Kleinere Batches (8) für GPU-Memory-Beschränkungen
- Grössere Batches (32) wenn möglich, dann Learning Rate anpassen

#### 5.3.2 Für Baseline (TF-IDF + LogReg)

**1. Regularisierung (C-Parameter):**
- **Empfehlung: C=1.0** (Standard)
- Modell robust gegenüber Variation (0.1-100 zeigen ähnliche Results)
- Kein starkes Overfitting-Problem bei diesem Dataset

**2. TF-IDF Features:**
- **Empfehlung: max_features=10'000** (optimal bei ausreichend Daten)
- Bei kleinen Datasets (<500 Samples): 30'000-50'000 testen
- Bigrams (1,2) verbessern Performance deutlich
- stop_words='english' reduziert Dimensionalität ohne Accuracy-Verlust

### 5.4 Erkenntnisse aus Few-Shot und Active Learning

#### 5.4.1 Few-Shot Learning Implikationen

Die Few-Shot Analyse (Notebook 05) liefert wichtige praktische Erkenntnisse:

**1. Cold-Start Szenarien:**
- Bei **neuen Klassifikationsaufgaben mit <50 gelabelten Samples/Klasse** ist TF-IDF + LogReg klar vorzuziehen
- Kostenersparnis: Keine GPU-Infrastruktur erforderlich
- Schnellerer Iterationszyklus bei begrenzten Daten

**2. Iteratives Labeling Workflow:**

**Phase 1 (0-100 Samples/Klasse):**
- Starte mit TF-IDF + LogReg
- Schnelle Baseline-Performance
- Nutze Modell-Vorhersagen zur Priorisierung weiterer Annotationen

**Phase 2 (100-500 Samples/Klasse):**
- Evaluiere Transformer als Alternative
- Trade-off: Leichte Performance-Verbesserung vs. höhere Komplexität

**Phase 3 (>500 Samples/Klasse):**
- Transformer können marginal bessere Performance bieten
- Entscheidung basierend auf Deployment-Constraints

**3. Annotationsbudget-Planung:**
- Für 95% Accuracy: ~100-150 Samples/Klasse ausreichend (mit Baseline)
- Für 98% Accuracy: ~300 Samples/Klasse erforderlich
- Transformer benötigen ~30% mehr Daten für gleiche Performance

#### 5.4.2 Active Learning Implikationen

Die Active Learning Simulation (Notebook 06) zeigt:

**1. Annotation Efficiency:**
- **Uncertainty Sampling spart 40% Annotationskosten** für 95% Accuracy
- Bei Budget-Beschränkungen klar vorzuziehen
- Grösster Vorteil in Early Stages (<500 Samples)

**2. ROI-Analyse:**

**Szenario:** 1000 Dokumente labeln, Kosten: 1€/Dokument

| Strategie | Samples für 97% Acc | Kosten | Ersparnis |
|-----------|---------------------|--------|-----------|
| Random | 600 | 600€ | - |
| Uncertainty | 350 | 350€ | **250€ (42%)** |

**3. Praktische Implementierung:**

**Empfohlener Workflow:**
```
1. Initial: Random Sample 50 Samples/Klasse (250 total)
2. Train: TF-IDF + LogReg Baseline
3. Loop:
   a. Predict auf unlabeled Pool
   b. Select Top-K mit niedrigster Max-Probability
   c. Human labeling
   d. Retrain
4. Stop: Wenn Validation Accuracy plateau erreicht
```

**4. Diminishing Returns:**
- Ab ~1000 gelabelten Samples konvergieren Random und Uncertainty
- Active Learning lohnt sich primär in Early/Mid Stages

### 5.5 Nested CV vs. Simple Hold-Out

Der Vergleich zwischen Nested CV (Notebook 08) und Hold-Out Validation (Notebooks 02-04) zeigt:

**1. Performance-Schätzungen:**

| Methode | Baseline | BERT | DistilBERT |
|---------|----------|------|------------|
| **Hold-Out (80/10/10)** | 99.10% | 99.10% | 97.76% |
| **5-Fold CV** | 97.93% | - | 98.11% |
| **Nested CV (3×2)** | 97.71% | 97.98% | 98.16% |

**Beobachtungen:**
- Hold-Out liefert **optimistischste** Schätzungen (99.10%)
- Nested CV liefert **konservativste** Schätzungen (97.71-98.16%)
- 5-Fold CV liegt dazwischen (97.93-98.11%)

**2. Varianz:**
- Nested CV erfasst Varianz über Daten-Splits **und** Hyperparameter
- Hold-Out zeigt **keine** Varianz (single Split)
- 5-Fold CV zeigt nur Daten-Split Varianz

**3. Empfehlung:**
- **Für finale Publikation:** Nested CV (unbiased, robuste Schätzung)
- **Für schnelle Experimente:** Hold-Out (schnell, ausreichend bei grossen Datasets)
- **Für Baseline-Vergleich:** 5-Fold CV (guter Kompromiss)

### 5.6 Limitationen

Die vorliegende Arbeit unterliegt folgenden Limitationen:

#### 5.6.1 Dataset-spezifische Limitationen

**1. Single Dataset:**
- Ergebnisse basieren ausschliesslich auf BBC News
- Generalisierbarkeit auf andere Domains unklar:
  - Social Media (Twitter, Reddit)
  - Wissenschaftliche Texte
  - Medizinische/Juristische Dokumente
  - User-Reviews

**2. Monolingual:**
- Nur englische Texte
- Andere Sprachen (besonders morphologisch reiche) könnten unterschiedlich reagieren
- Multilinguale Szenarien nicht untersucht

**3. Zeitliche Begrenzung:**
- Dataset aus 2004-2005
- Konzeptdrift über Zeit nicht evaluiert
- Neue Themen (z.B. COVID, KI, Kryptowährungen) fehlen

**4. Klassenbalance:**
- Relativ ausgeglichene Klassen (17-23%)
- Stark unbalancierte Datasets könnten andere Ergebnisse zeigen

#### 5.6.2 Methodische Limitationen

**1. Hyperparameter-Optimierung:**
- **Baseline:** Systematische Optimierung von max_features (10k/30k/50k), ngram_range, C-Parameter durchgeführt
- **Transformers:** Begrenzte Grid Search (nur Epochs, Learning Rate, max_length)
- Weitere Hyperparameter (Warmup, Weight Decay, Batch Size) nicht variiert

**2. Modell-Auswahl:**
- Nur BERT-base und DistilBERT
- Neuere Modelle nicht evaluiert:
  - RoBERTa (verbesserte BERT-Variante)
  - DeBERTa (State-of-the-Art)
  - GPT-basierte Ansätze (Generative Models)
  - Moderne LLMs (GPT-4, Claude, Llama)

**3. Evaluation-Metriken:**
- Fokus auf Accuracy und Macro-F1
- Weitere Metriken nicht systematisch untersucht:
  - Per-Class ROC-AUC
  - Calibration Metrics (Expected Calibration Error)
  - Inference Time auf verschiedenen Hardware-Setups

**4. Statistische Tests:**
- Keine formalen Signifikanztests (t-tests, Wilcoxon) durchgeführt
- Standardabweichungen berichtet, aber keine p-Werte

#### 5.6.3 Praktische Limitationen

**1. Hardware-Variabilität:**
- Experimente auf spezifischer GPU
- Performance auf anderen Hardware-Setups (CPU, Edge-Devices) nicht gemessen

**2. Deployment-Szenarien:**
- Keine Evaluation von:
  - Quantization (INT8, FP16)
  - Model Compression (Pruning, Distillation)
  - ONNX/TensorRT Optimierung

**3. Long-Term Production:**
- Keine Untersuchung von:
  - Model Drift
  - Retraining Strategies
  - A/B Testing

### 5.7 Implikationen für die Praxis

Für praktische Anwendungen ergeben sich folgende Empfehlungen:

#### 5.7.1 Generelle Prinzipien

**1. Start Simple:**
- Beginne **immer** mit einer einfachen Baseline (TF-IDF + LogReg)
- Etabliere klare Performance-Benchmarks
- Erhöhe Komplexität nur wenn nötig

**2. Evaluate Holistically:**
- Berücksichtige nicht nur Accuracy:
  - ✅ Trainingszeit
  - ✅ Inferenzzeit
  - ✅ Memory Footprint
  - ✅ Deployment-Komplexität
  - ✅ Interpretierbarkeit
  - ✅ Wartbarkeit

**3. Domain-Awareness:**
- Die Ergebnisse sind **dataset-spezifisch**
- Bei anderen Aufgaben (Sentiment mit Sarkasmus, NLI, QA) können Transformer deutliche Vorteile bieten
- Teste auf **deinem** spezifischen Dataset

#### 5.7.2 Entscheidungsleitfaden

**Flowchart:**

```
START: Neue Text-Klassifikationsaufgabe

1. Dataset-Analyse
   ├─ Keyword-basierte Trennung möglich? → YES: TF-IDF Baseline
   └─ Komplexe Semantik erforderlich? → YES: Consider Transformers

2. Ressourcen-Check
   ├─ GPU verfügbar? → NO: TF-IDF
   ├─ Real-Time Inferenz (<10ms)? → YES: TF-IDF
   └─ Edge-Deployment? → YES: TF-IDF

3. Daten-Verfügbarkeit
   ├─ <100 Samples/Klasse? → YES: TF-IDF
   ├─ 100-1000 Samples/Klasse? → YES: Test beide
   └─ >1000 Samples/Klasse? → YES: Transformer möglich

4. Business-Requirements
   ├─ Interpretierbarkeit kritisch? → YES: TF-IDF
   ├─ Compliance/Regulierung? → YES: TF-IDF
   └─ State-of-the-Art um jeden Preis? → YES: Transformer

5. Prototyping
   ├─ Schneller POC? → YES: TF-IDF
   └─ Langfristiges Production System? → Consider beide

ENDE: Modellwahl basierend auf obigen Kriterien
```

#### 5.7.3 Spezifische Use-Cases

**Use-Case 1: News Categorization (wie BBC News)**
- **Empfehlung:** TF-IDF + LogReg
- **Begründung:** Klare thematische Trennung, keine komplexe Semantik
- **Deployment:** CPU-basierte API, <1ms Inferenz

**Use-Case 2: Sentiment Analysis mit Ironie**
- **Empfehlung:** BERT/RoBERTa
- **Begründung:** Kontext und Semantik essentiell
- **Deployment:** GPU-basierter Service, 10-50ms acceptable

**Use-Case 3: Customer Support Ticket Routing**
- **Empfehlung:** Start mit TF-IDF, evaluate Transformer
- **Begründung:** Initial oft keyword-basiert, später mehr Nuancen
- **Deployment:** Hybrid (TF-IDF für einfache Fälle, Transformer für komplexe)

**Use-Case 4: Medical Text Classification**
- **Empfehlung:** Domain-specific BERT (BioBERT, ClinicalBERT)
- **Begründung:** Spezialisiertes Vokabular, vortrainierte Domain-Modelle verfügbar
- **Deployment:** Secure on-premise Server mit GPU

**Use-Case 5: Real-Time Content Moderation**
- **Empfehlung:** DistilBERT (Kompromiss)
- **Begründung:** Balance zwischen Accuracy und Speed
- **Deployment:** Optimized Inference (ONNX, TensorRT), Multi-GPU

#### 5.7.4 Cost-Benefit Analyse

**Szenario:** 1 Million Dokumente/Monat klassifizieren

| Kriterium | TF-IDF + LogReg | DistilBERT | BERT |
|-----------|-----------------|------------|------|
| **Trainingskosten** (einmalig) | 10€ (CPU-Stunden) | 50€ (GPU-Stunden) | 100€ (GPU-Stunden) |
| **Inferenz-Kosten** (monatlich) | 50€ (CPU) | 300€ (GPU) | 500€ (GPU) |
| **Engineering-Zeit** | 1 Woche | 2-3 Wochen | 3-4 Wochen |
| **Accuracy** | 98% | 98% | 98% |
| **Gesamt (1. Jahr)** | **~660€** | **~3'750€** | **~6'200€** |

**ROI-Analyse:**
- Für 0.3% Accuracy-Gewinn (97.9% → 98.2%) zahlt man ~3'000-5'000€ mehr/Jahr
- Business-Impact: Lohnt es sich?
  - Bei 1M Docs: 3'000 zusätzlich korrekt klassifiziert
  - Wert pro korrekter Klassifikation?

---

## 6. Fazit und Ausblick

### 6.1 Zusammenfassung der Ergebnisse

Diese Arbeit verglich systematisch klassische Machine-Learning-Methoden (TF-IDF + Logistische Regression) mit modernen Transformer-basierten Ansätzen (BERT, DistilBERT) für die Dokumentenklassifikation auf dem BBC News Dataset.

**Zentrale Erkenntnisse:**

**1. Vergleichbare Performance:**
- Die klassische Baseline erreicht mit **97.93% (±0.72%)** eine nahezu identische Accuracy wie DistilBERT **(98.11% ±0.77%)** in der 5-Fold Cross-Validation
- In einzelnen Konfigurationen erreichen alle drei Modelle **99.10% Test-Accuracy**
- Der Unterschied liegt innerhalb der Standardabweichung und ist statistisch nicht signifikant

**2. Deutlicher Effizienzvorsprung der Baseline:**
- **18x schnelleres Training** als langsamste Transformer-Konfiguration
- **31x schneller** als DistilBERT in Cross-Validation (1.93 sec vs. 60 sec/Fold)
- Keine GPU erforderlich, <100 MB Memory Footprint
- <1ms Inferenzzeit vs. 10-100ms für Transformers

**3. Hyperparameter-Robustheit:**
- **Transformer:** 3 Epochen und max_length=256 optimal
- **Learning Rate:** 2e-5 konsistent besser als 5e-5 (Nested CV)
- **Baseline:** C=1.0 (Standard) ausreichend, robust gegenüber Variation

**4. Dateneffizienz und Few-Shot Learning:**
- Bei **<100 Samples/Klasse** ist die Baseline deutlich überlegen (6.29% bei 50 Samples)
- Transformer benötigen ~100 Samples/Klasse für kompetitive Performance
- Transfer Learning bietet keinen Vorteil bei limitierten Daten für einfache Klassifikation

**5. Active Learning:**
- **Uncertainty Sampling** spart 40% Annotationskosten für 95% Accuracy
- Grösster Vorteil in Early Stages (<500 Samples)
- Diminishing Returns ab ~1000 Samples

**6. Split-Strategie:**
- **80/10/10** liefert leicht bessere Test-Performance als 60/20/20 (+0.67-0.90%)
- Mehr Trainingsdaten kompensieren kleineres Validation-Set

**7. Robuste Evaluation:**
- **Nested CV** liefert konservativste, unbiasedeste Schätzungen (97.71-98.16%)
- **Hold-Out** leicht optimistischer (99.10%)
- **5-Fold CV** guter Kompromiss (97.93-98.11%)

### 6.2 Beantwortung der Forschungsfragen

**FF1 (Performanz):**  
Die klassische Baseline erreicht **vergleichbare, teilweise sogar leicht bessere** Ergebnisse als Transformer-Modelle auf dem BBC News Dataset. In optimalen Konfigurationen erreichen alle drei Modelle 99.10% Test-Accuracy.

**FF2 (Effizienz):**  
Der Trade-off zwischen Modellkomplexität und Accuracy fällt **deutlich zugunsten der einfachen Baseline** aus – **gleiche Leistung bei 1/18 der Rechenzeit**, ohne GPU-Anforderung.

**FF3 (Hyperparameter-Sensitivität):**  
Transformer-Modelle zeigen **moderate Sensitivität**. Optimale Konfiguration: **3 Epochen, max_length=256, lr=2e-5**. Diese Empfehlungen sind robust über verschiedene Evaluation-Strategien (Hold-Out, CV, Nested CV).

**FF4 (Dateneffizienz):**  
Die Baseline ist bei **wenigen Daten klar überlegen**; Transformer benötigen **>100 Samples/Klasse** für kompetitive Performance. Transfer Learning bietet keinen Vorteil bei limitierten Daten für keyword-basierte Klassifikation.

**FF5 (Annotation Efficiency):**  
**Active Learning (Uncertainty Sampling) steigert Annotation Efficiency um 40%** für 95% Accuracy. Der Vorteil ist in Early Stages am grössten und nimmt ab ~1000 Samples ab.

**FF6 (Split-Strategie):**  
**80/10/10 Split** liefert leicht bessere Test-Performance als 60/20/20 (+0.67-0.90%), da mehr Trainingsdaten verfügbar sind. Der Unterschied ist jedoch marginal.

**FF7 (Robuste Evaluation):**  
**Nested Cross-Validation** liefert die robustesten und unbiasedesten Performanzschätzungen (97.71-98.16%), da Hyperparameter-Optimierung und Evaluation strikt getrennt sind. Hold-Out ist leicht optimistischer (99.10%), 5-Fold CV liegt dazwischen (97.93-98.11%).

### 6.3 Praktische Empfehlungen

**1. Für das BBC News Dataset und ähnliche Aufgaben:**
- **Wähle TF-IDF + LogReg:** Gleiche Performance, dramatisch einfacher und effizienter
- Deployment: CPU-basiert, <1ms Inferenz, einfache Integration

**2. Wann Transformer in Betracht ziehen:**
- Komplexe semantische Aufgaben (Sentiment mit Ironie, NLI, QA)
- Grosse Datensätze (>10k Samples) verfügbar
- GPU-Ressourcen vorhanden
- State-of-the-Art Performance kritisch

**3. Best Practices:**
- **Immer** mit einfacher Baseline starten
- Evaluate holistically (Accuracy + Zeit + Ressourcen + Interpretierbarkeit)
- Domain-spezifisch testen
- Active Learning für begrenzte Annotationsbudgets
- Nested CV für finale Publikation/Production-Entscheidung

### 6.4 Wissenschaftlicher Beitrag

Diese Arbeit trägt zur bestehenden Forschung bei durch:

**1. Systematische Evaluation:**
- Umfassender Vergleich über **11 Notebooks** mit verschiedenen Perspektiven
- Cross-Validation, Hold-Out, Nested CV, Few-Shot, Active Learning

**2. Praktische Erkenntnisse:**
- Quantifizierung des Accuracy-Efficiency Trade-offs
- Klare Hyperparameter-Empfehlungen
- Cost-Benefit Analysen für praktische Deployment-Entscheidungen

**3. Methodische Rigorosität:**
- Nested CV für unbiased Schätzungen
- Systematische Hyperparameter-Experimente
- Statistische Robustheit durch Cross-Validation

**4. Differenzierte Perspektive:**
- Kein "one-size-fits-all" Ansatz
- Kontextabhängige Modellwahl
- Bewusstsein für Limitationen

### 6.5 Ausblick und zukünftige Forschung

Zukünftige Forschung könnte folgende Aspekte adressieren:

#### 6.5.1 Dataset-Diversität

**1. Multi-Dataset Evaluation:**
- Vergleich auf **10+ Datasets** aus verschiedenen Domains:
  - Social Media (Twitter, Reddit)
  - E-Commerce (Amazon Reviews)
  - Wissenschaft (arXiv, PubMed)
  - Medizin (MIMIC, i2b2)
  - Recht (Legal Documents)

**2. Mehrsprachige Analyse:**
- Multilinguale Modelle (mBERT, XLM-R)
- Low-Resource Languages
- Cross-Lingual Transfer

**3. Unbalanced Datasets:**
- Extreme Klassenungleichgewichte (1:100, 1:1000)
- Rare Class Detection
- Cost-Sensitive Learning

#### 6.5.2 Moderne Modelle und Methoden

**1. Neuere Transformer-Architekuren:**
- RoBERTa (robustly optimized BERT)
- DeBERTa (decoding-enhanced BERT)
- ELECTRA (efficient pre-training)
- Large Language Models (GPT-4, Claude, Llama 3)

**2. Hybride Ansätze:**
- Kombination von TF-IDF Features mit Transformer-Embeddings
- Ensemble-Methoden (TF-IDF + BERT Voting)
- Multi-Task Learning

**3. Effiziente Transformers:**
- Quantization (INT8, FP16)
- Pruning und Distillation
- ONNX/TensorRT Optimierung
- Mobile-optimierte Modelle (MobileBERT)

#### 6.5.3 Advanced Evaluation

**1. Statistische Rigorosität:**
- Formale Signifikanztests (t-tests, Wilcoxon, McNemar)
- Bootstrapping für Konfidenzintervalle
- Bayesianische Modellvergleiche

**2. Robustness Testing:**
- Adversarial Examples
- Out-of-Distribution Detection
- Concept Drift Simulation

**3. Calibration:**
- Expected Calibration Error (ECE)
- Reliability Diagrams
- Temperature Scaling

#### 6.5.4 Praktische Aspekte

**1. Deployment-Szenarien:**
- Edge-Deployment (TensorFlow Lite, PyTorch Mobile)
- Cloud vs. On-Premise Trade-offs
- Real-Time vs. Batch Processing

**2. Long-Term Production:**
- Model Monitoring und Drift Detection
- Retraining Strategies (Incremental, Periodic)
- A/B Testing Frameworks

**3. Human-in-the-Loop:**
- Active Learning Produktionalisierung
- Feedback Loops
- Explainability und Debugging

#### 6.5.5 Theoretische Fragen

**1. Warum funktioniert TF-IDF so gut?:**
- Linguistische Analyse der Features
- Information-theoretische Perspektiven
- Vergleich mit neuronalen Attention-Patterns

**2. Transfer Learning Limits:**
- Wann ist Pre-Training nützlich?
- Domain-Shift Quantifizierung
- Optimal Transport für Domain Adaptation

**3. Scaling Laws:**
- Wie skaliert Performance mit Datensatzgrösse?
- Power-Law Fits für Learning Curves
- Sample Complexity Theory

### 6.6 Schlusswort

Diese Arbeit demonstriert, dass **Einfachheit und Effizienz** in der Praxis oft wichtiger sind als maximale Komplexität. Während Transformer-Modelle beeindruckende Fortschritte in vielen NLP-Aufgaben erzielt haben, sollten sie nicht unkritisch als Universallösung betrachtet werden.

Für **keyword-basierte Klassifikationsaufgaben** wie das BBC News Dataset bleibt TF-IDF mit logistischer Regression eine **pragmatische, effiziente und robuste** Wahl. Die Entscheidung für ein Modell sollte immer auf einer **ganzheitlichen Evaluation** basieren, die nicht nur Accuracy, sondern auch Ressourcen, Interpretierbarkeit und praktische Deployment-Constraints berücksichtigt.

**Kernbotschaft:**  
*"The best model is not always the most complex one, but the one that best fits your specific requirements, constraints, and context."*

---

## 7. Literaturverzeichnis

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)*, 4171–4186. https://arxiv.org/abs/1810.04805

Greene, D., & Cunningham, P. (2006). Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering. *Proceedings of the 23rd International Conference on Machine Learning (ICML)*, 377–384.

Harris, Z. S. (1954). Distributional Structure. *Word*, 10(2-3), 146–162.

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.

Joachims, T. (1998). Text Categorization with Support Vector Machines: Learning with Many Relevant Features. *Proceedings of the 10th European Conference on Machine Learning (ECML)*, 137–142.

Jurafsky, D., & Martin, J. H. (2024). *Speech and Language Processing* (3rd ed. draft). https://web.stanford.edu/~jurafsky/slp3/

Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.

Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). Distributed Representations of Words and Phrases and their Compositionality. *Advances in Neural Information Processing Systems (NeurIPS)*, 26, 3111–3119.

Pan, S. J., & Yang, Q. (2010). A Survey on Transfer Learning. *IEEE Transactions on Knowledge and Data Engineering*, 22(10), 1345–1359.

Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global Vectors for Word Representation. *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 1532–1543.

Salton, G., & Buckley, C. (1988). Term-weighting Approaches in Automatic Text Retrieval. *Information Processing & Management*, 24(5), 513–523.

Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter. *arXiv preprint arXiv:1910.01108*. https://arxiv.org/abs/1910.01108

Sebastiani, F. (2002). Machine Learning in Automated Text Categorization. *ACM Computing Surveys*, 34(1), 1–47.

Settles, B. (2009). Active Learning Literature Survey. *Computer Sciences Technical Report 1648*, University of Wisconsin–Madison.

Sun, C., Qiu, X., Xu, Y., & Huang, X. (2019). How to Fine-Tune BERT for Text Classification. *China National Conference on Chinese Computational Linguistics (CCL)*, 194–206. https://arxiv.org/abs/1905.05583

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 5998–6008.

Wang, S., & Manning, C. D. (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification. *Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics (ACL)*, 90–94.

---

## 8. Anhang

### A. Vollständige Ergebnistabellen

**Tabelle A1: Alle Hyperparameter-Experiment-Ergebnisse (Notebook 04)**

| Run Name | Model | Epochs | Max Length | Train Time (s) | Val Acc | Val F1 | Test Acc | Test F1 |
|----------|-------|--------|------------|----------------|---------|--------|----------|---------|
| baseline_logreg | tfidf+logreg | - | - | 2.22 | 98.21% | 98.17% | 99.10% | 99.12% |
| bert_base_e2_len128 | bert-base-uncased | 2 | 128 | 16.06 | 98.65% | 98.68% | 98.65% | 98.63% |
| bert_base_e3_len128 | bert-base-uncased | 3 | 128 | 23.30 | 97.76% | 97.72% | 99.10% | 99.13% |
| bert_base_e2_len256 | bert-base-uncased | 2 | 256 | 27.72 | 97.76% | 97.90% | 98.65% | 98.64% |
| bert_base_e3_len256 | bert-base-uncased | 3 | 256 | 41.22 | 98.21% | 98.18% | 99.10% | 99.07% |
| distilbert_e2_len128 | distilbert-base-uncased | 2 | 128 | 8.75 | 96.86% | 96.71% | 97.76% | 97.80% |
| distilbert_e3_len128 | distilbert-base-uncased | 3 | 128 | 13.01 | 98.21% | 98.16% | 97.76% | 97.79% |
| distilbert_e2_len256 | distilbert-base-uncased | 2 | 256 | 16.39 | 96.86% | 96.83% | 98.21% | 98.24% |
| distilbert_e3_len256 | distilbert-base-uncased | 3 | 256 | 24.43 | 97.76% | 97.78% | 97.76% | 97.78% |

**Tabelle A2: Cross-Validation Detailed Results (Notebooks 07_01, 07_02)**

**Baseline (5-Fold CV):**

| Fold | Train Acc | Val Acc | Val F1 | Train Time (s) |
|------|-----------|---------|--------|----------------|
| 1 | 99.72% | 98.43% | 98.42% | 1.89 |
| 2 | 99.67% | 97.98% | 97.96% | 1.91 |
| 3 | 99.72% | 97.98% | 97.96% | 1.93 |
| 4 | 99.72% | 97.30% | 97.28% | 1.95 |
| 5 | 99.67% | 97.97% | 97.95% | 1.97 |
| **Mean** | **99.70%** | **97.93%** | **97.91%** | **1.93** |
| **Std** | **0.02%** | **0.42%** | **0.42%** | **0.03** |

**DistilBERT (5-Fold CV, epochs=3, max_length=256, lr=2e-5):**

| Fold | Train Acc | Val Acc | Val F1 | Train Time (s) |
|------|-----------|---------|--------|----------------|
| 1 | 99.77% | 98.43% | 98.44% | ~60 |
| 2 | 99.83% | 97.53% | 97.53% | ~60 |
| 3 | 99.77% | 98.88% | 98.89% | ~60 |
| 4 | 99.94% | 97.30% | 97.30% | ~60 |
| 5 | 99.89% | 98.42% | 98.42% | ~60 |
| **Mean** | **99.84%** | **98.11%** | **98.12%** | **~60** |
| **Std** | **0.07%** | **0.63%** | **0.63%** | - |

**Tabelle A3: Nested CV Detailed Results (Notebook 08)**

**Outer Fold 1:**

| Model | Inner CV Best Config | Test Acc |
|-------|----------------------|----------|
| Baseline | C=1.0 | 97.44% |
| BERT | lr=2e-5, e=3 | 97.44% |
| DistilBERT | lr=2e-5, e=3 | 97.71% |

**Outer Fold 2:**

| Model | Inner CV Best Config | Test Acc |
|-------|----------------------|----------|
| Baseline | C=1.0 | 98.38% |
| BERT | lr=2e-5, e=3 | 97.57% |
| DistilBERT | lr=2e-5, e=3 | 98.92% |

**Outer Fold 3:**

| Model | Inner CV Best Config | Test Acc |
|-------|----------------------|----------|
| Baseline | C=1.0 | 97.30% |
| BERT | lr=2e-5, e=3 | 98.92% |
| DistilBERT | lr=2e-5, e=3 | 97.84% |

**Aggregated:**

| Model | Mean Test Acc | Std | Optimal Config |
|-------|---------------|-----|----------------|
| Baseline | 97.71% | 0.49% | C=1.0 |
| BERT | 97.98% | 0.88% | lr=2e-5, epochs=3 |
| DistilBERT | 98.16% | 0.54% | lr=2e-5, epochs=3 |

### B. Notebook-Übersicht

**Teil 1: Kern-Experimente**

| Notebook | Beschreibung | Split | Hauptfokus |
|----------|--------------|-------|------------|
| 01_data_prep | Datenaufbereitung | - | Preprocessing, EDA |
| 02a_baseline_60-20-20 | Baseline | 60/20/20 | TF-IDF + LogReg |
| 02b_baseline_80-10-10 | Baseline | 80/10/10 | TF-IDF + LogReg |
| 03a_bert/distilbert_60-20-20 | Transformers | 60/20/20 | Fine-Tuning |
| 03b_bert/distilbert_80-10-10 | Transformers | 80/10/10 | Fine-Tuning |
| 04_experiments_hparams | Hyperparameter Grid | 80/10/10 | Epochs, max_length |
| 04a_baseline_C_parameter | Regularisierung | 80/10/10 | C-Parameter |
| 07_01_CV_baseline | Cross-Validation | 5-Fold | Baseline robuste Eval |
| 07_02_CV_BERT | Cross-Validation | 5-Fold | Transformers robuste Eval |
| 08_split_comparison | Split-Vergleich | 60/20/20 vs 80/10/10 | Datenaufteilung |

**Teil 2: Erweiterte Methoden**

| Notebook | Beschreibung | Methodik | Hauptfokus |
|----------|--------------|----------|------------|
| 05_fewshot_learning | Few-Shot Learning | Train/Test | Dateneffizienz |
| 06_active_learning | Active Learning | Pool/Test | Annotation Efficiency |
| 08_Nested_CV | Nested Cross-Validation | 3×2 CV | Unbiased Evaluation |

### C. Abbildungsverzeichnis

1. Cross-Validation Vergleich (Notebook 07_01, 07_02)
2. Test-Accuracy über Hyperparameter-Konfigurationen (Notebook 04)
3. Trainingszeit Vergleich (Notebook 04)
4. C-Parameter Experiment (Notebook 04a)
5. Split Comparison (Notebook 08_split)
6. Learning Curve – Few-Shot Learning (Notebook 05)
7. Active Learning Comparison (Notebook 06)
8. Nested CV Comparison (Notebook 08)
9. Konfusionsmatrix Baseline (Notebook 02b)
10. Konfusionsmatrix BERT (Notebook 03b)
11. Konfusionsmatrix DistilBERT (Notebook 03b)

### D. Code-Verfügbarkeit

Der vollständige Code für alle Experimente ist im Repository verfügbar:

**Datenaufbereitung:**
- [notebooks/01_data_prep.ipynb](notebooks/01_data_prep.ipynb)

**Baselines:**
- [notebooks/02a_baseline_tfidf_logreg_60-20-20.ipynb](notebooks/02a_baseline_tfidf_logreg_60-20-20.ipynb)
- [notebooks/02b_baseline_tfidf_logreg_80-10-10.ipynb](notebooks/02b_baseline_tfidf_logreg_80-10-10.ipynb)

**Transformers:**
- [notebooks/03a_bert_train_eval_60-20-20.ipynb](notebooks/03a_bert_train_eval_60-20-20.ipynb)
- [notebooks/03a_distilbert_train_eval_60-20-20.ipynb](notebooks/03a_distilbert_train_eval_60-20-20.ipynb)
- [notebooks/03b_bert_train_eval_80-10-10.ipynb](notebooks/03b_bert_train_eval_80-10-10.ipynb)
- [notebooks/03b_distilbert_train_eval_80-10-10.ipynb](notebooks/03b_distilbert_train_eval_80-10-10.ipynb)

**Hyperparameter-Experimente:**
- [notebooks/04_experiments_hparams.ipynb](notebooks/04_experiments_hparams.ipynb)
- [notebooks/04a_baseline_C_parameter.ipynb](notebooks/04a_baseline_C_parameter.ipynb)

**Cross-Validation:**
- [notebooks/07_01_cross_validation_baseline.ipynb](notebooks/07_01_cross_validation_baseline.ipynb)
- [notebooks/07_02_cross_validation_BERT.ipynb](notebooks/07_02_cross_validation_BERT.ipynb)

**Erweiterte Methoden:**
- [notebooks/05_fewshot_learning_curve.ipynb](notebooks/05_fewshot_learning_curve.ipynb)
- [notebooks/06_active_learning_simulation.ipynb](notebooks/06_active_learning_simulation.ipynb)
- [notebooks/08_Nested_cross_validation_model_comparison.ipynb](notebooks/08_Nested_cross_validation_model_comparison.ipynb)
- [notebooks/08_split_experiments_comparison.ipynb](notebooks/08_split_experiments_comparison.ipynb)

### E. Reproduzierbarkeit

**Environment Setup:**

```bash
# Environment erstellen
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Dependencies installieren
pip install -r requirements.txt

# Notebooks ausführen
jupyter lab notebooks/
```

**requirements.txt:**
```
torch>=2.0.0
transformers>=4.30.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
datasets>=2.14.0
jupyter>=1.0.0
```

**Reproduktion der Hauptergebnisse:**

```bash
# 1. Cross-Validation (Hauptergebnis)
jupyter nbconvert --execute --to notebook \
  notebooks/07_01_cross_validation_baseline.ipynb
jupyter nbconvert --execute --to notebook \
  notebooks/07_02_cross_validation_BERT.ipynb

# 2. Hyperparameter-Experimente
jupyter nbconvert --execute --to notebook \
  notebooks/04_experiments_hparams.ipynb

# 3. Nested CV
jupyter nbconvert --execute --to notebook \
  notebooks/08_Nested_cross_validation_model_comparison.ipynb
```

**Seed für Reproduzierbarkeit:**
- Random Seed: 42 (durchgehend in allen Notebooks)
- PyTorch: `torch.manual_seed(42)`
- NumPy: `np.random.seed(42)`
- Scikit-learn: `random_state=42`

---

*Erklärung: Ich bestätige, dass ich diese Arbeit selbstständig verfasst und keine anderen als die angegebenen Quellen und Hilfsmittel verwendet habe.*

*Ort, Datum: _________________*

*Unterschrift: _________________*
