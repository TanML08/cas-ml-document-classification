# Ergebnis-Zusammenfassung: Document Classification

## Hauptergebnis: 5-Fold Stratified Cross-Validation

| Rang | Modell | Accuracy | Macro-F1 | Zeit (gesamt) |
|------|--------|----------|----------|---------------|
| 🥇 1 | **BERT-base** | **98.34% ± 0.58%** | **98.33% ± 0.60%** | ~230s |
| 🥈 2 | DistilBERT | 98.11% ± 0.72% | 98.11% ± 0.71% | ~135s |
| 🥉 3 | TF-IDF + LogReg | 97.93% ± 0.72% | 97.90% ± 0.73% | ~10s |

**Fazit:** BERT-base erzielt die beste Performance, jedoch mit erheblich höherem Rechenaufwand.

---

## Detaillierte Fold-Ergebnisse

### BERT-base (5-Fold CV)
| Fold | Accuracy | Macro-F1 | Zeit |
|------|----------|----------|------|
| 1 | 97.75% | 97.67% | 45.2s |
| 2 | 98.65% | 98.65% | 45.6s |
| 3 | 97.98% | 98.02% | 45.7s |
| 4 | 99.33% | 99.33% | 46.4s |
| 5 | 97.98% | 97.97% | 46.8s |

### DistilBERT (5-Fold CV)
| Fold | Accuracy | Macro-F1 | Zeit |
|------|----------|----------|------|
| 1 | 97.08% | 97.03% | 27.4s |
| 2 | 98.88% | 98.91% | 26.9s |
| 3 | 97.98% | 98.01% | 27.0s |
| 4 | 98.88% | 98.83% | 26.8s |
| 5 | 97.75% | 97.75% | 26.5s |

### TF-IDF + LogReg (5-Fold CV)
| Fold | Accuracy | Macro-F1 | Zeit |
|------|----------|----------|------|
| 1-5 | ~97.93% | ~97.90% | ~2.0s |

---

## Experimentelle Parameter

### Baseline (TF-IDF + Logistic Regression)
| Parameter | Wert |
|-----------|------|
| max_features | 50,000 |
| ngram_range | (1, 2) |
| stop_words | english |
| C (Regularization) | 1.0 (default) |
| max_iter | 2,000 |

### Transformer (BERT / DistilBERT)
| Parameter | Wert |
|-----------|------|
| Epochs | 3 |
| Learning Rate | 2e-5 |
| Batch Size | 16 |
| Max Length | 256 |
| Optimizer | AdamW |

### Cross-Validation
| Parameter | Wert |
|-----------|------|
| Methode | Stratified K-Fold |
| K (Folds) | 5 |
| Shuffle | True |
| Random State | 42 |

---

## Generierte Plots

| Datei | Beschreibung |
|-------|-------------|
| `cv_comparison_all_models.png` | Balkendiagramm: Alle 3 Modelle (Acc + F1) |
| `cv_transformer_accuracy_boxplot.png` | Boxplot: BERT vs DistilBERT |
| `cv_transformer_confusion_matrices.png` | Confusion Matrix: BERT + DistilBERT |
| `cv_baseline_confusion_matrix.png` | Confusion Matrix: Baseline |

---

## Interpretation

1. **Performance-Unterschiede sind gering** (~0.4% zwischen Rang 1 und 3)
   - Der BBC News Datensatz ist relativ "einfach" mit gut separierbaren Klassen

2. **BERT-base vs. DistilBERT**
   - BERT ist ~0.2% besser, aber ~70% langsamer
   - DistilBERT bietet guten Trade-off zwischen Performance und Effizienz

3. **Baseline ist überraschend stark**
   - TF-IDF + LogReg erreicht 97.93% – nur 0.4% unter BERT
   - Für einfache Klassifikationsaufgaben oft ausreichend

4. **Empfehlung für Praxis**
   - Einfache Aufgaben: TF-IDF + LogReg (schnell, interpretierbar)
   - Komplexe Aufgaben: BERT/DistilBERT (bessere Generalisierung)

---

*Generiert am: 2026-01-26*
