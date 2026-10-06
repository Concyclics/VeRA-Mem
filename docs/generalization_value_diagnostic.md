# Addressing and writer diagnostics after data augmentation

This development diagnostic identifies two specific problems: entity addressing for unseen questions remains near random, and values learned with ordinary multi-template augmentation strongly encode template differences. Changing the wording of an observation can even reverse a same-fact value's direction. Cross-expression consistency reduces those differences somewhat, but they still dominate value variance and do not restore addressing on unseen development questions.

Evidence is preserved in the [aggregate diagnostic JSON](results/generalization/value_diagnostic.json); the reproduction script is [diagnose_generalization_values.py](../scripts/diagnose_generalization_values.py). We compare the best checkpoints of `canonical4096`, `augment4096`, and `invariant4096`, all selected at step 1536. The third arm adds Q/K/value cross-expression consistency with weight 0.1 to ordinary augmentation. Statistics, prototypes, and diagnostic heads use training and development data only. Confirmation features, answers, and predictions are not used for diagnosis or next-step selection. When adding the third arm, all three checkpoints were rerun with the original script and identical parameters. Every numeric result for the first two arms reproduced exactly; the aggregate records provenance hashes for the unified run.

## 1. Addressing has not transferred to new questions

Development contains 64 entities unseen during training. Each retrieval bank holds the corresponding observations for those 64 entities. Values below are Recall@1 from frozen Q/K encoders on actual layer-input features, not language-model generation accuracy. Random top-1 retrieval is 1/64, or 1.56%.

| Development condition | Single-template training | Ordinary augmentation | Augmentation + consistency |
|---|---:|---:|---:|
| Original question + original observation format | 64/64 | 55/64 | 52/64 |
| Request-card question + original observation format | 3/64 | 2/64 | 1/64 |
| Leading-output-constraint question + original observation format | 2/64 | 2/64 | 3/64 |
| Request-card question + record-card observation | 1/64 | 2/64 | 2/64 |
| Leading-output-constraint question + new observation wording | 1/64 | 1/64 | 1/64 |

Neither ordinary augmentation nor consistency at the current strength reliably transfers entity matching to development questions. Adding more seen question forms is not equivalent to learning transferable addressing representations.

## 2. The writer reverses directions across training expressions

We separately encode observation expressions of the same training fact as values and perform a balanced template×answer variance decomposition over the complete training data. The template main effect is each template's value-mean offset from the global mean; the answer main effect is the corresponding answer-mean offset. Residuals include entity differences and answer–template interactions, not merely noise.

| Writer training condition | Template main effect | Answer main effect | Residual |
|---|---:|---:|---:|
| Single template | 0% (zero by definition) | 92.39% | 7.61% |
| Four observation expressions | 84.75% | 8.19% | 7.07% |
| Four observation expressions + consistency | 81.43% | 10.75% | 7.82% |

For the augmented model, mean same-fact value cosine between `train_support_00` and `train_support_02` is **−0.875**. The former uses “The assigned memory word ...”; the latter uses “Store this association ...”. Cosines between the original expression and the other two training expressions are **+0.932 and +0.940**. With consistency, 00/02 cosine is **−0.814**, still showing a strong reversal. Template-mean direction differences dominate value variance rather than producing only small numerical perturbations.

Effective value rank across all training expressions is 2.25 for augmentation, 2.60 for consistency, and 8.67 for the single-template model. Rank is estimated from the centered spectrum of at most 1024 uniformly sampled rows. The variance decomposition above instead uses all training rows and is more direct evidence of template dominance. Low effective rank alone does not establish memory failure.

For all three arms, the 64 values under each new development observation format are not exactly identical, but mean pairwise cosine is 0.918–0.970. For the augmented model, the second new observation wording has mean cosine −0.319 with the same fact's original-format value; the consistency arm gives −0.289. Checking value uniqueness alone is therefore insufficient to validate cross-expression compatibility.

## 3. Linear diagnostic heads measure transfer of a particular readout rule

We separately train 16-class linear heads to identify the answer word. These heads are not part of VeRA, are not written to the VDB, and do not participate in actual inference. All use training-fact labels, a fixed 256 steps × 128 examples, and Adam 0.001, with no step selection on development performance. Normalization centers come only from the corresponding training features. Random answer classification is 4/64, or 6.25%.

| Diagnostic input and training expressions | Original observation | New record card | New observation wording |
|---|---:|---:|---:|
| Single-template model's writer output | 64/64 | 4/64 | 7/64 |
| Augmented model's writer output | 62/64 | 11/64 | 14/64 |
| Augmentation + consistency writer output | 63/64 | 6/64 | 13/64 |
| Frozen backbone layer input, head trained on one template | 64/64 | 16/64 | 5/64 |
| Frozen backbone layer input, head trained on four templates | 64/64 | 6/64 | 9/64 |

Original-format writer outputs still contain answer information decodable by another linear head. These learned readout rules transfer poorly to new observation formats. **Low scores do not prove that hidden representations lack answer information**: head training, linear capacity, normalization, and limited budgets can all affect the result. Single- and multi-template heads also have different training-feature pool sizes despite matched updates and example exposures. In particular, a linear head's 64/64 is not VeRA generation accuracy of 64/64.

Cosine classification with training-answer prototypes points in the same direction. For the augmented writer, accuracy is 53/64 on the original format and 7/64 and 11/64 on the two new formats. Subtracting the training-value mean first yields 55/64, 9/64, and 11/64. The consistency arm gives 51/64, 9/64, and 7/64 before centering, and 54/64, 9/64, and 7/64 afterward. Removing one common mean does not restore cross-expression readout.

## 4. Implications for subsequent experiments

The current consistency term reduces the template main effect from 84.75% to 81.43%, without eliminating the dominant template differences. Addressing for unseen development questions remains near random, and additional linear heads do not consistently improve on new observation formats. These findings support the limited conclusion that weak constraints have not produced a memory interface compatible across expressions. They establish neither solved generalization nor failure of every consistency-training approach.

Future interventions need to satisfy three conditions together: substantially reduce incompatible value directions across training expressions, preserve discriminating addresses with other entities as negatives, and improve actual generation under sparse retrieval. Geometry or extra diagnostic-head gains cannot establish success alone. Confirmation generation scores were not used to adjust the consistency weight here, and no additional model experiments were run.

These conclusions concern 64 development entities, 16 shared answer words, and one training seed, and diagnose the current implementation. If the implementation changes based on them, confirmation must retain its independent role. Development improvements or historical paraphrases already used in training cannot be relabeled unseen-template generalization.
