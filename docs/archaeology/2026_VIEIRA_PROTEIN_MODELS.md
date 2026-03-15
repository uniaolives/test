# Intrinsic dataset features drive mutational effect prediction by protein language models
**Luiz C. Vieira, Sophia Lin, and Claus O. Wilke**
*March 8, 2026*

## Abstract
Protein language models (pLMs) are commonly used for predicting protein fitness landscapes, but their wide range of performance across datasets remains poorly understood. We evaluated supervised transfer learning on 41 viral and 33 cellular deep-mutational-scanning (DMS) datasets using embeddings from multiple pLMs. We observed consistently lower predictive performance on viral datasets compared to cellular datasets, independent of model architecture or transfer learning strategy. Surprisingly, a simple baseline model that predicted site mean fitness matched or outperformed supervised models on many datasets, highlighting the dominant role of site effects. Analysis of site variability using two metrics, relative variability of site means (RVSM) and fraction of highly variable sites (FHVS), revealed that patterns of fitness variation within and among sites constrain model performance and largely explain the observed differences between viral and cellular datasets. Moreover, splitting training and test data by site, rather than pooling, revealed that supervised models often rely on site effects rather than capturing broader mutational patterns. These findings highlight limitations of current pLMs for mutational effect prediction and suggest that dataset composition, rather than model architecture or training, is the primary driver of predictive success.

---

## Significance Statement
Mutational effects prediction with protein language models tends to vary widely in prediction accuracy, depending on the dataset considered. While poor performance is commonly equated with poor model quality, we show here that intrinsic dataset features, such as the variability of fitness values within and among sites, are critical predictors of model performance. Moreover, we show that many existing benchmarks overestimate model performance, by allowing training data to leak into the test set. In fact, in many cases, protein language models barely outperform a naive predictor relying entirely on mean fitness values at individual sites. In aggregate, our study reveals that protein language models fail to capture site mutational constraints critical for fitness prediction, despite claims of learning local sequence context.

---

## Summary of Findings
1. **Low Performance on Viral Data:** Consistently worse performance on viral proteins compared to cellular ones.
2. **Site Effects Dominate:** Lasso regression on pLM embeddings often fails to beat a simple average of fitness effects at each site.
3. **Variability Metrics:** RVSM and FHVS explain the performance gap. Viral data lacks highly variable sites.
4. **Data Leakage:** Pooled splits inflate performance estimates by allowing site-specific information to leak from training to test sets. Site-stratified splits show a massive drop in accuracy across all models (ESM-2, ESM C, etc.).
5. **Architectural Limits:** Even finetuning (LoRA) fails to overcome the generalization challenge of site-stratified splits.
6. **ESM C Performance:** ESM C performs exceptionally well on cellular data but poorly on viral data, likely due to the exclusion of viral sequences during pretraining for "safety concerns."
