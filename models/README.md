# AI Models

## Natureness Image Recognition

AI guesses a value between 0 and 1 to determine how natural an image is. 0 is not natural at all, 1 is very natural. The model is trained on a dataset of 6000+ annotated images, each labeled with a value 0 to 1 (not uniformly distributed).

## Model Comparison

Following criteria are used to compare models:

- Model type: can be `nature`, `urban` or `all`. `nature` models are trained on a dataset with more natural images, `urban` models are trained on a dataset with less natural images, `all` models are trained on the whole dataset (with train/test splits of course).
- Dataset type: models are evaluated on `nature`, `urban` and `all` datasets.
- Percentile: `nature` and `urban` datasets are split according to a percentile threshold. For example, the `nature` and `urban` dataset variations are even with 0.5 percentile threshold, while with a lower or higher threshold, the dataset variations are not even.
- Uniformity: After splitting by a percentile threshold, there is still a chance for an `urban` datapoint to be in the `nature` dataset and vice versa. The uniformity score determines, how uniform the dataset is after splitting by a percentile threshold. With perfect uniformity, the percentile threshold would not matter.

## Processing model losses

```python
import pandas as pd

df = pd.read_csv('model_losses.csv')
# One column
# p30_u10 - configuration key: Percentile 30, Uniformity 10
# nature - model type: nature, urban or all
# all - dataset type: nature, urban or all
# 0.6424242854118347 - loss value for picture 0
# 15 more float values for 15 more pictures
```

Configuration key does not matter for `all` type models, as they are trained on the whole dataset.

Output pictures are available [here](https://drive.google.com/file/d/1-Nyx5lqfkjnDmfgteBVBzgUPqAjflabv/view?usp=sharing).
