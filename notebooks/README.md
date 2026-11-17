# Teaching notebooks

A guided walk through the project for someone new to it: what was measured, how, and why each
choice was made. They're written to be read on GitHub (all outputs included). Each one ends with
general takeaways that apply beyond this project.

| # | notebook | what it covers |
|---|---|---|
| 1 | [The question and the data](01_question_and_data.ipynb) | the lakes; why TWDB "area" is really level pushed through a rating curve; how often clouds block the view |
| 2 | [Pixels and water indices](02_pixels_and_indices.ipynb) | spectra of water and land; NDWI vs MNDWI; how mapped area depends on the cutoff, and how the curves move between dates |
| 3 | [Labels](03_labels.ipynb) | weak labels for training vs hand labels for testing; an AI labeller vs a person, and why "unsure" isn't free; stratified point sampling and why it needs weights |
| 4 | [Baselines and honest splits](04_baselines_and_splits.ipynb) | splitting by lake and year; tuning baselines under the same rules; how the labels flipped NDWI from worst to best |
| 5 | [The model](05_model.ipynb) | U-Net with a TorchGeo pretrained encoder; training logs; ensemble uncertainty; distilling a radar model from the optical one; ablations |
| 6 | [Evaluation and uncertainty](06_evaluation_and_uncertainty.ipynb) | paired comparisons; why a one-line NDWI threshold beats the U-Net per pixel; the U-Net's pull towards the usual outline; calibration bin by bin; uncertainty as an error detector |
| 7 | [Area, level and radar](07_area_level_radar.ipynb) | lake area over time vs the official record (NDWI ties the U-Net); clouds; radar, the model's clearest win; noise floor vs responsiveness; rating-curve steps |
| 8 | [Transfer to small reservoirs](08_transfer_small_reservoirs.ipynb) | the shipped models on ten small West/Central Texas reservoirs they never saw, scored against TWDB as if ungauged; a modest edge for the U-Net on the smallest, muddiest lakes |
| 9 | [Truly ungauged: stock tanks](09_ungauged_stock_tanks.ipynb) | ponds with no gauge at all, scored against NAIP aerial photos; building and auditing a reference; chance baselines for detection; why the reservoir winner loses on ponds |

## Re-running

The notebooks are saved with their outputs, so they read fine on GitHub without running
anything. To re-run them, run the pipeline first (`make all`, which includes `make transfer` and
`make ponds`, needed by notebooks 8 and 9), then open them in Jupyter:

```
uv run --with jupyterlab jupyter lab notebooks/
```
