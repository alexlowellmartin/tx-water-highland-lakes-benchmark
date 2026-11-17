# tx-water-highland-lakes-benchmark

Mapping the surface area of the Texas Highland Lakes from satellites, and checking it against the
official lake level.

**[Interactive map](https://alo.mn/apps/mapping-texas-lakes-from-space/#travis)** ·
[teaching notebooks](notebooks/README.md)

**Question.** Does a deep-learning model, fine-tuned from public pretrained satellite weights,
map reservoir surface extent better than a simple water-index threshold, including through cloud
with radar? Does the mapped area track the official lake level? And does any of it work on water
nobody gauges?

**Two answer keys.** The 660 test points were labelled twice by design: by an AI model (Claude Opus 5.5)
and, blind, by hand. Every method is scored against both. The keys agree on 97 % of points
but rank the methods differently, because the AI left 99 shoreline points unsure. The hand
labels are the primary reference; `results/ai_labels/` holds the same tables against the AI key.

**About this project.** Most of my geospatial work is vector: parcels, networks, boundaries. This was a
project to learn remote sensing properly by building something end to end, so every baseline
gets the same tuning rules as the model, and the losses are reported as plainly as the wins.

**Study area and period.** Lakes Buchanan, Inks, LBJ, Marble Falls and Travis on the Colorado River
near Austin, January 2018 – October 2026. Travis swung from 704 ft (Oct 2018 flood) to 627 ft
(2023 drought) and back to full in July 2025; Buchanan from 1020 to 992 ft. LBJ, Inks and Marble
Falls are pass-through lakes held near constant level apart from a few 4–8 ft drawdowns. That
makes them a *noise-floor* test: on steady-level days a reliable mapper should report a flat
area, and during drawdowns it should show the drop.

**New to the project?** Start with the [teaching notebooks](notebooks/README.md). They walk
through the data, the labels, the baselines, the model and the evaluation, explaining why each
choice was made and what went wrong along the way.

## Results in brief

| Question | Answer |
|---|---|
| Is the U-Net more accurate than an index, pixel by pixel? | **No: NDWI is.** On 585 held-out hand points, a tuned NDWI threshold scores IoU 0.994 against 0.968 for the U-Net ensemble; paired difference +0.026 [95 % CI +0.011, +0.042]. An NDWI cutoff picked per scene with no labels at all (Otsu) does as well. The U-Net is probably ahead of tuned MNDWI (0.968 vs 0.939; +0.029 [−0.000, +0.065]). Against the AI answer key, NDWI ranks last instead (see *The hand labels*). |
| Does it measure the lake better? | **Not better than NDWI.** Across 121 clear dates at Travis, NDWI's area tracks level with r = 0.994 against 0.990 for the U-Net, with a slightly smaller error after a linear fit (328 vs 364 acres); MNDWI trails (0.972, 727 acres). At Buchanan all three are tied (r 0.992–0.993). On the three near-constant lakes NDWI is as steady as the U-Net on steady-level days (spread 1.2–3.5 % vs 1.5–3.5 %), **and it shows more of the official area drop during small drawdowns**. The U-Net is slow to call freshly exposed lakebed land. |
| Does radar work through cloud? | **Yes, and this is the model's clearest win.** The S1 U-Net reaches IoU 0.948 on the points, +0.08 [0.03, 0.13] over a tuned VV-backscatter threshold. Its area tracks level at r = 0.988 at both Travis and Buchanan on all ~250 radar dates, and its median area error there is 3–7 % on dates with no clear optical image within ±3 days. Optical maps on cloudy dates are off by a median of 10–12 % there (13–41 % on the small lakes). Adding radar raises the share of months with a usable map from 70–90 % to 100 %. |
| Is the uncertainty meaningful? | **Yes.** ECE is 0.006–0.007 for the deep ensemble and MC dropout, and the 57 points with intermediate probabilities show no detectable miscalibration. Ensemble disagreement separates wrong points from right ones with AUROC 0.95. An index threshold gives no such signal. |
| Does it work on small reservoirs it never saw? | **Yes, slightly better than the best index.** On ten small West/Central Texas reservoirs (285–7,239 acres), mapped with no retraining and no local tuning and scored against TWDB as if ungauged, the S2 U-Net's median area error is 6.0 %, against 7.5 % for NDWI and 8.1 % for MNDWI. It is closer than NDWI on most dates on four of the five smallest lakes and on muddy floodwater; NDWI is closer on the four largest. After a linear fit it tracks change better than NDWI on 8 of 10 lakes (p = 0.11) and than MNDWI on 9 of 10 (p = 0.02). See *Transfer to small reservoirs* below. |
| And on ungauged stock tanks? | **No; NDWI does far better.** On 461 stock tanks and farm ponds at eight sites across Texas, scored against outlines drawn from 0.6 m NAIP aerial photos on the nearest-day flight, NDWI with the Highland Lakes cutoff finds 74 % of ponds of 0.25 acre or more (70 % at a cutoff of 0), against 19 % for the S2 U-Net and 37 % for MNDWI. NDWI finds more than the U-Net at 6 of 7 sites, with 1 tie (p = 0.03). The U-Net catches up only above about 2 acres. Every method under-counts pond area, by roughly half for a 1–2 acre pond (mixed pixels). See *Ungauged stock tanks* below. |
| What did the design choices buy? | Per-scene SCL labels instead of a static JRC water map: **+0.175 IoU** [0.11, 0.24]. MoCo pretraining instead of random init: +0.003 [−0.012, +0.018], no detectable difference. |

Figures are in `results/figures/`; full tables are in `results/` (scored against the
hand labels) and `results/ai_labels/` (the same, against the AI labels).

### Pixel accuracy on hand-labelled points

The point estimates are stratified (Olofsson et al. 2014), with 95 % bootstrap intervals. The
strictest group is a held-out lake *and* held-out years (2024–2026): Travis points scored by the
model that never saw Travis, and likewise for Buchanan. Index cutoffs are tuned on the other
lakes' training-year points only.

| Method | All held-out (n = 585) | Held-out lake & years (n = 266) |
|---|---|---|
| NDWI, cutoff tuned on other lakes (≈ −0.03) | **0.994** [0.987–1.000] | **0.991** [0.978–1.000] |
| NDWI, Otsu per scene | 0.993 [0.985–1.000] | 0.991 [0.977–1.000] |
| S2 U-Net, single model | 0.970 [0.954–0.984] | 0.958 [0.926–0.982] |
| S2 U-Net, MC dropout (10 passes) | 0.968 | 0.953 |
| S2 U-Net, 3-seed ensemble | 0.968 [0.952–0.983] | 0.953 [0.921–0.978] |
| MNDWI, Otsu per scene | 0.957 | 0.944 |
| Weak label used for training (SCL ∩ JRC) | 0.940 | 0.914 |
| MNDWI, cutoff tuned on other lakes (≈ −0.14) | 0.939 [0.904–0.968] | 0.928 [0.887–0.967] |
| S1 U-Net (radar, n = 457 / 222) | 0.948 [0.928–0.966] | 0.933 [0.902–0.959] |
| S1 VV threshold, Otsu / tuned | 0.883 / 0.866 | 0.866 / 0.869 |

Part of NDWI's lead is what the test rewards: each label judges a single 10 m pixel, mostly at
the shoreline. NDWI uses only 10 m bands, while MNDWI and the U-Net also use 20 m short-wave
infrared bands, and the U-Net smooths over neighbouring pixels. Of the 22 points where the U-Net
and NDWI disagree, NDWI is right on 19. The U-Net's misses pull towards each lake's usual outline:
dry lakebed called water, and water in usually dry places (the 2018 flood, the 2026 refills)
called land. Its training label carried the JRC water history, and some of that leaked in.

### Mapped area vs official level (`results/area_vs_level.csv`)

| Lake | Method | Dates | r with level | RMSE vs TWDB area (acres) | RMSE after linear fit | CV of area |
|---|---|---|---|---|---|---|
| Travis | S2 U-Net | 121 | 0.990 | 1,014 | 364 | 20.4 % |
| | NDWI, tuned | 121 | **0.994** | **892** | **328** | 21.5 % |
| | NDWI, Otsu | 121 | 0.961 | 1,099 | 913 | 21.8 % |
| | MNDWI, tuned | 121 | 0.972 | 1,044 | 727 | 19.0 % |
| | S2 SCL class | 121 | 0.975 | 1,602 | 741 | 18.2 % |
| | S1 U-Net | 252 | 0.988 | 1,185 | 445 | 20.2 % |
| | S1 VV threshold | 252 | 0.913 | 1,463 | 1,363 | 17.9 % |
| Buchanan | S2 U-Net | 148 | 0.992 | **925** | 213 | 12.3 % |
| | NDWI, tuned | 148 | 0.993 | 992 | **211** | 13.0 % |
| | NDWI, Otsu | 148 | 0.994 | 1,010 | 226 | 13.1 % |
| | MNDWI, tuned | 148 | 0.993 | 1,017 | 235 | 12.9 % |
| | S1 U-Net | 250 | 0.988 | 730 | 312 | 11.9 % |
| | S1 VV threshold | 250 | 0.565 | 3,986 | 2,245 | 19.1 % |
| LBJ, Inks, Marble Falls | see the noise-floor table below | | | | | |

"RMSE after linear fit" removes a constant offset and scale between what each method counts as
lake and what the TWDB rating curve counts. The U-Net and NDWI map **less** area than TWDB: about
5 % at Travis and Buchanan and 5–15 % at the small lakes, most at full pool and in the Oct 2018
flood. MNDWI, with its low tuned cutoff, sits slightly above TWDB at Travis. Where a method sits
relative to TWDB depends heavily on its cutoff, which is why correlation and the after-fit error
are the fairer comparison (see Limitations).

### Pass-through lakes (`results/noise_floor.csv`)

The spread (CV) of mapped area on days when the level was within 0.5 ft of normal, and the
change in mapped area during drawdowns (level > 2 ft below normal) against the change in TWDB
area:

| | Steady-level CV: LBJ / Inks / Marble Falls | Drawdown area change: LBJ / Inks / Marble Falls |
|---|---|---|
| TWDB area (from level) | — | −14 % / −28 % / −27 % |
| NDWI, tuned | **1.2 / 3.5 / 3.4 %** | −7 % / −24 % / −13 % |
| S2 U-Net | 1.5 / 3.5 / 3.5 % | −4 % / −19 % / −6 % |
| MNDWI, tuned | 2.2 / 5.6 / 7.3 % | −5 % / −28 % / −15 % |
| S1 U-Net | 2.3 / 3.6 / 6.9 % | −4 % / −23 % / −15 % |
| S1 VV threshold | 7.1 / 7.4 / 10.4 % | −3 % / −17 % / −15 % |

NDWI and the U-Net are equally steady, but the U-Net under-reports drawdowns most. NDWI and
MNDWI follow them more closely. Only 3–6 optical dates per lake fall in a drawdown, so these drop
estimates are rough, but radar shows the same pattern. TWDB's area at the *same* level also steps
up by 5 % (LBJ, 2020), 5 % (Inks, 2021–22) and 11 % (Marble Falls, 2020). This looks like a
rating-curve revision; the survey dates have not been checked. No such step appears in any
satellite record.

### Cloud (`results/cloudy.csv`)

Median absolute area error vs TWDB:

| Case | Travis | Buchanan |
|---|---|---|
| S2 U-Net, clear dates | 4.6 % | 4.3 % |
| S2 U-Net, cloudy dates (20–90 % clear), clouds left in | 10.4 % | 11.7 % |
| S2 U-Net, cloudy dates, cloud pixels masked out | 36.0 % | 29.5 % |
| S1 U-Net, dates with no clear S2 within ±3 days | 6.5 % (n = 194) | 3.0 % (n = 177) |
| Months with a usable map, S2 clear only → S2 + S1 | 70 % → 100 % | 84 % → 100 % |

Masking clouds removes water as well as cloud, so the area drops. Leaving the clouds in lets the
model guess beneath them; it guesses well under thin cloud but fails badly under thick cloud
(90th percentile error 37–67 %). Radar avoids that trade-off.

### Transfer to small reservoirs (`results/transfer/`)

The Highland Lakes are gauged daily, so there satellite area mostly confirms the gauge. The useful
case is a reservoir with no gauge. As a test, `hlw.transfer` maps ten small reservoirs that have a
TWDB record **as if they had none** (`configs/transfer.yaml`, notebook 8):

- **Models:** the shipped models, trained only on the Highland Lakes, with no fine-tuning.
- **Baseline cutoffs:** NDWI, MNDWI and VV cutoffs are the Highland Lakes medians (tuned on the hand labels); per-scene Otsu is also reported.
- **Zones:** from JRC around TWDB's published location point.
- **Scoring:** TWDB is used only to score.

The lakes were chosen by fixed rules from TWDB's list of 122:
- maximum area roughly 300–7,000 acres;
- an area swing of at least 45 % since 2018;
- a complete record;
- a setting unlike the training lakes (turbid red-clay water, mudflat basins, canyons).

| median over 10 lakes (one clear S2 / two S1 dates per month) | raw error | median signed error | error after linear fit | r with TWDB area |
|---|---|---|---|---|
| S2 U-Net | **6.0 %** | −2 % | **5.6 %** | 0.99 |
| NDWI, Highland Lakes cutoff | 7.5 % | −7 % | 6.8 % | 0.99 |
| NDWI, per-scene Otsu | 8.5 % | −8 % | 7.3 % | 0.98 |
| MNDWI, Highland Lakes cutoff | 8.1 % | −7 % | 7.7 % | 0.98 |
| MNDWI, per-scene Otsu | 12.4 % | −12 % | 8.2 % | 0.97 |
| S2 SCL water class | 12.5 % | −11 % | 16.1 % | 0.87 |
| S1 U-Net | 11.0 % | −7 % | 13.9 % | 0.89 |
| S1 VV threshold, HL cutoff | 20.9 % | +6 % | 22.1 % | 0.73 |

Raw error is what an ungauged user would get; the linear fit needs a gauge and is a diagnostic
(it shows whether a lead comes only from mapping more water). The U-Net and NDWI split the ten
lakes 5–5 on raw error, but the split follows size: on the five smallest lakes (all under 800
acres) the U-Net is closer to TWDB than NDWI on 84–100 % of dates (Hords Creek: 49 %), and on the
four largest NDWI is closer on most dates. NDWI's under-mapping grows as lakes shrink. On Medina's
turbid July 2026 refill (TWDB ~2,500 acres) the U-Net mapped ~2,140 acres, NDWI ~1,750 and MNDWI
400–1,400. After a linear fit the U-Net tracks change better than NDWI on 8 of 10 lakes (sign
test p = 0.11) and than MNDWI on 9 of 10 (p = 0.02). Lake Abilene is 17–21 % low for the U-Net and
both indices, all at high water, which points at flooded vegetation or its rating curve. The
S1 U-Net beats the VV threshold on 8 of 10 lakes, and radar fills every month (clear S2 alone: a
median of 91 %), but its correlation with TWDB on these small lakes (0.71–0.97) is well below the
0.99 seen at Travis and Buchanan. Figures: `results/transfer/transfer_timeseries.png`,
`transfer_error_vs_size.png`, `transfer_maps.png`.

### Ungauged stock tanks (`results/ponds/`)

The water nobody gauges is mostly far smaller: stock tanks and farm ponds of a few acres. They
have no record to score against, so `hlw.ponds` uses **independent imagery** as the reference
(`configs/ponds.yaml`, notebook 9):

- **Reference:** USDA NAIP aerial photos (0.6 m, RGB + NIR; Texas flights in 2018, 2020 and 2022),
  read at 1 m. Water is NAIP NDWI > 0.2, grown to > 0.1, with NIR below a fraction of red, and
  with shape rules that drop shadow strips. The rules were fixed after visual checks showed
  failures. Built-up land (WorldCover) is masked.
- **Sites:** eight 5 × 5 km boxes in ranch and farm country, one per landscape. The first South
  Texas box landed on a refinery and was moved before any satellite comparison there.
- **Satellite maps:** the shipped models and Highland Lakes cutoffs, unchanged, on the clear S2
  day nearest each flight (median gap 1 day) and the nearest S1 day (median 3 days).
- **Scoring:** within each pond's footprint plus 20 m. "Detected" means at least one 10 m water
  pixel there. A chance rate comes from the same footprints moved ~650 m onto dry land.
- **Reference check:** 60 random reference outlines (20 per size class) were judged from the
  photos, first by the AI labeller (Claude Opus 5.5) and then blind by hand (`labels/tool/ponds.html`). The
  AI doubted 17, mostly small ponds; the hand check found 59 of 60 plainly water and 1 unsure (a pond
  half in sun glint). Glint can make an outline too small. The headline uses ponds of ≥ 0.25
  acre (340 pond-dates with S2), about ten S2 pixels; below that NDWI finds 27 % and the U-Net 1 %.

| ponds ≥ 0.25 acre | detected | chance | detected, ≥ 1 acre | total area vs NAIP | dry tanks flagged | false blobs per km² |
|---|---|---|---|---|---|---|
| NDWI, HL cutoff (−0.025) | **74 %** | 0 % | **81 %** | **−40 %** | 7 % | 0.2 |
| NDWI > 0 | 70 % | 0 % | 76 % | −44 % | 6 % | 0.1 |
| S2 U-Net | 19 % | 0 % | 71 % | −58 % | 2 % | 0.04 |
| S2 U-Net, expected area | — | — | — | −52 % | — | — |
| MNDWI, HL cutoff (−0.14) | 37 % | 0 % | 66 % | −53 % | 11 % | 0.4 |
| S2 SCL water class | 8 % | 0 % | 34 % | −76 % | 1 % | 0 |
| S1 U-Net (S1 dates, n = 395) | 16 % | 1 % | 62 % | −51 % | 2 % | 0.5 |
| S1 VV threshold, HL cutoff | 99 % | 85 % | 100 % | −1 % | 70 % | 152 |

The U-Net's edge on small reservoirs does not carry down to ponds. Below about 2 acres it gives a
faint probability blob that rarely clears 0.5. MNDWI's 20 m SWIR band blurs a pond a few pixels
wide. NDWI uses only 10 m bands. Above 2 acres the U-Net catches up (92 % detected at 2–5 acres,
−9 % median area error at 5–50 acres). The VV threshold "detects" almost every pond only because
it scatters water everywhere: 85 % of the same footprints moved onto dry land also "detect".
Area is the weak point for everyone: a 1–2 acre pond is ~40 pixels, half of them on the shore.
Satellite area at this size is a wetness index, not acreage. Figures:
`results/ponds/ponds_detection.png`, `ponds_area_error.png`, `ponds_gallery.png`.

### What this means

On optical imagery, a well-tuned NDWI threshold is the method to beat, and the U-Net doesn't beat
it on the Highland Lakes. Per pixel NDWI is more accurate; for lake area at Travis and Buchanan it
is as good; on the steady lakes it is as steady and responds more to small drawdowns. The cutoff
matters: tuned on the hand labels it sits near the textbook 0, and an Otsu cutoff picked per
scene does almost as well. Maps of where NDWI and the U-Net disagree
(`results/figures/blog_ndwi_vs_unet.png`) show NDWI catching the narrow tips of coves that the
U-Net's smoothing loses.

The U-Net earns its place elsewhere:

- **Radar.** Distilling the optical model into an S1 model gives a lake-area record that matches
  the optical one (r = 0.988) through cloud, at twice the revisit, without any radar labels. No
  radar threshold comes close. This is the result a reservoir monitoring service would actually
  use.
- **Uncertainty.** The ensemble's disagreement marks where the map is likely wrong (AUROC 0.95);
  an index gives no such signal.
- **Small, muddy, unfamiliar reservoirs.** On the smallest transfer lakes and on turbid
  floodwater, the U-Net is a few points more accurate than NDWI with a cutoff tuned elsewhere.

And it has a clear limit. Below about 2 acres the advantage reverses sharply, and on stock tanks
the simple 10 m index finds four times as many ponds. The model also leans towards each lake's
usual outline, a bias traced to its training label. The honest scope for the shipped U-Net is
reservoirs of a few hundred acres and up, and above all as the teacher for a radar model.

## Limitations

- **One person labelled the test points.** The primary hand labels cover 660 points, labelled
  blind, with a blind second look at the 26 points where they disagreed with the AI labeller. That is careful but not an expert panel, and 20 points (mostly exact shoreline)
  were still marked unsure and dropped. The AI labels are kept and every method is also scored
  against them (`results/ai_labels/`). The two keys rank methods differently per pixel, so
  treat that ranking as specific to this labelling.
- **The U-Net leans towards each lake's usual outline.** It shows a quarter to two thirds of the
  official area drop during small drawdowns, and on the hand points it calls usually-wet dry
  lakebed water and misses water in usually-dry places. The likely cause is the JRC history in
  its training label. Calibrate, or retrain with labels that don't carry the history, before
  monitoring lakes that move a few feet.
- **The per-pixel test favours sharp, 10 m methods.** Each label judges one 10 m pixel, mostly at
  the shoreline. NDWI (10 m bands only) has an advantage there over methods that use 20 m bands
  or spatial context. The area comparisons are the check on that.
- **Point counts are small at the class boundary.** Only 57 held-out points received
  probabilities between 0.05 and 0.95, so calibration in that range is loosely constrained.
- **"Official area" is not measured area.** TWDB's surface area is the level passed through a
  bathymetric rating curve. The 5–15 % shortfall of the U-Net and NDWI, and the larger
  shortfall at full pool, could come from flooded vegetation, coves narrower than 10 m pixels, the lake zone
  (1984–2020 maximum extent plus 300 m), an out-of-date rating curve, or a mix of these. These
  results cannot tell them apart. Correlation with level is the robust measure; absolute acres
  are not.
- **The labels for the radar model come from the optical model, not from people.** Radar
  accuracy is bounded by the optical model's accuracy, and it inherits any optical biases.
  Wind-roughened water and flooded trees are the known failure modes for C-band radar. They were
  not studied separately.
- **One region.** Generalisation is tested by leaving one lake out and by ten unseen small
  reservoirs in West/Central Texas. All are scored against rating-curve areas, not measured
  ones, and none is smaller than ~150 acres. Stock tanks were tested against NAIP aerial-photo
  outlines at eight sites. That is one snapshot per site every two years, so pond time series
  are untested. The outlines were checked by eye (60 of them), not surveyed, and the built-up
  mask drops farmstead ponds.
- **Small ablations.** The pretraining and label ablations use one seed, on the Travis fold only.
  The pretraining effect (+0.003) is within seed-to-seed variation.
- **The weights are research artefacts.** They were trained on 2018–2023 Highland Lakes scenes
  only. The transfer test supports using them on other Texas reservoirs of a few hundred acres
  or more, where they are slightly better than NDWI. On clear optical imagery of familiar water,
  NDWI with a cutoff near 0 is as good or better. The stock-tank test says not to use them on
  ponds under ~2 acres; use NDWI there. Elsewhere, treat output as unvalidated.

## Data (all public, all opened and checked)

| What | Source | Notes |
|---|---|---|
| Sentinel-2 L2A (10–20 m optical) | [Planetary Computer `sentinel-2-l2a`](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a) | Duplicate items (original + 2024 reprocessing) are de-duplicated; the +1000 DN offset of processing baseline ≥ 04.00 is removed so 2018 and 2026 reflectances are comparable. |
| Sentinel-1 RTC (10 m C-band radar, VV/VH) | [Planetary Computer `sentinel-1-rtc`](https://planetarycomputer.microsoft.com/dataset/sentinel-1-rtc) | Radiometrically terrain-corrected gamma0, converted to dB. CC-BY-4.0. |
| JRC Global Surface Water v1.3 | [Planetary Computer `jrc-gsw`](https://planetarycomputer.microsoft.com/dataset/jrc-gsw) | Landsat 1984–2020 *summary* layers only (occurrence, extent…); no per-month history on PC. Used for lake zones, strata and as a weak label. |
| ESA WorldCover 2021 | Planetary Computer `esa-worldcover` | Static weak label (class 80); built-up mask (class 50) in the stock-tank test. |
| Impact Observatory annual LULC v02 | Planetary Computer `io-lulc-annual-v02` | Annual weak label (class 1), 2017–2023 only. |
| Lake level, area, storage | [TWDB Water Data for Texas](https://www.waterdatafortexas.org/reservoirs/individual/travis.csv) daily CSVs | `surface_area` is level converted through an elevation–area rating curve, **not** an independent area measurement. One-day spikes (e.g. Inks 764.8 ft in 2023) are removed. A snapshot is in `data/levels/`. |
| USDA NAIP aerial imagery | [Planetary Computer `naip`](https://planetarycomputer.microsoft.com/dataset/naip) | 0.6 m RGB + NIR, Texas flights 2018, 2020, 2022 (per-quad flight dates in the item metadata). Not radiometrically calibrated. Reference for the stock-tank test only. |
| Pretrained weights | [TorchGeo 0.10.1](https://torchgeo.readthedocs.io/en/stable/tutorials/pretrained_weights.html) | `ResNet18_Weights.SENTINEL2_ALL_MOCO` (SSL4EO-S12, 13-band L1C) and `ResNet50_Weights.SENTINEL1_ALL_MOCO` (SSL4EO-S12, VV/VH GRD dB). |

Not used, and why: JRC monthly history and Dynamic World need Google Earth Engine credentials; the
LCRA Hydromet API was not opened, so TWDB is the only level source.

## Method

**Lake zones.** Each lake is the JRC 1984–2020 maximum water extent, cut at the dams
(`configs/lakes.yaml`, drawn by hand), restricted to the component touching the lake and dilated
300 m. Area is counted only inside the zone. Each lake has its own 10 m UTM 14N grid.

**Weak labels (training).** No hand-drawn masks exist. Five public proxies were evaluated against
the hand labels (below). Training uses `scl_jrc`: per-scene Sentinel-2 Scene Classification water
(class 6), with SCL water outside JRC history set to *ignore*, and cloud/shadow pixels ignored.

**Hand labels (validation).** 660 stratified random points on 18 scenes (6 each for Travis and
Buchanan at low/median/high water in both the training and held-out years; 2 for each small lake).
Strata come only from JRC occurrence (A ≥ 90 %, B 1–89 %, C 0 %), so no compared method decides
where points fall. Two answer keys: the hand labels (primary; 20 unsure, excluded) made blind in the browser
tool, with a blind second look at the 26 conflicts, and labels by Claude Opus 5.5 viewing the same
true-colour and SWIR/NIR/red views (99 unsure, almost all on the exact shoreline). Accuracy is estimated with the stratified
estimators of Olofsson et al. (2014), with 95 % stratified-bootstrap intervals. Audit them with
the labelling tool (below) or `make labels` (contact sheets in `data/label_sheets/`).

**Splits — by lake and by time, never random tiles.** Two leave-one-lake-out folds: one never sees
Travis, one never sees Buchanan; the small lakes are always in training. Training uses only
2018–2023 scenes; 2024–2026 is held out. Every Travis map (including its whole timeline) comes from
the model that never saw Travis, and likewise for Buchanan.

**Models.**
1. *Baselines:* NDWI (B3/B8) and MNDWI (B3/B11). Cutoffs tuned for weighted F1 on training-year
   hand points from the *other* lakes; also per-scene Otsu.
2. *S2 U-Net:* `segmentation_models_pytorch` U-Net, ResNet-18 encoder from the SSL4EO MoCo weights
   (L2A has no B10, so a zero band is inserted), 256 px chips biased toward the fluctuation zone,
   masked BCE, AdamW + one-cycle, 2000 iterations × 16 chips, AMP.
3. *S1 U-Net:* ResNet-50 encoder from SSL4EO MoCo S1 weights, trained to reproduce the fold's S2
   ensemble on clear S2 scenes paired with an S1 pass within ±2 days and < 0.15 ft level change
   (optical → radar distillation; no S2 is needed at inference).
4. *Uncertainty:* deep ensemble (3 seeds) and MC dropout (10 passes, Dropout2d before the head),
   with a reliability diagram and ECE on held-out hand points.
5. *Ablations (Travis fold):* random initialisation instead of MoCo; static JRC ≥ 50 % labels.

**Area vs level.** Every clear S2 day (≥ 99 % of the zone cloud-free per SCL; ≤ 2 per month) and
every S1 day is mapped; areas are compared with TWDB level and area by Pearson r and RMSE.

**Reference labels.** 660 stratified points (JRC occurrence strata: stable water, fluctuation zone,
never water) on 18 scenes, labelled blind by hand in a browser tool, with a blind second
look at the 26 points of disagreement with the AI labelling. All tuned cutoffs come from
these labels, so `make evaluate` runs before the timelines; `hlw.timeline --fix-index` and
`hlw.transfer fix-cutoffs` recompute the index areas if the labels (and so the cutoffs) change.

## Reproduce

```bash
uv sync                 # pinned environment (uv.lock), Python 3.12, PyTorch CUDA 12.6 wheels
make test               # offline unit tests
make all                # data -> train -> evaluate -> timeline -> analyze -> weights -> web
                        #   -> blogfigs -> transfer -> ponds
```

Steps can be run one at a time (`make data`, `make train`, ...). Downloads go to `data/`
(~6 GB). The TWDB snapshot, scene inventory, scene selection and hand labels are committed, so
re-running uses the same scenes; re-running `make scenes`/`make levels` refreshes them.

**Compute actually used:** one desktop, NVIDIA RTX 4070 SUPER (12 GB), 6-core CPU, 16 GB RAM.
Each S2 training run took 6–8 min (12.5 min without pretraining; data-loader bound) and each S1
run 3–4 min, for 10 runs in total (~70 min of GPU time). Evaluating the 18 hand-labelled scenes
took ~10 min. The timelines are network bound: ~1,800 scene loads took ~3 h for S2 and ~1.5 h for
S1. Downloads total ~6 GB. Training never used more than ~4 GB of GPU memory, so a free Colab T4
works; expect 3–4× longer training (`make train ITERS=1000` halves it). Run the steps one at a
time on a 16 GB machine: running training, teacher-target building and the timelines at once
exhausted RAM here. The transfer test (`make transfer`) scanned ~7,000 S2 and ~3,600 S1 days and
mapped ~3,500 scenes in about 3.5 h, CPU bound (COG decoding), with up to 7 lakes in parallel.
The stock-tank test (`make ponds`) reads 24 NAIP site-years (~1 min each, cached as ~1.4 GB in
`data/raw/naip/`) and maps 91 S2 and 48 S1 days on 5 km boxes, about 40 min in total.

## The hand labels, and labelling them yourself

The test points were labelled twice on purpose, to see whether AI labels give the same verdict
as a person's: first by an AI (Claude Opus 5.5, viewing image chips), then by hand in a browser tool that
never shows the AI's answer. The hand labels are the primary reference
(`labels/hand_points_human_final.csv`).
The two agree on 97 % of the points both committed on (Cohen's kappa 0.94). The difference that
mattered is commitment: the AI marked 99 points unsure, the hand labels 20. Dropped points are the hard
shoreline ones, so scoring against the AI labels makes the test easier and hides the gap between
NDWI and the other methods (see *Pixel accuracy*).

The tool works for anyone who wants to add an independent opinion:

```bash
make label-tool                                     # renders chips (~100 MB, git-ignored)
uv run python -m http.server 8000 -d labels/tool    # open http://localhost:8000
```

Press **W** (water), **N** (not water) or **U** (unsure) for the 10 m pixel in the magenta marks.
Judge what covers most of that 10 m square. Shoreline points come first. The tool never shows a
model's answer or anyone else's label, and progress saves in the browser. When you finish,
**Review labels** shows everything you called water, then not water, then unsure, as a grid in
the order you labelled them, so slips stand out; one click (in the grid or the enlarged view)
relabels a point, and **I'm done reviewing** records that the labels are final. **Export CSV**
downloads `hand_points_human.csv`. Save it to `labels/` under your own name
(`hand_points_human.csv` holds the published hand labels), then:

```bash
make human-score LABELS=labels/hand_points_yourname.csv   # agreement with the AI labels
                                                          # (Cohen's kappa) + every method re-scored
```

Results go to `results/human_labels/`, leaving the published `results/` untouched. A partial file works, since the estimators re-weight by
how many points were labelled in each scene and stratum. 660 points take about 30–45 minutes.

A second page, `ponds.html`, checks the stock-tank reference outlines the same way (Water /
Partly / Not water / Unsure, plus "outline misses water"):

```bash
make pond-audit-tool     # renders 60 NAIP close-ups (git-ignored); open http://localhost:8000/ponds.html
make pond-audit-score    # compares labels/ponds_naip_audit_human.csv with the AI's check
```

A blind **second look** at the points where two labellers disagree:

```bash
uv run python -m hlw.handlabel recheck   # conflicts + equal number of agreed controls, shuffled
# serve the tool and open http://localhost:8000/?set=recheck, export hand_points_recheck.csv
make recheck-score                       # merge, report flips and self-consistency, re-score
```

## Inference

```bash
# S2 model shipped in weights/ (trained on Buchanan, Inks, LBJ, Marble Falls 2018-2023)
uv run hlw infer --lake travis --date 2023-10-16 --mc 10 --out travis.tif
uv run hlw infer --bbox -98.0 30.38 -97.88 30.45 --date 2025-07-20 --out out.tif
# S1 model, also shipped in weights/ (65 MB)
uv run hlw infer --lake buchanan --date 2023-08-15 --sensor s1 --out s1.tif
```

Output bands: water probability, MC-dropout std, binary mask. Area in acres is printed.

## Web app

**Live:** [alo.mn/apps/mapping-texas-lakes-from-space](https://alo.mn/apps/mapping-texas-lakes-from-space/)

Lake tabs, a before/after slider with the mapped shoreline and ensemble uncertainty, the mapped
area vs the official record over time, area vs level, and the accuracy tables; a second row of
tabs shows the ten transfer reservoirs. `web/` holds the page source and its data: a static page
with no build step and no external libraries. To view it locally, run
`python -m http.server -d web`.

## Repository layout

```
configs/lakes.yaml     lakes, dam cut polygons, folds, years
configs/transfer.yaml  ten small "pseudo-ungauged" transfer reservoirs and why they were chosen
configs/ponds.yaml     eight stock-tank sites and the NAIP reference rules
src/hlw/               pipeline (aoi, levels, scenes, selection, download, labels, dataset,
                       model, train, teacher, handlabel, metrics, evaluate, timeline, analyze,
                       webexport, infer, blogfigs, transfer, transfer_figs, ponds, ponds_figs, nb)
labels/hand_points.csv validation points, strata and the AI answer key
labels/hand_points_human*.csv  the hand labels: first pass, final (after the second look)
labels/hand_points_recheck.csv, recheck_manifest.csv  the blind second look
labels/ponds_naip_audit.csv  visual check of 60 NAIP reference ponds (AI labeller)
labels/ponds_naip_audit_human.csv  the blind hand check of the same 60 (used in results)
labels/tool/            browser labelling tools: index.html (points), ponds.html (pond outlines)
notebooks/             teaching notebooks (executed, with outputs)
results/               metrics tables, figures, summary, scored against the hand labels
                       (results/ai_labels/: the same against the AI labels; results/transfer/:
                       transfer test; results/ponds/: stock-tank test)
weights/               shipped S2 model (+ MANIFEST.json with hashes and training metadata)
web/                   static web app
tests/                 offline unit tests
```

## License and attribution

Code: MIT. Contains modified Copernicus Sentinel data (2018–2026). JRC Global Surface Water
© European Commission. ESA WorldCover © ESA. Impact Observatory LULC CC-BY-4.0. Lake data: Texas
Water Development Board. Pretrained weights: SSL4EO-S12 (Wang et al. 2023) via TorchGeo.
