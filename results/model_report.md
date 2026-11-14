# GridSentinel AI - offline model evaluation

All numbers below were produced by `python -m ml.train_all` on independent simulation runs (run-group split: models fit on *train* runs, thresholds / regime chosen on *validation* runs, results reported on unseen *test* and *stress* runs).

## Data

| split | rows | attack fraction | additive | scaling | replay | coordinated |
|---|---|---|---|---|---|---|
| stress_heavy_load | 2500 | 0.364 | 231 | 314 | 108 | 256 |
| stress_high_noise | 2500 | 0.340 | 205 | 198 | 252 | 195 |
| stress_low_intensity | 2500 | 0.376 | 176 | 479 | 164 | 122 |
| test | 5000 | 0.363 | 502 | 499 | 482 | 330 |
| train | 15000 | 0.363 | 1287 | 1471 | 912 | 1777 |
| validation | 5000 | 0.358 | 335 | 602 | 354 | 497 |

## Model A - detection (threshold 0.528, 1287 features)

| split | accuracy | precision | recall | F1 | FPR | FNR | ROC AUC |
|---|---|---|---|---|---|---|---|
| stress_heavy_load | 0.9660 | 0.9791 | 0.9263 | 0.9520 | 0.0113 | 0.0737 | 0.9756 |
| stress_high_noise | 0.8700 | 0.9070 | 0.6882 | 0.7826 | 0.0364 | 0.3118 | 0.8778 |
| stress_low_intensity | 0.8464 | 0.9696 | 0.6111 | 0.7497 | 0.0115 | 0.3889 | 0.8552 |
| test | 0.9214 | 0.9772 | 0.8020 | 0.8809 | 0.0107 | 0.1980 | 0.9271 |
| validation | 0.9450 | 0.9755 | 0.8680 | 0.9186 | 0.0121 | 0.1320 | 0.9522 |

Detection recall by attack type:

| split | additive | scaling | replay | coordinated |
|---|---|---|---|---|
| stress_heavy_load | 1.0000 | 0.9873 | 0.5093 | 0.9609 |
| stress_high_noise | 0.9317 | 0.8283 | 0.3095 | 0.7795 |
| stress_low_intensity | 0.9375 | 0.4906 | 0.6220 | 0.5984 |
| test | 0.9163 | 0.9960 | 0.4523 | 0.8455 |
| validation | 1.0000 | 0.9468 | 0.5085 | 0.9396 |

## Model B - localisation (selected regime: **attack_only**, feature scope: local)

| regime | split | oracle micro-F1 | oracle macro-F1 | two-stage micro-F1 | two-stage macro-F1 |
|---|---|---|---|---|---|
| attack_only | validation | 0.7223 | 0.7122 | 0.7365 | 0.7295 |
| attack_only | stress_heavy_load | 0.7417 | 0.7326 | 0.7406 | 0.7269 |
| attack_only | stress_high_noise | 0.5056 | 0.4725 | 0.5020 | 0.4492 |
| attack_only | stress_low_intensity | 0.4961 | 0.4487 | 0.5476 | 0.4803 |
| attack_only | test | 0.5921 | 0.5583 | 0.6080 | 0.5696 |
| all | validation | 0.7270 | 0.7231 | 0.7212 | 0.7176 |
| all | stress_heavy_load | 0.7089 | 0.6919 | 0.7012 | 0.6853 |
| all | stress_high_noise | 0.5169 | 0.4454 | 0.5009 | 0.4313 |
| all | stress_low_intensity | 0.5359 | 0.4803 | 0.5276 | 0.4724 |
| all | test | 0.5796 | 0.5307 | 0.5730 | 0.5253 |

*Oracle* = localiser applied to truly attacked snapshots only. *Two-stage* = every snapshot; the localiser only runs when the detector raises an alarm, so detection misses and false alarms count.

### Per-bus localisation on the test runs (selected regime, oracle setting)

| bus | actual + | predicted + | precision | recall | F1 |
|---|---|---|---|---|---|
| 1 | 255 | 328 | 0.6677 | 0.8588 | 0.7513 |
| 2 | 102 | 47 | 0.8936 | 0.4118 | 0.5638 |
| 3 | 78 | 72 | 0.1806 | 0.1667 | 0.1733 |
| 4 | 212 | 101 | 0.9208 | 0.4387 | 0.5942 |
| 5 | 110 | 164 | 0.5854 | 0.8727 | 0.7007 |
| 6 | 274 | 152 | 0.9342 | 0.5182 | 0.6667 |
| 7 | 173 | 378 | 0.3942 | 0.8613 | 0.5408 |
| 8 | 120 | 64 | 0.5938 | 0.3167 | 0.4130 |
| 9 | 112 | 153 | 0.4314 | 0.5893 | 0.4981 |
| 10 | 158 | 111 | 0.8198 | 0.5759 | 0.6766 |
| 11 | 198 | 84 | 0.7738 | 0.3283 | 0.4610 |
| 12 | 81 | 115 | 0.3217 | 0.4568 | 0.3776 |
| 13 | 87 | 109 | 0.3394 | 0.4253 | 0.3776 |
| 14 | 178 | 206 | 0.3544 | 0.4101 | 0.3802 |
| 15 | 347 | 280 | 0.8250 | 0.6657 | 0.7368 |
| 16 | 269 | 122 | 0.8934 | 0.4052 | 0.5575 |
| 17 | 198 | 148 | 0.8108 | 0.6061 | 0.6936 |
| 18 | 264 | 219 | 0.7945 | 0.6591 | 0.7205 |
| 19 | 257 | 148 | 0.8581 | 0.4942 | 0.6272 |
| 20 | 252 | 76 | 0.9342 | 0.2817 | 0.4329 |
| 21 | 205 | 251 | 0.6454 | 0.7902 | 0.7105 |
| 22 | 281 | 304 | 0.5921 | 0.6406 | 0.6154 |
| 23 | 200 | 114 | 0.7719 | 0.4400 | 0.5605 |
| 24 | 161 | 170 | 0.8529 | 0.9006 | 0.8761 |
| 25 | 169 | 152 | 0.6974 | 0.6272 | 0.6604 |
| 26 | 53 | 155 | 0.0581 | 0.1698 | 0.0865 |
| 27 | 163 | 98 | 0.8776 | 0.5276 | 0.6590 |
| 28 | 123 | 84 | 0.8214 | 0.5610 | 0.6667 |
| 29 | 102 | 172 | 0.4070 | 0.6863 | 0.5109 |
| 30 | 115 | 172 | 0.3837 | 0.5739 | 0.4599 |

## Model C - attack-type classification

| split | accuracy (attacked snapshots) | macro-F1 | two-stage accuracy | two-stage macro-F1 |
|---|---|---|---|---|
| stress_heavy_load | 0.6634 | 0.6794 | 0.6520 | 0.6622 |
| stress_high_noise | 0.4106 | 0.3655 | 0.3726 | 0.3385 |
| stress_low_intensity | 0.2232 | 0.2616 | 0.2313 | 0.2396 |
| test | 0.6680 | 0.6652 | 0.6726 | 0.6478 |
| validation | 0.5850 | 0.5828 | 0.5735 | 0.5470 |

Test confusion matrix (rows = simulator truth, columns = prediction):

| truth \ pred | additive | scaling | replay | coordinated |
|---|---|---|---|---|
| additive | 342 | 35 | 112 | 13 |
| scaling | 92 | 306 | 35 | 66 |
| replay | 10 | 30 | 337 | 105 |
| coordinated | 29 | 16 | 59 | 226 |

## Model D - severity regression (attacked snapshots)

| split | MAE | RMSE | R² | Spearman |
|---|---|---|---|---|
| stress_heavy_load | 2.0152 | 2.5748 | 0.6415 | 0.7933 |
| stress_high_noise | 1.7234 | 2.0494 | 0.4500 | 0.8851 |
| stress_low_intensity | 1.6086 | 2.0269 | 0.2079 | 0.7938 |
| test | 1.9643 | 2.9978 | 0.7386 | 0.9210 |
| validation | 1.9416 | 2.7229 | 0.6945 | 0.9016 |
