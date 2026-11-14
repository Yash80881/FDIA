# GridSentinel AI

**A Physics-Informed Artificial Intelligence Framework for Continuous False Data Injection Attack Detection, Localization and Mitigation in Smart Grids**

GridSentinel AI is a final-year Electrical Engineering research platform. It continuously simulates the PYPOWER IEEE 30-bus power system with AC power flow, generates noisy sensor measurements, injects randomised cyberattacks (False Data Injection Attacks, FDIA) into the received measurements, and then detects, localises, classifies and mitigates those attacks with Random Forest models combined with Weighted-Least-Squares state estimation. Everything is shown live in an interactive dashboard with a 30-bus digital twin.

It extends last year's project, *Detection of False Data Injection Attacks in Power Grids Using Machine Learning*, which did mainly binary detection on a static dataset.

> **Honesty statement.** Every metric in this README and in `results/` was computed by the code in this repository on simulated data, and is reproducible with the commands below. Simulator ground truth, AI predictions, physics verification and mitigation recommendations are always reported separately. Known limitations are listed in [Limitations](#limitations).

---

## Contents

1. [What is new compared with the previous project](#what-is-new)
2. [Architecture and data flow](#architecture)
3. [Electrical engineering background](#electrical-background)
4. [FDIA fundamentals and the attack engine](#fdia)
5. [Ground-truth labels and dataset](#labels)
6. [Feature engineering](#features)
7. [Random Forest models](#models)
8. [Physics-based verification (WLS state estimation)](#physics)
9. [Mitigation and recovery](#mitigation)
10. [Continuous monitoring, API and dashboard](#monitoring)
11. [Database structure](#database)
12. [Installation (Windows 10/11, Python 3.11)](#installation)
13. [Running the platform (PowerShell)](#running)
14. [Evaluation methodology and results](#results)
15. [Tests](#tests)
16. [Limitations](#limitations)
17. [Future scope](#future)
18. [Project structure](#structure)

---

<a id="what-is-new"></a>
## 1. What is new compared with the previous project

| Previous limitation | How GridSentinel AI addresses it |
|---|---|
| Mainly binary classification | Four models: detection, 30-bus localisation, attack type, severity |
| Limited synthetic measurements | Continuous PYPOWER AC power flow: 30 buses x (V, θ, P, Q) and 41 lines x (P/Q at both ends, I at both ends, loading), true and received values stored separately |
| No continuous monitoring | Live simulator (1 s default) + replay mode, heartbeat, stale-stream detection, background monitor that processes every snapshot once |
| Weak localisation, "all buses predicted 0" | One balanced Random Forest per bus with bus-local, physics-informed consistency features (optional WLS-residual features), per-bus calibrated thresholds, per-bus reporting, explicit "insufficient data" handling |
| Limited attack diversity | Additive, scaling, replay and coordinated FDIA, random start, duration, buses, channels and magnitudes |
| No electrical consistency check | AC WLS state estimation, chi-square and largest-normalised-residual tests, bad-data identification |
| Incomplete impact analysis | Severity target, recovered-state comparison, line-loading recovery, PTDF re-dispatch suggestions |
| Repeated / constant dashboard values, constant 89-90 % probability, constant load, only bus 1 voltage | All values come from the solved power flow and real model outputs, all 30 buses and 41 lines shown, step counter and heartbeat visible, "NOT LIVE" banner when the stream stops |
| API timeouts, slow CSV reloads | SQLite (WAL) with indexes, predictions cached by (run ID, step), no CSV reads at run time |
| Ground truth shown as AI prediction | Separate `ground_truth`, `ai`, `physics` and `mitigation` fields everywhere, colour-coded in the UI |

<a id="architecture"></a>
## 2. Architecture and data flow

```mermaid
flowchart LR
    subgraph SIM["Simulator process (simulator/)"]
        LP[Load profiles<br/>periodic + correlated + random + scenarios] --> GD[Generator re-dispatch<br/>within Pmin/Pmax]
        GD --> PF[PYPOWER runpf<br/>AC power flow + convergence check]
        PF --> MG[True measurements<br/>V, θ, P, Q, line flows, I, loading]
        MG --> NZ[Sensor noise<br/>z = z_true + e]
        NZ --> AT[Attack engine (attacks/)<br/>additive / scaling / replay / coordinated]
        AT --> LB[Ground-truth labels<br/>+ severity]
    end
    LB --> DB[(SQLite WAL<br/>runs, measurements,<br/>attack_events)]
    subgraph API["API process (backend/)"]
        MON[Monitor thread<br/>new snapshot -> once] --> FE[Feature engineering<br/>electrical + temporal + WLS residuals]
        FE --> RF[Random Forests<br/>A detect -> B localise -> C type -> D severity]
        MON --> SE[WLS state estimation<br/>χ² + normalised residuals]
        RF --> CMB[Combined decision<br/>+ alerts]
        SE --> CMB
        CMB --> CACHE[(predictions / alerts)]
        CACHE --> EP[FastAPI endpoints]
        MIT[Mitigation advisor<br/>isolate -> re-estimate -> recommend] --> EP
    end
    DB --> MON
    DB --> EP
    EP --> DASH[Dash dashboard<br/>digital twin + monitoring]
    DB --> TRAIN[Offline training & experiments<br/>ml/, experiments/]
    TRAIN --> MODELS[(models/*.joblib<br/>+ metrics)]
    MODELS --> RF
```

Three independent processes share one SQLite database in WAL mode: the **simulator** (only writer of measurements), the **API** (reads measurements, writes predictions/alerts) and the **dashboard** (only talks to the API). Stopping one of them never corrupts the others: every write is a short `BEGIN IMMEDIATE ... COMMIT` transaction.

<a id="electrical-background"></a>
## 3. Electrical engineering background

### 3.1 The network: PYPOWER `case30`

The network is loaded with `from pypower.case30 import case30` and is never edited (we deep-copy the case dictionary every step). Facts verified from the loaded case (`python cli.py check`, `tests/test_grid_powerflow.py`):

| Quantity | Value from the case |
|---|---|
| Buses / branches / generators | 30 / 41 / 6 |
| Reference (slack, "external grid") bus | 1 |
| Generator buses | 1, 2, 22, 27, 23, 13 |
| Base load | 189.2 MW, 107.2 MVAr (20 load buses) |
| Bus shunts | bus 5: B = 0.19 pu, bus 24: B = 0.04 pu |
| Branch ratings RATE_A | available for all 41 branches (16-130 MVA) |
| Voltage limits | 0.95-1.05 pu (PV buses up to 1.10 pu) |
| Base voltage | 135 kV for all buses (as given in the case file) |

**Important:** PYPOWER's `case30` is the MATPOWER 30-bus case (Alsac & Stott 1974 with generator data from Ferrero et al. 1997). It is *not* identical to the textbook "IEEE 30-bus" data set with generators at buses 1, 2, 5, 8, 11 and 13 (283.4 MW). As required, we use exactly the PYPOWER case and do not invent data.

### 3.2 AC power flow

For every step the bus loads and generator set points are updated, then

```python
results, success = runpf(ppc, ppoption(VERBOSE=0, OUT_ALL=0))
```

solves the Newton-Raphson AC power flow. A step whose power flow does not converge, has non-finite values, unrealistic voltages or a bus power mismatch above 0.1 MVA is **never** turned into measurements; the engine retries once with the base generator dispatch and otherwise skips the step and counts the failure (the run stops after `max_consecutive_pf_failures`).

### 3.3 Measurement model and equations

Complex bus voltage $V_i = |V_i| e^{j\theta_i}$; bus admittance matrix $Y_{bus}$ (built by PYPOWER's `makeYbus`, includes line charging and shunts).

* **Bus injection** (sign convention: generation positive, load negative):

  $S_i = P_i + jQ_i = V_i \left(\sum_k Y_{ik} V_k\right)^* = (P_{g,i} - P_{d,i}) + j(Q_{g,i} - Q_{d,i})$

* **Branch flow** at the from end (flowing *into* the branch), with $Y_f$ from `makeYbus`:

  $S_{f,\ell} = V_{f(\ell)} \left(Y_{f} V\right)_\ell^*$, similarly $S_{t,\ell}$; branch losses $P_{f,\ell} + P_{t,\ell} \ge 0$.

* **Branch current** in amperes (line-to-line base voltage $V_{base}$ in kV):

  $I_{f,\ell} = \dfrac{|S_{f,\ell}|\,[\text{MVA}]}{\sqrt{3}\,|V_{f}|\,[\text{pu}]\; V_{base}\,[\text{kV}]} \times 1000$

* **Loading**: $\text{loading}_\ell = \max(|S_{f,\ell}|, |S_{t,\ell}|) / \text{RATE\_A}_\ell \times 100\,\%$ (marked unavailable, never invented, if a rating is missing).

* **Sensor noise**: $z_{meas} = z_{true} + \varepsilon$, $\varepsilon \sim \mathcal N(0, \sigma^2)$ with $\sigma_V = 0.002$ pu, $\sigma_\theta = 0.10°$, $\sigma_{P,Q} = 1\,\% \cdot |z| + 0.2$ MW, $\sigma_I = 1\,\% \cdot |I| + 1$ A (all configurable in `config/simulation_config.yaml`).

### 3.4 Time-varying operation

Load at bus $i$: $P_i(t) = P_{i,base} \cdot S(t) \cdot D_i(t) \cdot C(t) \cdot E_i(t)$ where $S$ = Markov-switching scenario level (light 0.80 / normal 1.00 / heavy 1.15), $D_i$ = compressed daily profile (residential / commercial / industrial peak phase, ±18 %), $C$ = system-wide AR(1) factor (correlated variation), $E_i$ = independent AR(1) factor per bus; reactive load has an extra power-factor jitter. Demand changes are shared by the PV generators in proportion to their operating range (plus a small random re-dispatch), clipped to $[P_{min}, P_{max}]$; the slack generator absorbs the rest and the losses. The generated dataset covers total loads between about 109 and 260 MW.

<a id="fdia"></a>
## 4. FDIA fundamentals and the attack engine

A False Data Injection Attack changes what the control centre *receives*, $z_a = z + a$, without changing the physical grid. The engine (`attacks/`) therefore manipulates only a **copy of the received measurements**; the PYPOWER case, the true measurements and a clean received reference are never touched (tested in `tests/test_attacks.py`). Default attack surface: bus voltage-magnitude and angle sensors (`attacks.channels: [V, Ang]`); P/Q sensors and a labelled line-sensor mode can be switched on.

| Type | Manipulation | Randomised |
|---|---|---|
| Additive | $z_a = z + a$, independent offset per bus/channel | 1-3 buses, sign, ±0.008-0.04 pu / ±0.4-3° |
| Scaling | $z_a = \lambda z$, $\lambda = 1 \pm \delta$ | 1-3 buses, ±0.8-4 % on V, ±5-30 % on θ (never on the 0° reference angle) |
| Replay | $z_a(t) = z(t - \text{lag})$ - the attacker plays back legitimate recordings | 2-6 buses, lag 60-300 steps; only starts when ≥320 steps of history exist |
| Coordinated | Correlated offsets on a connected cluster: $a_i = a_0 \cdot 0.7^{hop(i)} \cdot (1 \pm 15\,\%)$ | 3-7 neighbouring buses, common sign/magnitude |

Scheduling: after an initial delay and a minimum normal gap, an attack starts with probability 0.04 per step (geometric inter-arrival times), lasts 10-40 steps, and its type is drawn with configurable weights. Attack, load and noise use separate seeded random streams, so changing the noise level does not change the attack sequence (tested).

<a id="labels"></a>
## 5. Ground-truth labels and dataset

Every snapshot stores `run_id, timestamp, step, attack_active, attack_id, attack_type, attack_duration_steps, attacked_bus_count, attack_severity, attack_channels, attack_bus_1 ... attack_bus_30`. Labels are derived from the *actual* perturbation: bus *i* is labelled attacked iff one of its received values differs from the clean value. **Severity** is the noise-normalised RMS perturbation of the attacked sensors, $\sqrt{\mathrm{mean}((z_a - z)/\sigma)^2}$, i.e. "how many noise standard deviations the attacker moved the sensors"; it is 0 for normal operation and is only ever a target, never an input.

`python cli.py validate` checks schema, missing values, convergence, label consistency (any bus label ⇔ global label), counts, normal rows (type `normal`, severity 0, no attack ID), that non-attacked received values stay within 6σ of the truth (using each run's own noise settings), unique steps and contiguous attack events. It also prints attack-type and per-bus distributions.

**Research dataset** (`python cli.py generate-dataset`): 13 independent runs of 2,500 steps (32,500 snapshots, about 36 % attacked): 6 train, 2 validation, 2 test, plus 3 stress-test runs (high noise ×2, heavy load, half attack intensity). Each run has its own seed and therefore its own load trajectory, noise and attack sequence. The validation report is in `results/dataset_validation.json`.

<a id="features"></a>
## 6. Feature engineering (`ml/feature_engineering.py`, `ml/physics_features.py`)

Inputs are only received measurements plus static network data and the sensor-accuracy specification; a leakage guard rejects any column containing `_true`, `attack`, `severity`, the scenario, loads, run ID or timestamp. The same function is used for training and live inference (a test checks they are identical).

| Group | Features |
|---|---|
| `bus` (120) | received V, θ, P, Q of all buses |
| `line` (287) | received P/Q at both ends, currents, loading of all lines |
| `electrical` (585) | V and θ differences across every branch; **consistency z-scores**: measured branch flows/injections minus the AC flows computed from the *measured* voltages and angles, divided by their expected noise standard deviation; KCL z-scores (injection minus the sum of measured flows); per-bus min/max/mean of incident-branch z-scores; losses; global voltage statistics |
| `temporal` (295) | changes from the previous step, rolling standard deviation and deviation from the rolling mean (window 5), in units of sensor σ; line-flow and loading changes |
| `physics` (152, **opt-in**: `features.physics_features: true`, then retrain) | **WLS state-estimation normalised residuals** of every bus sensor (signed), max residual of the flow sensors at each bus, $J(\hat x)/(m-n)$, largest residual |

The z-score normalisation uses the same σ values a WLS estimator puts in $R$. This makes the features comparable between sensor classes; without it, the first version of the detector raised an alarm on 99.9 % of normal snapshots in the high-noise stress run.

<a id="models"></a>
## 7. Random Forest models (`ml/`)

| Model | Task | Training data | Notes |
|---|---|---|---|
| A - detection | NORMAL / ATTACK + probability | all train snapshots | `class_weight='balanced'`; threshold chosen for maximum F1 on the validation runs |
| B - localisation | 30 outputs (one RF per bus) | attack-only **and** all-snapshot regimes trained and compared; the regime with better validation two-stage micro-F1 is kept | bus-local features (own + neighbours + incident branches + global), balanced classes, per-bus thresholds calibrated on validation snapshots that pass the detector, buses with too few positives are flagged instead of predicting 0 |
| C - attack type | additive / scaling / replay / coordinated | truly attacked train snapshots | simulator attack type is the target only |
| D - severity | regression of the severity defined above | truly attacked train snapshots | reported with MAE, RMSE, R², Spearman |

Inference is two-stage: the detector runs first; buses are declared compromised and the type/severity reported only when it raises an alarm. Per-bus probabilities are still computed for the digital twin. Model files store their feature list, thresholds, scikit-learn version, schema and feature version; a mismatched model is refused with a clear "retrain" message.

**Validation protocol.** Splits are by run (6 train / 2 validation / 2 test runs + 3 stress runs). Random splits of adjacent snapshots are never used, because consecutive snapshots of one attack are almost identical and would leak. For a single run, `ml.dataset.chronological_split` provides a past/future split with a gap longer than any attack.

<a id="physics"></a>
## 8. Physics-based verification (`physics/`)

Measurement model $z = h(x) + e$, state $x = [\theta_2..\theta_{30}, V_1..V_{30}]$ (59 states), 284 measurements (30 V, 30 PMU angles, 30 P, 30 Q injections, P/Q at both ends of 41 branches). The AC **WLS estimator** minimises

$J(x) = (z - h(x))^T R^{-1} (z - h(x))$

with Gauss-Newton iterations $G\,\Delta x = H^T R^{-1}(z - h(x))$, $G = H^T R^{-1} H$, using PYPOWER's `dSbus_dV` / `dSbr_dV` Jacobians. Observability is checked from the rank of $H$; an unobservable set raises an error instead of returning a fake estimate.

* **Chi-square test:** bad data if $J(\hat x) > \chi^2_{m-n,\,0.99}$. On noise-only snapshots the mean of $J$ is about $m - n = 225$ (tested), which confirms the noise model.
* **Normalised residuals:** $r_i^N = |r_i| / \sqrt{\Omega_{ii}}$, $\Omega = R - H G^{-1} H^T$; $r^N > 3$ flags a measurement.
* **LNR identification:** remove the largest normalised residual, re-estimate, repeat (only while the set stays observable).
* **Combined decision:** ATTACK if the RF probability exceeds its threshold, or if physics raises bad data and the RF gives at least half the threshold. Agreement is reported as "both detect", "ML only", "physics only" or "both normal".

**Limitation:** residual tests only see perturbations that are inconsistent with the model. An attacker who knows $H$ and injects $a = Hc$ (Liu, Ning & Reiter, 2009) leaves the residuals unchanged. The four attack types here are not built to be stealthy in that sense, so the residual tests can see them; this would not hold against a full-knowledge attacker.

<a id="mitigation"></a>
## 9. Mitigation and recovery (`physics/mitigation.py`)

Operator-supervised only - nothing is switched automatically:

1. suspicious measurements = V/θ sensors of ML-localised buses + measurements identified by LNR;
2. they are excluded from the trusted set (only if the set stays observable);
3. WLS is re-run on the trusted set → recovered state;
4. branch flows and loading are recomputed from the recovered state; excluded sensors are reconstructed as $h(\hat x)$;
5. received vs recovered values are compared (and, for offline evaluation only and clearly labelled, against simulator ground truth);
6. recommendations: quarantine sensors, use reconstructed values, widen investigation if residuals persist, voltage support, and a DC-PTDF re-dispatch suggestion (`pypower.makePTDF`) if a recovered line loading exceeds 100 %.

<a id="monitoring"></a>
## 10. Continuous monitoring, API and dashboard

**Monitor (`backend/services.py`):** a background thread in the API process polls the newest step (indexed query), and for every new snapshot computes the AI prediction, physics verification, combined decision and alert state exactly once, caching them in the `predictions` table keyed by (run ID, step). If it falls behind by more than 20 steps it jumps to the newest snapshot and counts the skipped ones as dropped. Each run uses its own sensor-accuracy specification (replays use their source run's).

**Stream status:** the simulator writes a heartbeat every step. The API reports `running`, `stale` (no heartbeat for `max(6 s, 3 × interval)`), `stopped`, `completed`, `failed` or `interrupted`; the dashboard shows a "NOT LIVE" banner whenever data is not arriving.

**API (`python cli.py api`, docs at http://127.0.0.1:8000/docs):**

| Endpoint | Content |
|---|---|
| `GET /health` | API, database, model status, stream state, monitor statistics (processed / dropped / latency) |
| `GET /latest` | latest snapshot: system totals, generators, received V/θ, ground truth, AI, physics, combined decision |
| `GET /history?groups=system,voltage,...&limit=` | time series (system, voltage, voltage_true, angle, power, loading, line_p, line_q, current, generators, labels) + AI probability history |
| `GET /network` | topology, layout, ratings, latest flows |
| `GET /buses`, `GET /lines` | per-bus / per-line received values, ground truth and AI outputs (separate fields) |
| `GET /alerts` | alert / event history |
| `GET /attack-statistics` | ground-truth events per type, per-bus counts, AI-predicted type counts |
| `GET /predictions` | AI output + live rolling comparison with ground truth |
| `GET /physics-verification` | WLS results, residuals, suspicious sensors, agreement |
| `GET /mitigation` | recovery analysis and recommendations |
| `GET /model-metrics` | offline evaluation results |
| `GET /runs` | all runs |

Errors return `{"error", "detail", "hint"}` (e.g. 404 `no_data` with "Start the simulator", 503 `models_unavailable` with "Train the models").

**Dashboard (`python cli.py dashboard`, http://127.0.0.1:8050):** refreshes every 1.5 s and only renders the visible tab.
* *Overview* - status, step, buses/lines monitored, total P/Q load, attack events, active attacks (ground truth), AI probability, detection and physics status; AI probability vs ground-truth timeline; demand/generation history; voltage heat-map of all 30 buses.
* *Digital twin* - real PYPOWER topology (schematic, not geographic), generator buses (diamonds), reference bus (square), flow direction arrows, line width ∝ |P|, overloaded lines in red, ML-predicted compromised buses filled red, ground-truth attacked buses with a black ring; click a bus for received values, AI probability/prediction and ground truth in separate tables.
* *Measurements* - voltage (all 30 buses vs truth, limits, warnings, selectable history), angles (all buses, branch angle differences, history), P/Q injections, generator outputs, bus loads, line loading with overload alerts, sending vs receiving-end P and I, selectable line history.
* *Attack monitoring* - current ground-truth attack vs AI prediction, event counts per type (truth vs predicted), separate timelines for each of the four types, per-bus attacked steps, event history table.
* *AI analytics* - offline test metrics (accuracy, precision, recall, F1, FPR, localisation micro/macro F1, attack-type accuracy), per-bus F1, confusion matrices, regime comparison - labelled OFFLINE; live rolling metrics and per-bus probabilities vs ground truth - labelled LIVE.
* *Physics verification* - received vs WLS-estimated V and θ, chi-square history vs threshold, top residuals, LNR-identified sensors, AI/physics agreement.
* *Alerts & mitigation* - alert history (time, AI type, buses, probability, severity, verification, ground-truth attack ID or "false alarm", recommended action) and the recovery analysis for the latest snapshot.

<a id="database"></a>
## 11. Database structure (`storage/database.py`)

| Table | Content |
|---|---|
| `meta` | schema version (refuses to append to a database with another column structure) |
| `runs` | run ID, mode (live/replay), purpose (live/train/validation/test/stress_*), seed, full config JSON, status, heartbeat, last step, interval, source run |
| `measurements` | one row per (run_id, step): meta, 30×4 true + 30×4 received bus values, 41×7 received + 41×7 true line values, generator outputs, bus loads, labels (938 columns). Indexes on (run_id, step), timestamp, (run_id, attack_active) |
| `attack_events` | attack ID, type, start/end step, duration, buses, channels, parameters, mean/max severity |
| `predictions` | cached AI + physics + combined payload per (run_id, step) |
| `alerts` | alert events (start/end, AI type, buses, probability, severity, verification, recommended action, ground-truth attack IDs) |
| `monitor_stats` | processed / dropped snapshots, latency |
| `hardware_telemetry` | optional MQTT telemetry |

Old data is never deleted when a new run starts; CSV exports go to `data/exports/<run_id>_schema_v2.csv` and never overwrite an existing file. Live runs are pruned only beyond `retention.live_max_steps`.

<a id="installation"></a>
## 12. Installation (Windows 10/11, Python 3.11, VS Code)

```powershell
cd C:\path\to\GridSentinel-AI
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
python cli.py check                   # verifies interpreter, PYPOWER 5.1.21 and the 30-bus case
```

In VS Code choose *Python: Select Interpreter* → `.venv`. PYPOWER is used as an installed package; its source is never modified.

<a id="running"></a>
## 13. Running the platform (PowerShell)

**First time (≈10-15 min on a 4-core laptop):**

```powershell
python cli.py generate-dataset        # 13 independent runs -> data/gridsentinel.db
python cli.py validate                # dataset validation report
python cli.py train                   # trains + evaluates all models -> models/, results/model_report.md
```

**Everything in one terminal:**

```powershell
python run_all.py                     # API + simulator + dashboard; Ctrl+C stops all cleanly
# open http://127.0.0.1:8050  (API docs: http://127.0.0.1:8000/docs)
```

**Or one service per terminal (recommended while developing):**

| Terminal | Command | Purpose |
|---|---|---|
| 1 | `python cli.py api` | FastAPI backend + monitor (start first) |
| 2 | `python cli.py simulate` | live simulator, 1 s steps (`--interval 0.5`, `--seed 7`) |
| 3 | `python cli.py dashboard` | dashboard on port 8050 |
| 4 | `python cli.py stop` | stop the simulator gracefully (or Ctrl+C in terminal 2) |

**Other commands:**

```powershell
python cli.py replay --source-run <RUN_ID> --rate 2      # dataset replay mode (same pipeline as live)
python cli.py list-runs
python cli.py export --run <RUN_ID>                      # CSV export (never overwrites)
python cli.py export --purpose test
python cli.py predict --run <RUN_ID> --step 1200         # models on one stored snapshot (+ ground truth)
python cli.py experiments                                # research experiments -> results/experiments_report.md
python cli.py experiments --quick
python cli.py network                                    # topology figure -> results/ieee30_network.html
python cli.py test                                       # automated tests
python cli.py simulate --set noise.voltage_pu=0.004      # override any config key
```

Configuration lives in `config/simulation_config.yaml` (seed, interval, load variation, noise, attack timing/probability/duration/bus counts/magnitudes, model hyper-parameters, thresholds, dashboard refresh, retention, ports).

<a id="results"></a>
## 14. Evaluation methodology and results

All numbers come from `results/model_report.md` (generated by `python cli.py train`) on **independent simulation runs**: 6 train runs (15,000 snapshots), 2 validation runs (thresholds and regime selection), 2 unseen test runs, and 3 stress runs (2× sensor noise, heavy load, half attack intensity). Each run is 2,500 snapshots with about 36 % attacked. The shipped models use the base feature set (1,287 features, no WLS-residual group).

**Model A - detection** (threshold 0.528 chosen on validation)

| split | accuracy | precision | recall | F1 | FPR | ROC AUC |
|---|---|---|---|---|---|---|
| test | 0.921 | 0.977 | 0.802 | 0.881 | 0.011 | 0.927 |
| stress: heavy load | 0.966 | 0.979 | 0.926 | 0.952 | 0.011 | 0.976 |
| stress: 2× noise | 0.870 | 0.907 | 0.688 | 0.783 | 0.036 | 0.878 |
| stress: low intensity | 0.846 | 0.970 | 0.611 | 0.750 | 0.012 | 0.855 |

Test recall by type: additive 0.92, scaling 1.00, replay 0.45, coordinated 0.85.

**Model B - 30-bus localisation** (selected regime: attack-only training, bus-local features)

| regime | test oracle micro-F1 | test oracle macro-F1 | test two-stage micro-F1 | test two-stage macro-F1 |
|---|---|---|---|---|
| attack-only (selected) | 0.592 | 0.558 | 0.608 | 0.570 |
| all-snapshot | 0.580 | 0.531 | 0.573 | 0.525 |

Validation oracle micro-F1 is 0.72, so there is a validation-to-test gap. Per-bus F1 ranges from about 0.17 (bus 3) to 0.75 (bus 1); the full table is in the report.

**Model C - attack type:** test accuracy 0.668, macro-F1 0.665 on attacked snapshots (two-stage 0.673 / 0.648). It drops to 0.41 accuracy under 2× noise and 0.22 at half intensity.

**Model D - severity:** test MAE 1.96, RMSE 3.00, R² 0.739, Spearman 0.921 (severity is the noise-normalised RMS perturbation).

**Physics (WLS):** the residual test and largest-normalised-residual identification run live on every snapshot and are reported separately from the AI and combined into a final decision.

<a id="tests"></a>
## 15. Tests

`python -m pytest -q tests` → **56 passed** (see `results/test_results.txt`). Coverage includes: case30 loading and AC power-flow convergence and balance; measurement noise and the true-vs-received split; each attack type and its labels; replay history checks; SQLite WAL concurrent write/read, schema version checks and non-overwriting CSV export; dataset validation; the feature leakage guard; run-group split disjointness; small end-to-end training and inference; WLS estimation, the chi-square test and bad-data identification; mitigation; every API endpoint (including 503 without models and stale-stream detection); and the dashboard renderers against a fake API.

<a id="limitations"></a>
## 16. Limitations

* **Simulated data only.** No field measurements or real PMU/SCADA data were used.
* **Replay attacks are hard to detect** (test recall 0.45) because replayed values are physically consistent past snapshots.
* **Low-intensity and high-noise attacks** reduce detection recall to about 0.6–0.7 and attack-type accuracy sharply.
* **Localisation is moderate** (test micro-F1 about 0.6) with a validation-to-test gap. Some buses (3, 12, 13, 14) localise poorly. The optional WLS-residual feature group targets this (the physics LNR baseline localises additive attacks far better), but it costs about 15–60 ms per snapshot to compute, so the shipped models were not retrained with it.
* **Stealthy attacks of the form a = Hc** that respect the measurement model can evade residual-based tests and consistency features by design.
* Features use the **sensor-accuracy specification** (σ per instrument class), as a WLS estimator would.
* The case30 used is the **PYPOWER/MATPOWER variant**, which has a different generator set from the original IEEE 30-bus data.
* Saved models depend on the pinned **scikit-learn version** (1.9.1). Retrain after upgrading.
* Mitigation advice is decision support for an operator. Nothing is applied automatically.

<a id="future"></a>
## 17. Future scope

* Stealthy $a = Hc$ attacks with partial network knowledge, and detectors aimed at them (temporal consistency, moving-target defence, PMU-based checks).
* Sequence models (temporal CNN / LSTM / graph neural networks on the bus graph) compared against the Random Forest.
* Robust estimators (LAV, Huber M-estimators) and optimal PMU placement for better observability.
* Larger systems (IEEE 57/118/300 are available in PYPOWER) and N-1 contingencies.
* Calibrated probability outputs (isotonic / Platt) and drift monitoring for live operation.
* Hardware-in-the-loop demonstration with the optional ESP32/MQTT telemetry gateway on an isolated low-voltage bench.

<a id="structure"></a>
## 18. Project structure

```text
GridSentinel-AI/
├── config/            settings.py (loader, overrides), simulation_config.yaml (all parameters)
├── simulator/         grid_model.py, power_flow.py, load_profiles.py, dispatch.py,
│                      measurement_generator.py, schema.py, continuous_simulator.py, network_visualization.py
├── attacks/           base.py, attack_scheduler.py, additive.py, scaling.py, replay.py, coordinated.py
├── storage/           database.py (SQLite/WAL), repository.py, dataset_validation.py
├── ml/                dataset.py, preprocessing.py, feature_engineering.py, physics_features.py,
│                      train_detection.py, train_localization.py, train_attack_type.py, train_severity.py,
│                      train_all.py, inference.py, common.py
├── physics/           state_estimation.py (AC WLS), residual_analysis.py, verification.py, mitigation.py
├── backend/           main.py (FastAPI), services.py (monitor, cache, alerts), schemas.py
├── dashboard/         app.py, layout.py, callbacks.py, charts.py, network_view.py, api_client.py, assets/style.css
├── hardware/          mqtt_gateway.py (optional), esp32/telemetry_publisher.ino
├── experiments/       evaluate_detection.py, evaluate_localization.py, evaluate_attack_types.py,
│                      benchmark_latency.py, run_all_experiments.py
├── tests/             pytest suite
├── models/            trained models + metrics/*.json
├── results/           model_report.md, experiments_report.md, tables/, figures/, test_results.txt
├── data/              gridsentinel.db (created), exports/, processed/
├── logs/
├── cli.py             single command-line entry point
├── run_all.py         one-terminal launcher
└── requirements.txt
```
