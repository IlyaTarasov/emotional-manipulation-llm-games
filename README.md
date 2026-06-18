# Emotional Manipulation in Sequential LLM Games

This repository contains the experimental framework and final result artifacts for a term-paper project on multi-agent game-theoretic modeling of emotional manipulation in LLM agents.

The project studies affective-strategic manipulation in three sequential games:

1. repeated bilateral negotiation with hidden reservation values;
2. repeated trust game;
3. three-player coalition choice with private communication.

The framework is intentionally narrower than general emotional reasoning benchmarks. It focuses on repeated social influence, belief change, trust repair, private commitments, and coalition persuasion.

## Repository Structure

```text
agents/        LLM agent wrappers for the three games
games/         game environments and state transitions
prompts/       final role-specific prompt protocols
utils/         Ollama model interface
results/final/ final logs, summary tables, confidence intervals, and process checks
```

## Requirements

- Python 3.10+
- Ollama running locally
- Models used in final experiments should be installed in Ollama before re-running experiments.

Install Python dependencies:

```bash
pip install -r requirements.txt
```

The code uses local Ollama chat completion through `utils/model_interface.py`.

## Final Experiment Commands

Negotiation:

```powershell
python run_negotiation_experiment.py
```

Trust game:

```powershell
python run_trust_experiment.py
```

Coalition game:

```powershell
python run_coalition_experiment.py
```

The scripts write timestamped logs and summary CSV files to the working directory unless environment variables are used to override run parameters.

## Final Results and Logs

The `results/final/` directory contains the final artifacts used in the term paper.

Main compact tables:

- `experiment1_negotiation/experiment_1_strategy_summary.csv`;
- `experiment2_trust/experiment_2_strategy_summary.csv`;
- `experiment3_coalition/experiment_3_summary.csv`;
- `summary/cross_game_summary.csv`.

Full episode logs:

- `experiment1_negotiation/experiment_1_full_log.json`;
- `experiment2_trust/experiment_2_full_log.json`;
- `experiment3_coalition/experiment_3_full_log.json`.

These JSON files contain the final episode histories: model assignments, strategy labels, structured actions, natural-language messages, payoffs, belief-tracking fields, and technical quality flags.

Additional derived artifacts:

- `summary/key_metric_confidence_intervals.csv` contains the main 95% bootstrap confidence intervals used in the paper;
- `summary/bootstrap_confidence_intervals_exploratory.csv` contains a broader exploratory CI table;
- `summary/model_level_confidence_intervals.csv` contains model-level bootstrap intervals;
- `summary/negotiation_round_trajectories.csv`, `summary/trust_round_trajectories.csv`, and `summary/coalition_round_trajectories.csv` contain round-level process summaries;
- `summary/experiment_parameter_check.csv` records final run parameters;
- `summary/negotiation_target_perception_proxy.csv`, `summary/negotiation_adherence_proxy.csv`, and `summary/trust_adherence_proxy.csv` are exploratory proxy checks. They are included for transparency and should not be treated as independent human-coded manipulation checks.

Experiment-specific supplementary tables are stored next to the corresponding full logs. They include model-level summaries, scenario-level summaries, pair-by-strategy summaries, and quality reports where applicable.

## Notes on Reproducibility

LLM outputs are stochastic even at low temperature, and local Ollama model versions may differ across machines. The included JSON logs and CSV files document the final runs used in the paper. The scripts provide the reproducible experimental protocol, while the included logs provide the exact final artifacts analyzed in the submitted version.
