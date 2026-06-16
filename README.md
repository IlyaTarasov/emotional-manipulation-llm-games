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
results/final/ compact final summary tables
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

## Final Results

The `results/final/` directory contains:

- `experiment1_negotiation/experiment_1_strategy_summary.csv`;
- `experiment2_trust/experiment_2_strategy_summary.csv`;
- `experiment3_coalition/experiment_3_summary.csv`;
- `summary/cross_game_summary.csv`.

## Notes on Reproducibility

LLM outputs are stochastic even at low temperature, and local Ollama model versions may differ across machines. The included CSV files are compact summaries of the final runs used in the paper. The scripts provide the reproducible experimental protocol.

## License

Add a license before public release if the repository is made public.
