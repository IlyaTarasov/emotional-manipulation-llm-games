#!/usr/bin/env python3
import csv
import json
import os
from collections import defaultdict
from datetime import datetime

from agents.trust_agent import TrustAgent
from games.trust_game import TrustGame
from prompts.trust_game_prompts import TRUST_STRATEGIES
from utils.model_interface import OllamaModel


DEFAULT_MODELS = ["llama3.1:8b", "mistral", "qwen2.5", "granite3.3:8b"]


def parse_list(name, default):
    raw = os.environ.get(name)
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


def ordered_pairs(models):
    return [(a, b) for a in models for b in models if a != b]


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else 0.0


def model_errors(result):
    return sum(
        bool(move.get("manipulator_message_model_error"))
        or bool(move.get("victim_investment_model_error"))
        or bool(move.get("manipulator_return_model_error"))
        for move in result.get("history", [])
    )


def quality_events(result):
    return {
        "model_errors": model_errors(result),
        "investment_clipped": sum(bool(move.get("investment_clipped")) for move in result.get("history", [])),
        "return_clipped": sum(bool(move.get("return_clipped")) for move in result.get("history", [])),
        "investment_parse_fallback": sum(bool(move.get("investment_parse_fallback")) for move in result.get("history", [])),
        "return_parse_fallback": sum(bool(move.get("return_parse_fallback")) for move in result.get("history", [])),
    }


def run_game(model_manipulator, model_victim, strategy, max_rounds, belief_every_n_rounds, endowment, multiplier):
    manipulator = TrustAgent(
        name="manipulator",
        model=OllamaModel(model_name=model_manipulator, temperature=0.2, num_predict=260),
        role="manipulator",
        strategy=strategy,
    )
    victim = TrustAgent(
        name="victim",
        model=OllamaModel(model_name=model_victim, temperature=0.2, num_predict=260),
        role="victim",
        strategy="neutral_victim",
    )
    game = TrustGame(
        manipulator=manipulator,
        victim=victim,
        max_rounds=max_rounds,
        endowment=endowment,
        multiplier=multiplier,
    )
    game.verbose = os.environ.get("VERBOSE_ROUNDS", "0") == "1"
    game.belief_every_n_rounds = belief_every_n_rounds
    return game.run()


def summarize(results):
    groups = defaultdict(list)
    for result in results:
        groups[
            (
                result["manipulator_model"],
                result["victim_model"],
                result["manipulator_strategy"],
            )
        ].append(result)

    rows = []
    for (manipulator_model, victim_model, strategy), group in sorted(groups.items()):
        investments = []
        returns = []
        return_rates = []
        victim_payoffs = []
        manipulator_payoffs = []
        trust_deltas = []
        model_error_moves = 0
        belief_events = 0
        manipulation_detected = []

        for result in group:
            model_error_moves += model_errors(result)
            history = result.get("history", [])
            inv = [move.get("investment") for move in history if isinstance(move.get("investment"), int)]
            if len(inv) >= 2:
                trust_deltas.append(inv[-1] - inv[0])
            for move in history:
                investments.append(move.get("investment"))
                returns.append(move.get("return_amount"))
                return_rates.append(move.get("return_rate"))
                victim_payoffs.append(move.get("victim_payoff"))
                manipulator_payoffs.append(move.get("manipulator_payoff"))
                belief = move.get("victim_belief_estimate")
                if isinstance(belief, dict):
                    belief_events += 1
                    if "manipulation_detected" in belief:
                        manipulation_detected.append(bool(belief.get("manipulation_detected")))

        rows.append({
            "manipulator_model": manipulator_model,
            "victim_model": victim_model,
            "strategy": strategy,
            "games": len(group),
            "avg_investment": round(mean(investments), 4),
            "avg_return": round(mean(returns), 4),
            "avg_return_rate": round(mean(return_rates), 4),
            "avg_victim_round_payoff": round(mean(victim_payoffs), 4),
            "avg_manipulator_round_payoff": round(mean(manipulator_payoffs), 4),
            "avg_manipulator_advantage": round(mean(manipulator_payoffs) - mean(victim_payoffs), 4),
            "avg_trust_delta": round(mean(trust_deltas), 4),
            "model_error_moves": model_error_moves,
            "belief_events": belief_events,
            "manipulation_detected_rate": round(mean(manipulation_detected), 4),
        })
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    models = parse_list("TRUST_MODELS", DEFAULT_MODELS)
    strategies = parse_list("TRUST_STRATEGIES", TRUST_STRATEGIES)
    repeats = int(os.environ.get("TRUST_REPEATS_PER_CONDITION", "3"))
    max_rounds = int(os.environ.get("TRUST_MAX_ROUNDS", "10"))
    endowment = int(os.environ.get("TRUST_ENDOWMENT", "100"))
    multiplier = int(os.environ.get("TRUST_MULTIPLIER", "3"))
    belief_every_n_rounds = int(os.environ.get("BELIEF_EVERY_N_ROUNDS", "2"))
    max_allowed_model_errors = int(os.environ.get("MAX_ALLOWED_MODEL_ERRORS", "0"))
    max_allowed_CLIPPED = int(os.environ.get("MAX_ALLOWED_CLIPPED", "0"))
    max_allowed_parse_fallback = int(os.environ.get("MAX_ALLOWED_PARSE_FALLBACK", "0"))

    pairs = ordered_pairs(models)
    total = len(pairs) * len(strategies) * repeats
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json = f"final_trust_{timestamp}.json"
    output_csv = f"final_trust_summary_{timestamp}.csv"

    payload = {
        "test_config": {
            "timestamp": timestamp,
            "test_name": "final repeated trust game experiment",
            "models": models,
            "model_pairs": pairs,
            "strategies": strategies,
            "repeats_per_condition": repeats,
            "max_rounds": max_rounds,
            "endowment": endowment,
            "multiplier": multiplier,
            "belief_every_n_rounds": belief_every_n_rounds,
            "temperature": 0.2,
            "memory": "off; independent episodes",
            "quality_thresholds": {
                "max_allowed_model_errors": max_allowed_model_errors,
                "max_allowed_clipped": max_allowed_CLIPPED,
                "max_allowed_parse_fallback": max_allowed_parse_fallback,
            },
        },
        "results": [],
    }

    index = 0
    cumulative_quality = {
        "model_errors": 0,
        "investment_clipped": 0,
        "return_clipped": 0,
        "investment_parse_fallback": 0,
        "return_parse_fallback": 0,
    }
    for model_manipulator, model_victim in pairs:
        for strategy in strategies:
            for repeat in range(1, repeats + 1):
                index += 1
                print(
                    f"[{index}/{total}] manipulator={model_manipulator} victim={model_victim} "
                    f"strategy={strategy} repeat={repeat}",
                    flush=True,
                )
                result = run_game(
                    model_manipulator,
                    model_victim,
                    strategy,
                    max_rounds=max_rounds,
                    belief_every_n_rounds=belief_every_n_rounds,
                    endowment=endowment,
                    multiplier=multiplier,
                )
                result["manipulator_model"] = model_manipulator
                result["victim_model"] = model_victim
                result["model_pair"] = [model_manipulator, model_victim]
                result["manipulator_strategy"] = strategy
                result["victim_strategy"] = "neutral_victim"
                result["repeat_id"] = repeat
                result["quality_events"] = quality_events(result)
                payload["results"].append(result)
                for key, value in result["quality_events"].items():
                    cumulative_quality[key] += value
                payload["cumulative_quality"] = cumulative_quality

                investments = [move.get("investment") for move in result.get("history", [])]
                returns = [move.get("return_amount") for move in result.get("history", [])]
                print(
                    f"  avg_payoffs={result.get('avg_payoffs_by_role')} "
                    f"investments={investments} returns={returns} quality={result['quality_events']}",
                    flush=True,
                )

                payload["summary"] = summarize(payload["results"])
                with open(output_json, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2, ensure_ascii=False)
                write_csv(output_csv, payload["summary"])

                clipped_total = cumulative_quality["investment_clipped"] + cumulative_quality["return_clipped"]
                fallback_total = cumulative_quality["investment_parse_fallback"] + cumulative_quality["return_parse_fallback"]
                if cumulative_quality["model_errors"] > max_allowed_model_errors:
                    raise RuntimeError(f"Stopping: model error threshold exceeded: {cumulative_quality}")
                if clipped_total > max_allowed_CLIPPED:
                    raise RuntimeError(f"Stopping: clipping threshold exceeded: {cumulative_quality}")
                if fallback_total > max_allowed_parse_fallback:
                    raise RuntimeError(f"Stopping: parse fallback threshold exceeded: {cumulative_quality}")

    print(f"Saved {output_json}")
    print(f"Saved {output_csv}")


if __name__ == "__main__":
    main()
