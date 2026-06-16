#!/usr/bin/env python3
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.agent import NegotiationAgent
from games.negotiation_game import NegotiationGame
from utils.model_interface import OllamaModel


EMOTIONS = [
    "neutral_control",
    "fake_anger",
    "fake_disappointment",
    "fake_urgency",
    "fake_confidence",
]


DEFAULT_MODELS = ["llama3.1:8b", "qwen2.5", "gemma3:4b"]


def parse_list(env_name, default):
    raw = os.environ.get(env_name)
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


def ordered_pairs(models):
    return [(a, b) for a in models for b in models if a != b]


def manipulator_offers(result):
    return [
        move.get("offer") for move in result.get("history", [])
        if move.get("proposer_role") == "manipulator"
    ]


def count_model_errors(result):
    return sum(
        bool(move.get("proposer_model_error")) or bool(move.get("responder_model_error"))
        for move in result.get("history", [])
    )


def run_game(model_manipulator, model_victim, emotion, max_rounds, manipulator_starts, belief_every_n_rounds):
    manipulator = NegotiationAgent(
        name="manipulator",
        model=OllamaModel(model_name=model_manipulator, temperature=0.2),
        role="manipulator",
        emotion_strategy=emotion,
        reservation_value=30,
        memory_store=None,
    )
    victim = NegotiationAgent(
        name="victim",
        model=OllamaModel(model_name=model_victim, temperature=0.2),
        role="victim",
        emotion_strategy="neutral_control",
        reservation_value=45,
        memory_store=None,
    )
    game = NegotiationGame(manipulator, victim, max_rounds=max_rounds, random_start=False)
    game.agent_a_starts = bool(manipulator_starts)
    game.verbose = os.environ.get("VERBOSE_ROUNDS", "0") == "1"
    game.belief_every_n_rounds = belief_every_n_rounds
    return game.run()


def summarize(results):
    groups = defaultdict(list)
    for result in results:
        groups[
            (
                result["model_pair"][0],
                result["model_pair"][1],
                result["emotion_manipulator"],
            )
        ].append(result)

    rows = []
    for (manipulator_model, victim_model, emotion), group in sorted(groups.items()):
        agreements = [bool(result.get("agreement")) for result in group]
        rounds = [int(result.get("rounds") or 0) for result in group]
        manip_payoffs = [
            result.get("final_payoffs_by_role", {}).get("manipulator", 0)
            for result in group
        ]
        victim_payoffs = [
            result.get("final_payoffs_by_role", {}).get("victim", 0)
            for result in group
        ]
        model_errors = [count_model_errors(result) for result in group]
        all_offers = [manipulator_offers(result) for result in group]
        rows.append({
            "manipulator_model": manipulator_model,
            "victim_model": victim_model,
            "emotion": emotion,
            "games": len(group),
            "agreement_rate": sum(agreements) / len(group),
            "avg_rounds": sum(rounds) / len(rounds),
            "avg_manipulator_payoff": sum(manip_payoffs) / len(manip_payoffs),
            "avg_victim_payoff": sum(victim_payoffs) / len(victim_payoffs),
            "model_error_moves": sum(model_errors),
            "avg_manipulator_offer_count": sum(len(offers) for offers in all_offers) / len(all_offers),
            "avg_first_manipulator_offer": sum((offers[0] if offers else 0) for offers in all_offers) / len(all_offers),
            "avg_last_manipulator_offer": sum((offers[-1] if offers else 0) for offers in all_offers) / len(all_offers),
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
    models = parse_list("MODELS", DEFAULT_MODELS)
    emotions = parse_list("EMOTIONS", EMOTIONS)
    repeats = int(os.environ.get("REPEATS_PER_CONDITION", "5"))
    max_rounds = int(os.environ.get("MAX_ROUNDS", "10"))
    belief_every_n_rounds = int(os.environ.get("BELIEF_EVERY_N_ROUNDS", "1"))

    pairs = ordered_pairs(models)
    total = len(pairs) * len(emotions) * repeats * 2
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json = f"final_balanced_v6_{timestamp}.json"
    output_csv = f"final_balanced_v6_summary_{timestamp}.csv"

    payload = {
        "test_config": {
            "timestamp": timestamp,
            "test_name": "final balanced v6 negotiation experiment",
            "models": models,
            "model_pairs": pairs,
            "emotions": emotions,
            "repeats_per_condition": repeats,
            "start_counterbalancing": "each repeat has manipulator-start and victim-start games",
            "max_rounds": max_rounds,
            "belief_every_n_rounds": belief_every_n_rounds,
            "temperature": 0.2,
            "memory": "off; independent episodes",
            "reservation_values": {"manipulator": 30, "victim": 45},
        },
        "results": [],
    }

    index = 0
    for model_manipulator, model_victim in pairs:
        for emotion in emotions:
            for repeat in range(1, repeats + 1):
                for manipulator_starts in [True, False]:
                    index += 1
                    print(
                        f"[{index}/{total}] manipulator={model_manipulator} "
                        f"victim={model_victim} emotion={emotion} repeat={repeat} "
                        f"manipulator_starts={manipulator_starts}",
                        flush=True,
                    )
                    result = run_game(
                        model_manipulator=model_manipulator,
                        model_victim=model_victim,
                        emotion=emotion,
                        max_rounds=max_rounds,
                        manipulator_starts=manipulator_starts,
                        belief_every_n_rounds=belief_every_n_rounds,
                    )
                    result["model_pair"] = [model_manipulator, model_victim]
                    result["emotion_manipulator"] = emotion
                    result["emotion_victim"] = "neutral_control"
                    result["repeat_id"] = repeat
                    result["manipulator_starts"] = manipulator_starts
                    result["reservation_values"] = {"manipulator": 30, "victim": 45}
                    payload["results"].append(result)

                    offers = manipulator_offers(result)
                    errors = count_model_errors(result)
                    print(
                        f"  agreement={result.get('agreement')} rounds={result.get('rounds')} "
                        f"payoffs={result.get('final_payoffs_by_role')} "
                        f"manip_offers={offers} model_errors={errors}",
                        flush=True,
                    )

                    summary_rows = summarize(payload["results"])
                    payload["summary"] = summary_rows
                    with open(output_json, "w", encoding="utf-8") as handle:
                        json.dump(payload, handle, indent=2, ensure_ascii=False)
                    write_csv(output_csv, summary_rows)

    print(f"Saved {output_json}")
    print(f"Saved {output_csv}")


if __name__ == "__main__":
    main()
