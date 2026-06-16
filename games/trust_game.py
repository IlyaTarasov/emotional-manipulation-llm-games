class TrustGame:
    def __init__(self, manipulator, victim, max_rounds=10, endowment=100, multiplier=3):
        self.manipulator = manipulator
        self.victim = victim
        self.max_rounds = max_rounds
        self.endowment = endowment
        self.multiplier = multiplier
        self.history = []
        self.verbose = False
        self.belief_every_n_rounds = 1

    def run(self):
        total_manipulator_payoff = 0
        total_victim_payoff = 0

        for round_number in range(1, self.max_rounds + 1):
            if self.verbose:
                print(f"    trust_round={round_number}", flush=True)

            message_result = self.manipulator.send_trust_message(
                self.history, round_number, self.max_rounds
            )
            manipulator_message = message_result.get("message", "")
            manipulator_message_error = self.manipulator.last_model_error
            if manipulator_message_error:
                raise RuntimeError(f"Model error while generating manipulator message in round {round_number}: {self.manipulator.last_raw_response}")

            investment_result = self.victim.choose_investment(
                manipulator_message,
                self.history,
                round_number,
                self.max_rounds,
            )
            investment = int(investment_result.get("investment", 0))
            victim_message = investment_result.get("message", "")
            victim_investment_error = self.victim.last_model_error
            if victim_investment_error:
                raise RuntimeError(f"Model error while generating victim investment in round {round_number}: {self.victim.last_raw_response}")
            investment_clipped = self.victim.last_value_clipped
            investment_parse_fallback = self.victim.last_parse_fallback

            return_result = self.manipulator.choose_return(
                investment,
                self.multiplier,
                self.history,
                round_number,
                self.max_rounds,
            )
            return_amount = int(return_result.get("return_amount", 0))
            return_message = return_result.get("message", "")
            manipulator_return_error = self.manipulator.last_model_error
            if manipulator_return_error:
                raise RuntimeError(f"Model error while generating manipulator return in round {round_number}: {self.manipulator.last_raw_response}")
            return_clipped = self.manipulator.last_value_clipped
            return_parse_fallback = self.manipulator.last_parse_fallback

            multiplied = investment * self.multiplier
            victim_payoff = self.endowment - investment + return_amount
            manipulator_payoff = multiplied - return_amount
            total_victim_payoff += victim_payoff
            total_manipulator_payoff += manipulator_payoff
            return_rate = round(return_amount / multiplied, 4) if multiplied > 0 else None
            victim_net_from_investment = return_amount - investment

            move = {
                "round": round_number,
                "manipulator_message": manipulator_message,
                "investment": investment,
                "victim_message": victim_message,
                "multiplied_investment": multiplied,
                "return_amount": return_amount,
                "return_message": return_message,
                "return_rate": return_rate,
                "victim_payoff": victim_payoff,
                "manipulator_payoff": manipulator_payoff,
                "victim_net_from_investment": victim_net_from_investment,
                "total_victim_payoff_after_round": total_victim_payoff,
                "total_manipulator_payoff_after_round": total_manipulator_payoff,
                "manipulator_message_model_error": manipulator_message_error,
                "victim_investment_model_error": victim_investment_error,
                "manipulator_return_model_error": manipulator_return_error,
                "investment_clipped": investment_clipped,
                "return_clipped": return_clipped,
                "investment_parse_fallback": investment_parse_fallback,
                "return_parse_fallback": return_parse_fallback,
            }

            should_track_belief = (
                self.belief_every_n_rounds > 0
                and (round_number % self.belief_every_n_rounds == 0 or round_number == self.max_rounds)
            )
            if should_track_belief:
                try:
                    belief = self.victim.estimate_trustworthiness(self.history + [move])
                except Exception as exc:
                    belief = {"error": str(exc)}
            else:
                belief = None
            move["victim_belief_estimate"] = belief

            self.history.append(move)

        return {
            "game_type": "repeated_trust_game",
            "rounds": len(self.history),
            "max_rounds": self.max_rounds,
            "endowment": self.endowment,
            "multiplier": self.multiplier,
            "history": self.history,
            "final_payoffs_by_role": {
                "manipulator": total_manipulator_payoff,
                "victim": total_victim_payoff,
            },
            "avg_payoffs_by_role": {
                "manipulator": total_manipulator_payoff / max(1, len(self.history)),
                "victim": total_victim_payoff / max(1, len(self.history)),
            },
            "metadata": {
                "manipulator_strategy": self.manipulator.strategy,
                "victim_strategy": self.victim.strategy,
            },
        }
