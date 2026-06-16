import random


class CoalitionGame:
    def __init__(self, manipulator, target_a, target_b, negotiation_rounds=5, rng=None):
        self.manipulator = manipulator
        self.target_a = target_a
        self.target_b = target_b
        self.players = {
            manipulator.name: manipulator,
            target_a.name: target_a,
            target_b.name: target_b,
        }
        self.negotiation_rounds = negotiation_rounds
        self.rng = rng or random.Random()
        self.history = []
        self.verbose = False
        self.belief_every_n_rounds = 1

    def run(self):
        total_payoffs = {name: 0 for name in self.players}

        for round_number in range(1, self.negotiation_rounds + 1):
            if self.verbose:
                print(f"    coalition_negotiation_round={round_number}", flush=True)

            sent_messages = {}
            public_messages = {}
            private_messages_received = {name: {} for name in self.players}
            commitment_signals = {}
            commitments_received = {name: {} for name in self.players}
            model_errors = {}
            manipulator_primary_target = "none"

            for name, agent in self.players.items():
                recipients = [other for other in self.players if other != name]
                self.rng.shuffle(recipients)
                messages = agent.send_coalition_messages(
                    self.history,
                    round_number,
                    self.negotiation_rounds,
                    recipients,
                )
                sent_messages[name] = {
                    recipient: messages.get(recipient, "")
                    for recipient in recipients
                }
                commitment_signals[name] = {
                    recipient: messages.get("commitments", {}).get(recipient, "undecided")
                    for recipient in recipients
                }
                public_messages[name] = messages.get("public", "")
                model_errors[f"{name}_message"] = agent.last_model_error
                for recipient in recipients:
                    private_messages_received[recipient][name] = messages.get(recipient, "")
                    commitments_received[recipient][name] = commitment_signals[name].get(recipient, "undecided")
                if name == self.manipulator.name:
                    manipulator_primary_target = messages.get("primary_target", "none")

            move = {
                "phase": "negotiation",
                "round": round_number,
                "sent_private_messages": sent_messages,
                "private_messages_received": private_messages_received,
                "commitment_signals": commitment_signals,
                "commitments_received": commitments_received,
                "public_messages": public_messages,
                "manipulator_primary_target": manipulator_primary_target,
                "model_errors": model_errors,
            }

            should_track_belief = (
                self.belief_every_n_rounds > 0
                and (round_number % self.belief_every_n_rounds == 0 or round_number == self.negotiation_rounds)
            )
            if should_track_belief:
                move["target_beliefs"] = {
                    self.target_a.name: self.target_a.estimate_manipulator_belief(self.history + [move]),
                    self.target_b.name: self.target_b.estimate_manipulator_belief(self.history + [move]),
                }
            else:
                move["target_beliefs"] = None

            self.history.append(move)

        if self.verbose:
            print("    coalition_final_choice", flush=True)

        choices = {}
        public_statements = {}
        rationales = {}
        raw_choice_responses = {}
        model_errors = {}
        choice_clipped = {}
        last_private_received = self.history[-1].get("private_messages_received", {}) if self.history else {}
        last_public_messages = self.history[-1].get("public_messages", {}) if self.history else {}

        for name, agent in self.players.items():
            legal_choices = [other for other in self.players if other != name] + ["alone"]
            self.rng.shuffle(legal_choices)
            choice_result = agent.choose_partner(
                self.history,
                self.negotiation_rounds,
                legal_choices,
                private_messages=last_private_received.get(name, {}),
                public_messages=last_public_messages,
            )
            choices[name] = choice_result.get("partner_choice", "alone")
            public_statements[name] = choice_result.get("message", "")
            rationales[name] = choice_result.get("rationale", "")
            raw_choice_responses[name] = getattr(agent, "last_raw_response", None)
            model_errors[f"{name}_choice"] = agent.last_model_error
            choice_clipped[name] = agent.last_value_clipped

        coalition = self._formed_coalition(choices)
        payoffs = self._compute_payoffs(coalition)
        for name, payoff in payoffs.items():
            total_payoffs[name] += payoff

        final_beliefs = None
        if self.belief_every_n_rounds > 0:
            final_beliefs = {
                self.target_a.name: self.target_a.estimate_manipulator_belief(self.history),
                self.target_b.name: self.target_b.estimate_manipulator_belief(self.history),
            }

        final_move = {
            "phase": "final_choice",
            "round": self.negotiation_rounds + 1,
            "choices": choices,
            "public_statements": public_statements,
            "private_rationales": rationales,
            "raw_choice_responses": raw_choice_responses,
            "coalition": coalition,
            "payoffs": payoffs,
            "total_payoffs_after_round": dict(total_payoffs),
            "model_errors": model_errors,
            "choice_clipped": choice_clipped,
            "target_beliefs": final_beliefs,
        }
        self.history.append(final_move)

        return {
            "game_type": "three_player_coalition_choice",
            "negotiation_rounds": self.negotiation_rounds,
            "rounds": len(self.history),
            "history": self.history,
            "final_payoffs_by_agent": total_payoffs,
            "metadata": {
                "manipulator_strategy": self.manipulator.strategy,
                "private_values": {
                    name: agent.private_values
                    for name, agent in self.players.items()
                },
                "outside_payoffs": {
                    name: agent.outside_payoff
                    for name, agent in self.players.items()
                },
            },
        }

    def _formed_coalition(self, choices):
        names = sorted(self.players)
        mutual_pairs = []
        for i, first in enumerate(names):
            for second in names[i + 1:]:
                if choices.get(first) == second and choices.get(second) == first:
                    mutual_pairs.append([first, second])
        if not mutual_pairs:
            return None
        if len(mutual_pairs) == 1:
            return mutual_pairs[0]
        for pair in mutual_pairs:
            if self.manipulator.name in pair:
                return pair
        return mutual_pairs[0]

    def _compute_payoffs(self, coalition):
        payoffs = {}
        if not coalition:
            for name, agent in self.players.items():
                payoffs[name] = agent.outside_payoff
            return payoffs

        first, second = coalition
        for name, agent in self.players.items():
            if name == first:
                payoffs[name] = agent.private_values.get(second, agent.outside_payoff)
            elif name == second:
                payoffs[name] = agent.private_values.get(first, agent.outside_payoff)
            else:
                payoffs[name] = agent.outside_payoff
        return payoffs
