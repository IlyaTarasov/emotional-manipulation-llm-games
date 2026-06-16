import json
import random

class NegotiationGame:
    def __init__(self, agent_a, agent_b, max_rounds=15, random_start=True):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.max_rounds = max_rounds
        self.history = []
        self.random_start = random_start
        self.verbose = False
        self.belief_every_n_rounds = 1
        # Randomly determine who starts if enabled
        if random_start:
            self.agent_a_starts = random.choice([True, False])
        else:
            self.agent_a_starts = True  # Default: agent_a starts

    def run(self):
        """
        Main negotiation loop with belief tracking integration.
        """
        agreement = False
        final_split = None
        final_payoffs_by_role = None
        final_payoffs_by_agent = None

        for r in range(1, self.max_rounds + 1):
            # Determine proposer and responder
            # Round 1: random if random_start enabled, otherwise agent_a
            # Subsequent rounds: alternate
            if r == 1:
                proposer = self.agent_a if self.agent_a_starts else self.agent_b
            else:
                proposer = self.agent_b if (r % 2 == 1) != self.agent_a_starts else self.agent_a
            responder = self.agent_b if proposer is self.agent_a else self.agent_a
            if self.verbose:
                print(f"    round={r} proposer={proposer.role} responder={responder.role}", flush=True)

            # Proposer makes an offer (now returns dict with 'offer' and 'message')
            try:
                offer_result = proposer.make_offer(self.history)
                if isinstance(offer_result, dict):
                    offer = offer_result.get("offer", 50)
                    proposer_message = offer_result.get("message", "")
                else:
                    # backward compatibility
                    offer = offer_result
                    proposer_message = ""
            except Exception as e:
                print(f"Error making offer: {e}")
                offer = 50
                proposer_message = ""

            # Responder decides
            try:
                response, accepted = responder.respond_to_offer(offer, self.history, self.max_rounds)
            except Exception as e:
                print(f"Error responding to offer: {e}")
                response = "Error"
                accepted = False

            # Determine if decision is rational based on reservation value
            # is_rational = False only in two cases:
            # 1. Proposer offers a split where his payoff < his RV (irrational to offer)
            # 2. Responder accepts an offer where his payoff < his RV (irrational to accept)
            # All other cases are rational
            
            # Calculate payoffs
            proposer_payoff = 100 - offer  # what proposer gets if accepted
            responder_payoff = offer       # what responder gets if accepted
            
            # Check proposer rationality (only if offer is made)
            # Proposer is irrational if he offers a split where his payoff < his RV
            proposer_rational = (proposer_payoff >= proposer.reservation_value)
            
            # Check responder rationality (only if decision is made)
            if accepted:
                # Responder is irrational if he accepts an offer where his payoff < his RV
                responder_rational = (responder_payoff >= responder.reservation_value)
            else:
                # Rejection is always rational (can reject any offer)
                responder_rational = True
            
            # Overall move is rational if both proposer and responder acted rationally
            is_rational = proposer_rational and responder_rational
            
            # Record move information
            move = {
                "round": r,
                "proposer": proposer.name,
                "offer": offer,
                "proposer_message": proposer_message,
                "response": response,
                "accepted": accepted,
                "is_rational": is_rational,
                "raw_offer": getattr(proposer, "last_raw_offer", None),
                "offer_clipped": getattr(proposer, "last_offer_clipped", False),
                "raw_decision": getattr(responder, "last_raw_decision", None),
                "decision_overridden": getattr(responder, "last_decision_overridden", False),
                "proposer_model_error": getattr(proposer, "last_model_error", False),
                "responder_model_error": getattr(responder, "last_model_error", False),
                "proposer_role": proposer.role,
                "responder_role": responder.role,
                "proposer_rv": proposer.reservation_value,
                "responder_rv": responder.reservation_value,
                "proposer_payoff_if_accept": proposer_payoff,
                "responder_payoff_if_accept": responder_payoff,
            }

            # Request victim belief estimate (VICTIM estimates MANIPULATOR's reservation)
            # Identify victim agent (role="victim") not responder
            victim_agent = self.agent_a if self.agent_a.role == "victim" else self.agent_b
            should_track_belief = (
                self.belief_every_n_rounds > 0
                and (r % self.belief_every_n_rounds == 0 or accepted or r == self.max_rounds)
            )
            try:
                if should_track_belief and hasattr(victim_agent, "estimate_opponent_reservation"):
                    belief = victim_agent.estimate_opponent_reservation(self.history + [move])
                else:
                    belief = None
            except Exception as e:
                belief = {"error": str(e)}
            move["victim_belief_estimate"] = belief

            # Store belief in victim memory
            try:
                if victim_agent.memory_store is not None:
                    victim_agent.memory_store.add_event(f"round={r} belief={belief}")
            except Exception as e:
                print(f"Error storing belief: {e}")

            self.history.append(move)

            if accepted:
                agreement = True
                final_payoffs_by_agent = {
                    proposer.name: proposer_payoff,
                    responder.name: responder_payoff,
                }
                final_payoffs_by_role = {
                    proposer.role: proposer_payoff,
                    responder.role: responder_payoff,
                }
                final_split = final_payoffs_by_agent.copy()
                break

        # Apply penalty for negotiation breakdown
        penalty = 0
        if not agreement:
            penalty = 5  # reduced penalty points for each agent
            # Create a final split with penalty (negative payoff)
            final_split = {
                self.agent_a.name: -penalty,
                self.agent_b.name: -penalty,
            }
            final_payoffs_by_agent = final_split.copy()
            final_payoffs_by_role = {
                self.agent_a.role: -penalty,
                self.agent_b.role: -penalty,
            }
        else:
            penalty = 0

        # Identify manipulator and victim for metadata
        manipulator = self.agent_a if self.agent_a.role == "manipulator" else self.agent_b
        victim = self.agent_a if self.agent_a.role == "victim" else self.agent_b
        
        # Return game result
        result = {
            "agreement": agreement,
            "rounds": len(self.history),
            "final_split": final_split,
            "final_payoffs_by_role": final_payoffs_by_role,
            "final_payoffs_by_agent": final_payoffs_by_agent,
            "history": self.history,
            "penalty_applied": penalty if not agreement else 0,
            "metadata": {
                "manipulator_emotion": manipulator.emotion_strategy,
                "victim_emotion": victim.emotion_strategy,
                "manipulator_rv": manipulator.reservation_value,
                "victim_rv": victim.reservation_value,
            }
        }

        return result
