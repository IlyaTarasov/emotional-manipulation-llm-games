class NegotiationState:

    def __init__(
        self,
        agent_a_rv,
        agent_b_rv,
        max_rounds=10,
        delay_penalty=2
    ):
        self.round = 1

        self.max_rounds = max_rounds
        self.delay_penalty = delay_penalty

        self.agent_a_reservation = agent_a_rv
        self.agent_b_reservation = agent_b_rv

        self.history = []

        self.current_offer = None

        self.agreement = False
        self.final_split = None

    def apply_delay_penalty(self, value):
        penalty = (self.round - 1) * self.delay_penalty
        return max(0, value - penalty)

    def add_event(self, event):
        self.history.append(event)

    def next_round(self):
        self.round += 1

    def finished(self):
        return self.round > self.max_rounds or self.agreement