import re
from deeppdcfr.lookup import lookup


def get_hand_strength(info):
    private_card = info["private_cards"]
    public_card = info["public_cards"]
    hand_strength = lookup.calc(private_card, public_card) * 1326 - 663
    return hand_strength


class RuleAgent:
    def __call__(self, state):
        info = self.parse_state(state)
        policy = self.get_policy(info)
        assert sum(list(policy.values())) == 1
        return policy

    def parse_state(self, state):
        info_str = state.information_state_string()
        private = re.search(r"\[Private: ([^\]]+)\]", info_str).group(1)
        public = re.search(r"\[Public: ([^\]]+)\]", info_str)
        private_cards = [private[i : i + 2] for i in range(0, len(private), 2)]
        public_cards = (
            public
            and [public.group(1)[i : i + 2] for i in range(0, len(public.group(1)), 2)]
            or []
        )
        action_strings = ["fold", "call", "raise"]
        legal_actions = [action_strings[action] for action in state.legal_actions()]
        info = {
            "private_cards": private_cards,
            "public_cards": public_cards,
            "legal_actions": legal_actions,
        }
        return info

    def get_policy(self, info):
        raise NotImplementedError

    def pick_first_legal_action(self, info, action_list):
        for action in action_list:
            if action in info["legal_actions"]:
                return action


class HandStrengthAgent(RuleAgent):
    def get_policy(self, info):
        action = self.get_action(info)
        policy = {}
        action_dict = {
            "fold": 0,
            "call": 1,
            "raise": 2,
        }
        policy[action_dict[action]] = 1
        return policy

    def get_action(self, info):
        raise NotImplementedError

    def get_hand_strength_band(self):
        raise NotImplementedError

    def get_action(self, info):
        hand_strength_band_low, hand_strength_band_high = self.get_hand_strength_band()
        hand_strength = get_hand_strength(info)
        if hand_strength < hand_strength_band_low:
            return self.pick_first_legal_action(info, ["fold", "call"])
        elif hand_strength < hand_strength_band_high:
            return "call"
        else:
            return self.pick_first_legal_action(info, ["raise", "call"])


class CandidStatistician(HandStrengthAgent):
    def get_hand_strength_band(self):
        return (-100, 100)


class LooseAggressive(HandStrengthAgent):
    def get_hand_strength_band(self):
        return (-100, -300)


class LoosePassive(HandStrengthAgent):
    def get_hand_strength_band(self):
        return (-300, 500)


class TightPassive(HandStrengthAgent):
    def get_hand_strength_band(self):
        return (100, 500)


class TightAggressive(RuleAgent):
    hand_strength_band_high = 100
    hand_strength_band_low = -100
    bluffing_rate = 0.2

    def get_policy(self, info):
        hand_strength = get_hand_strength(info)
        if hand_strength < self.hand_strength_band_low:
            if "raise" in info["legal_actions"]:
                action = self.pick_first_legal_action(info, ["fold", "call"])
                policy = {
                    "raise": self.bluffing_rate,
                    action: 1 - self.bluffing_rate,
                }
            else:
                action = self.pick_first_legal_action(info, ["fold", "call"])
                policy = {action: 1}

        elif hand_strength < self.hand_strength_band_high:
            policy = {"call": 1}
        else:
            action = self.pick_first_legal_action(info, ["raise", "call"])
            policy = {action: 1}

        action_dict = {
            "fold": 0,
            "call": 1,
            "raise": 2,
        }
        policy = {action_dict[action]: prob for action, prob in policy.items()}
        return policy
