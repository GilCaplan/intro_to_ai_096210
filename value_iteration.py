import json
import os
import hashlib


class ValueIterationCache:
    def __init__(self, cache_dir="vi_cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _generate_state_hash(self, initial_state):
        """Generate a unique hash for the initial state configuration"""
        # Convert the initial state to a stable string representation
        state_str = json.dumps(initial_state, sort_keys=True)
        return hashlib.md5(state_str.encode()).hexdigest()

    def _get_cache_path(self, state_hash, rounds):
        """Get the cache file path for a specific state and number of rounds"""
        return os.path.join(self.cache_dir, f"vi_{state_hash}_{rounds}.json")

    def load_cached_values(self, initial_state, rounds):
        """Load cached Value Iteration results if they exist"""
        state_hash = self._generate_state_hash(initial_state)
        cache_path = self._get_cache_path(state_hash, rounds)

        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def save_values(self, initial_state, rounds, values):
        """Save Value Iteration results to cache"""
        state_hash = self._generate_state_hash(initial_state)
        cache_path = self._get_cache_path(state_hash, rounds)

        try:
            with open(cache_path, 'w') as f:
                json.dump(values, f)
            return True
        except IOError:
            return False


# Modified OptimalWizardAgent class methods
def Value_Iteration(self):
    cache = ValueIterationCache()
    cached_values = cache.load_cached_values(self.initial, self.rounds)

    if cached_values is not None:
        return json.dumps(cached_values)

    # Original Value Iteration calculation
    V = []
    for t in range(self.rounds + 1):
        vs = {}
        for state in self.all_states:
            state_key = str(state_to_key(state))
            actions = self.get_actions(state)
            action_values = []

            for action in actions:
                immediate_reward = calculate_reward(state, action)
                new_states, probs = self.apply_action(state, action)

                future_value = 0
                if t > 0:
                    for n_state, prob in zip(new_states, probs):
                        next_key = str(state_to_key(n_state))
                        future_value += prob * V[t - 1][next_key]['score']

                if action == ('termination',):
                    future_value = 0
                elif action == ('reset',):
                    if t > 0:
                        initial_key = str(state_to_key(self.start_state))
                        future_value = V[t - 1][initial_key]['score']

                total_value = immediate_reward + future_value
                action_values.append((action, total_value))

            best_action, max_score = max(action_values, key=lambda x: x[1])
            vs[state_key] = {'action': best_action, 'score': max_score}
        V.append(vs)

    # Cache the results before returning
    cache.save_values(self.initial, self.rounds, V)
    return json.dumps(V)