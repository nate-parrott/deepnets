from math import log
from collections import defaultdict

START_STATE = '$START'
END_STATE = '$END'
UNK = '$UNK'
unk_threshold = 2

def smooth_prob(p):
    return p * (1 - 0.001) + 0.001

class ProbabilityCounter(object):
    def __init__(self):
        self.counts = defaultdict(int)
        self.total = 0
    
    def insert(self, item):
        self.counts[item] += 1
        self.total += 1
    
    def prob(self, item):
        if item in self.counts:
            return log(smooth_prob(self.counts[item] * 1.0 / self.total))
        else:
            return log(smooth_prob(0))

def hmm(transition_probs, token_probs, states, tokens):
    candidates = [(0, [START_STATE])]
    for token in tokens:
        best_candidates_by_state = {}
        for to_state in states:
            for candidate_prob, candidate_states in candidates:
                old_state = candidate_states[-1]
                new_prob = candidate_prob + transition_probs[old_state].prob(to_state) + token_probs[to_state].prob(token)
                # new_prob = candidate_prob + token_probs[to_state].prob(token)
                if to_state not in best_candidates_by_state or new_prob > best_candidates_by_state[to_state][0]:
                    best_candidates_by_state[to_state] = (new_prob, candidate_states + [to_state])
        candidates = best_candidates_by_state.values()
    # print candidates
    return max(candidates, key=lambda cand: cand[0])[1][1:]

def add_unks(training):
    word_counts = defaultdict(int)
    for para, _ in training:
        for word in para:
            word_counts[word] += 1
    def unk_para(para):
        return [(word if word_counts[word] >= unk_threshold else UNK) for word in para]
    return [(unk_para(para), states) for para, states in training]

def hmm_model(training):
    training = add_unks(training)
    transitions = defaultdict(ProbabilityCounter)
    token_probs = defaultdict(ProbabilityCounter)
    all_states = set()
    for paragraph, states in training:
        states_plus = [START_STATE] + states + [END_STATE]
        for prev_state, next_state in zip(states_plus[:-1], states_plus[1:]):
            transitions[prev_state].insert(next_state)
        for token, state in zip(paragraph, states):
            token_probs[state].insert(token)
            all_states.add(state)
    
    def model(tokens):
        return hmm(transitions, token_probs, all_states, tokens)
    
    return model
