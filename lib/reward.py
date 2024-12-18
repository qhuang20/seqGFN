

"""Reward function"""

# target_sequences = [['C'], ['C', 'A'], ['C', 'B'], ['C', 'C']]
target_sequences = [['A', 'B', 'B', 'C'], 
                    ['A', 'B', 'C', 'ε'], 
                    ['C', 'A', 'C', 'C'], 
                    ['C', 'B', 'A', 'ε'], 
                    ['C', 'C', 'B', 'A'], 
                    ['C', 'C', 'C', 'A']] 

replay_buffer = [
    [([0, ['ε', 'ε', 'ε', 'ε']], 0),
     ([1, ['A', 'ε', 'ε', 'ε']], 2), 
     ([2, ['C', 'A', 'ε', 'ε']], 2),
     ([3, ['C', 'C', 'A', 'ε']], 20),
     ([4, ['C', 'B', 'A', 'ε']], -1),
    ],
    [([0, ['ε', 'ε', 'ε', 'ε']], 0),
     ([1, ['A', 'ε', 'ε', 'ε']], 2),
     ([2, ['C', 'A', 'ε', 'ε']], 2), 
     ([3, ['C', 'C', 'A', 'ε']], 2),
     ([4, ['C', 'C', 'C', 'A']], -1),
    ],
]

def sequence_reward(seq):
    # Check if sequence matches any target sequence
    if seq in target_sequences:
        return 1.0
    return r_min

# def sequence_reward(seq):
#     # Count number of 'A's in sequence
#     num_As = seq.count('A')
#     if num_As > 0:
#         return float(num_As)  # Return number of A's as reward
#     return r_min


