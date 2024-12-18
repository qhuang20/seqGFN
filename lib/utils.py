
# Global constants
alphabet = ['A', 'B', 'C']
n_timesteps = 5  # t=0 to t=4 
vocab_size = len(alphabet) 
max_len = n_timesteps - 1
max_actions = max_len * vocab_size + max_len + max_len * vocab_size


actions_list = []
# Insertions
for pos in range(max_len):
    for char in alphabet:
        actions_list.append(('insert', pos, char))
# Deletions 
for pos in range(max_len):
    actions_list.append(('delete', pos))
# Mutations
for pos in range(max_len):
    for char in alphabet:
        actions_list.append(('mutate', pos, char))

print("Table of actions:")
print(actions_list)
print(len(actions_list))




def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
    
def calculate_forward_mask_from_state(seq):
    """Here, we mask forward actions to prevent the selection of invalid configurations."""
    mask = np.zeros(max_actions)
    
    seq_len = len([x for x in seq if x != 'ε'])
    
    #  insertions
    if seq_len < max_len:
        for pos in range(seq_len + 1):
            for char_idx, char in enumerate(['A', 'B', 'C']):
                action_idx = pos * vocab_size + char_idx
                mask[action_idx] = 1
                
    #  deletions
    deletion_offset = vocab_size * max_len
    if seq_len > 0:
        for pos in range(seq_len):
            mask[deletion_offset + pos] = 1
            
    #  mutations
    mutation_offset = deletion_offset + max_len
    for pos in range(seq_len):
        current_char = seq[pos]
        if current_char != 'ε':
            for char_idx, char in enumerate(['A', 'B', 'C']):
                action_idx = mutation_offset + pos * vocab_size + char_idx
                mask[action_idx] = 1

    return torch.Tensor(mask).bool()



def calculate_backward_mask_from_state(timestep, seq):
    """Calculate backward mask considering the timestep and possible parent states.
    
    Args:
        timestep: Current timestep (0-based)
        seq: List of characters representing current sequence
    """
    seq_len = len([c for c in seq if c != 'ε'])  # Current sequence length
    
    mask = [0] * max_actions
    
    # Calculate offsets
    insertion_offset = 0
    deletion_offset = max_len * vocab_size
    mutation_offset = deletion_offset + max_len

    # At t=0, no backward actions possible (root state)
    if timestep == 0:
        return torch.Tensor(mask).bool()
    
    max_prev_len = timestep - 1
    
    # Special case: at t=2 with one character sequence, only mutations allowed
    if timestep == 2 and seq_len == 1:
        # Handle mutations only
        for pos, char in enumerate(seq):
            if char != 'ε':
                for char_idx, new_char in enumerate(['A', 'B', 'C']):
                    mask[mutation_offset + pos * vocab_size + char_idx] = 1
        return torch.Tensor(mask).bool()
    
    # If current sequence length > max_prev_len, only deletions possible
    if seq_len > max_prev_len:
        # Handle deletions - any non-empty position could have been deleted
        for pos, char in enumerate(seq):
            if char != 'ε':
                mask[deletion_offset + pos] = 1
                
    # If current sequence length == max_prev_len, mutations and deletions possible
    elif seq_len == max_prev_len:
        # Handle deletions 
        for pos, char in enumerate(seq):
            if char != 'ε':
                mask[deletion_offset + pos] = 1
                
        # Handle mutations 
        for pos, char in enumerate(seq):
            if char != 'ε':
                for char_idx, new_char in enumerate(['A', 'B', 'C']):
                    mask[mutation_offset + pos * vocab_size + char_idx] = 1
                    
    # If current sequence length < max_prev_len, all actions possible
    else:
        # Handle insertions 
        for pos in range(seq_len + 1):
            for char_idx in range(vocab_size):
                mask[insertion_offset + pos * vocab_size + char_idx] = 1
                
        # Handle deletions 
        for pos, char in enumerate(seq):
            if char != 'ε':
                mask[deletion_offset + pos] = 1
                
        # Handle mutations 
        for pos, char in enumerate(seq):
            if char != 'ε':
                for char_idx, new_char in enumerate(['A', 'B', 'C']):
                    mask[mutation_offset + pos * vocab_size + char_idx] = 1

    return torch.Tensor(mask).bool()



def state_to_tensor(state):
    # Unpack state
    timestep, seq = state
    
    # Create one-hot encoding for timestep
    time_tensor = torch.zeros(n_timesteps)
    time_tensor[timestep] = 1
    
    # Create sequence tensor of shape (max_len, vocab_size + 1 for epsilon) filled with zeros
    seq_tensor = torch.zeros(max_len, vocab_size + 1)
    
    # For each position in the sequence
    for i, char in enumerate(seq):
        if char == 'ε':
            # Last index is for epsilon
            seq_tensor[i, -1] = 1
        else:
            # Convert A->0, B->1, C->2
            char_idx = ord(char) - ord('A')
            seq_tensor[i, char_idx] = 1
            
    # Concatenate time and sequence tensors
    return torch.cat([time_tensor, seq_tensor.flatten()])  # (n_timesteps + max_len * (vocab_size+1))




def perform_action(state, action_idx):

    timestep, sequence = state  # timestep is a dummy variable in this function.
    action = actions_list[action_idx]
    action_type = action[0]
    new_sequence = sequence.copy()
    
    if action_type == 'insert':
        _, insert_pos, char = action
        # Shift elements right starting from insert position
        for i in range(len(sequence)-1, insert_pos, -1):
            new_sequence[i] = new_sequence[i-1]
        new_sequence[insert_pos] = char
        
    elif action_type == 'delete':
        _, del_pos = action
        # Shift elements left starting from delete position
        for i in range(del_pos, len(sequence)-1):
            new_sequence[i] = new_sequence[i+1]
        new_sequence[-1] = 'ε'  # Fill last position with epsilon
        
    else: # mutate
        _, mut_pos, char = action
        new_sequence[mut_pos] = char
        
    return [timestep + 1, new_sequence]



def infer_action_id(current_state, next_state):
    """Infer the action index that led from current_state to next_state"""
    # Try each possible action until we find the one that gives us next_state
    for idx in range(max_actions):
        if perform_action(current_state, idx) == next_state:
            return idx
    raise ValueError("No valid action found between states")

