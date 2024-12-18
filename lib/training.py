

"""Training"""

set_seed(seed)

model = TBModel(n_hid_units, uniform_backward)
opt = torch.optim.Adam(model.parameters(),  learning_rate)

losses, sampled_states, logZs = [], [], []
minibatch_loss = 0
for episode in tqdm(range(n_episodes), ncols=40):
    state = [0, ['ε'] * max_len]
    P_F, _ = model(state_to_tensor(state))  # starting at t=0
    total_log_P_F, total_log_P_B = 0, 0
    
    # Decide how likely to use replay buffer
    use_replay = np.random.random() < replay_freq   
    if use_replay:
        traj = random.choice(replay_buffer)     
        # traj = replay_buffer[1]
    
    for t in range(n_actionsteps):
        # P_F for current state. 
        mask = calculate_forward_mask_from_state(state[1])  
        P_F = torch.where(mask, P_F, -100)  # Make invalid actions unlikely to be selected. 
        categorical = Categorical(logits=P_F)  # softmax
        if use_replay:
            action_idx = torch.tensor(traj[t][-1])
        else:
            action_idx = categorical.sample()
        total_log_P_F += categorical.log_prob(action_idx)  
        
        # P_B for new_state - uniform if enabled
        new_state = perform_action(state, action_idx)  
        P_F, _ = model(state_to_tensor(new_state))
        if uniform_backward:
            mask = calculate_backward_mask_from_state(new_state[0], new_state[1])
            valid_actions = mask.sum()
            total_log_P_B += -torch.log(valid_actions)  # Uniform probability over valid actions
        else:
            mask = calculate_backward_mask_from_state(new_state[0], new_state[1])
            P_B = torch.where(mask, P_B, -100)  # Make invalid actions unlikely to be selected. 
            total_log_P_B += Categorical(logits=P_B).log_prob(action_idx)   
        
        # print("t:", t)
        # print("state:", state) 
        state = new_state  # Continue iterating.

        if t == n_actionsteps - 1:  # End of trajectory.
            reward = torch.tensor(sequence_reward(state[1])).float()
            # print("t: e", )
            # print("state:", state) 
            # print("reward:", reward)
            # print("--------------------------------")

    # We're done with the trajectory(episode), let's compute its loss. 
    minibatch_loss += trajectory_balance_loss(
        model.logZ,
        total_log_P_F,
        total_log_P_B,
        reward,
    )
    # Take a gradient step, if we are at an update episode.
    sampled_states.append(state)
    if episode % update_freq == 0:
        losses.append(minibatch_loss.item())
        logZs.append(model.logZ.item())
        minibatch_loss.backward()
        opt.step()
        opt.zero_grad()
        minibatch_loss = 0


