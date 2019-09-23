import numpy as np


def make_sample_her_transitions(replay_strategy, replay_k, reward_fun):
    """Creates a sample function that can be used for HER experience replay.

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if set to 'none',
            regular DDPG experience replay is used
        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if replay_strategy in ['future', 'final']:
        if replay_k == "inf":
            future_p = 1
        else:
            future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    def _sample_her_transitions(episode_batch, batch_size_in_transitions):
        """episode_batch is {key: array(buffer_size x T x dim_key)}
        """
        T = episode_batch['ep_T']
        rollout_batch_size = episode_batch['a'].shape[0]
        batch_size = batch_size_in_transitions

        # Select which episodes and time steps to use.
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.asarray([np.random.randint(T[episode_idx]) for episode_idx in episode_idxs])
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                       for key in episode_batch.keys() if key not in ['ep_T'] }

        # Select future time indexes proportional with probability future_p. These
        # will be used for HER replay by substituting in future goals.
        her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)
        if replay_strategy == 'future':
            future_offset = np.squeeze(np.asarray([np.random.uniform() * (T[episode_idx] - t_sample) for episode_idx, t_sample in zip(episode_idxs, t_samples)]))
            future_offset = future_offset.astype(int)
            future_t = (t_samples + future_offset)[her_indexes]
        else:
            future_t = np.squeeze([T[episode_idx]-1 for episode_idx in episode_idxs])[her_indexes]

        # Replace goal with achieved goal but only for the previously-selected
        # HER transitions (as defined by her_indexes). For the other transitions,
        # keep the original goal.
        future_ag = episode_batch['ag_next'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag

        # Re-compute reward since we may have substituted the goal.
        transitions['r'] = np.asarray([reward_fun(new_ag=ag_next, new_g=g, t=None, T=T[episode_idx], extract_goal=False)[0]
                                       for ag_next, g, t_sample, episode_idx in zip(transitions['ag_next'], transitions['g'],
                                                                            t_samples, episode_idxs)])
        transitions['r'] = np.squeeze(transitions['r'])

        # print(f'r_pos: {np.sum([r == 0 for r in transitions["r"]])}, r_neg: {np.sum([r == -1 for r in transitions["r"]])}')
        # print(np.sum([offset == 0 for offset in future_offset]))

        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                       for k in transitions.keys()}

        assert(transitions['a'].shape[0] == batch_size_in_transitions)

        return transitions

    return _sample_her_transitions

if __name__ == '__main__':
    sampler = make_sample_her_transitions('future', 4, lambda x: x)
    sample = dict(
        ep_T= [400, 400],
        a=np.random.rand(2,400,10)
    )
    sampler(sample, 64)
