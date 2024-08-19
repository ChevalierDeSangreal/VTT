import numpy as np
from scripts.generate_trajectory import load_prepare_trajectory


def full_state_training_data(
	len_data, ref_length=5, dt=0.05, speed_factor=.6, **kwargs
):
	"""
	Use trajectory generation of Elia to generate random trajectories and then
	position the drone randomly around the start
	Arguments:
		reset_strength: how much the drone diverges from its desired state
	"""
	ref_size = 9
	sample_freq = ref_length * 2
	# TODO: might want to sample less frequently
	drone_states = np.zeros((len_data + 200, 12))
	ref_states = np.zeros((len_data + 200, ref_length, ref_size))

	counter = 0
	while counter < len_data:
		traj = load_prepare_trajectory(
			"/home/cgv841/wzm/FYP/AGAPG/aerial_gym/data/traj_data_1", dt, speed_factor, test=0
		)[:, :ref_size]
		traj_cut = traj[:-(ref_length + 1)]
		# traj_cut = traj
		# select every xth sample as the current drone state
		selected_starts = traj_cut[::sample_freq, :]
		
		nr_states_added = len(selected_starts)
		print(nr_states_added)
		full_drone_state = np.hstack(
			(selected_starts, np.zeros((len(selected_starts), 3)))
		)
		# add drone states
		drone_states[counter:counter + nr_states_added, :] = full_drone_state
		# add ref states
		for i in range(1, ref_length + 1):
			ref_states[counter:counter + nr_states_added,
					   i - 1] = (traj[i::sample_freq])[:nr_states_added]

		counter += nr_states_added

	return drone_states[:len_data], ref_states[:len_data]


if __name__ == "__main__":
	states, ref = full_state_training_data(1000)