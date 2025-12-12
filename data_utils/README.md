# build_vqh_dataset.py
This script restructures the Isaac GR00T dataset to facilitate integration with the Elegant Critic. Specifically, it extends the original dataset schema by adding the following features:
* `observation.images.image.vqh`
* `observation.images.wrist_image.vqh`
* `observation.state.vqh`
* `next_action`

# remove_vqh_dataset.py
Used to sanitize the dataset by automatically detecting and removing specific added keys (e.g., next_action and *.vqh features), thereby restoring the dataset to its original, clean state.

# convert_dataset.py
Renamed action to actions: Adjusted the dataset schema to match the specific format required by Pi.

# collect_intervention_libero.py
Designed for tasks lacking corresponding "Elegant" demonstrations. This script facilitates data collection via human intervention during policy inference, capturing supplementary data to bridge coverage gaps.

# extract_episode.py
Extracts specified episodes from an existing dataset to create a new dataset.

# extract_task.py
Extracts specified tasks from an existing dataset to create a new dataset.

# filter_dataset_by_action.py
Filters out idle frames within an episode based on the magnitude of action changes between adjacent timestamps.

# generate_reward_annotations.py
Generates annotations from successful episodes to validate binary rewards.

# merge_dataset.py
Combines multiple datasets into a single unified dataset.

# mirrior_dataset.py
Corrects image orientation by rotating images (originally flipped 180 degrees) back to upright.

# reconstruct_dataset.py
Constructs a dataset with reward labels based on JSON annotation files.

# segment_annotator_viewer.py
A graphical user interface (GUI) for annotating segment-level rewards.

# split_dataset.py
Splits a large dataset into smaller subsets based on specified episode intervals.

# utils.py
Contains utility functions for logging configuration and logic for saving runtime rollouts to the dataset.

# viewer.py
A visualization interface for inspecting annotation results and Elegant Critic outputs.

# plot_trajectory.py
Used to visualize the final states of successful episodes in Task-2 and Task-3, thereby validating the reasonableness of the Elegant Criteria settings.