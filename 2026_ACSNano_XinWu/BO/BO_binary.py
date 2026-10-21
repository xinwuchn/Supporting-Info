"""
@Author   : Xin Wu
@Time     : 2025/3/11 17:22
@File     : BO_binary.py
@Remark   : This code is for Bayesian optimization with Gaussian Process for binary sequences as the surrogate model based on the PHYSBO
"""

import numpy as np
import physbo
import itertools
import os
import matplotlib.pyplot as plt


class BayesianOptimizer:
    """Bayesian Optimizer for binary sequences"""

    def __init__(self, n_dim, seed=None, initial_data_file=None, opt_type="min", score="TS", interval=0, num_rand_basis=5000, NUM_SELECTION=5):
        self.n_dim = n_dim
        self.seed = seed
        self.candidate_space = np.array(list(itertools.product([0, 1], repeat=n_dim)), dtype=int)
        self.policy = None
        self.opt_type = opt_type
        self.initial_data_file = initial_data_file
        self.score = score
        self.interval = interval
        self.num_rand_basis = num_rand_basis
        self.num_search_each_probe = NUM_SELECTION

    def find_index_sequence(self, sequence):
        """Find the index of a sequence in the candidate space"""
        sequence = np.array(sequence, dtype=int)
        matches = np.all(self.candidate_space == sequence, axis=1)
        if np.any(matches):
            return np.where(matches)[0][0]
        raise ValueError("There has Sequence not found in candidate space")

    def initialize_optimizer(self):
        """Initialize optimizer with initial data (Batch 0)"""

        # Check if already initialized, or if initial data file exists
        save_dir = './batch_0'
        if os.path.exists(f'{save_dir}/history_0.npz'):
            raise ValueError(f"Batch 0 already initialized. Run batch 1 to start optimization.")
        if not os.path.exists(self.initial_data_file):
            raise FileNotFoundError(f"Initial data file '{self.initial_data_file}' not found.")

        # Load initial data
        initial_dataset = np.loadtxt(self.initial_data_file, delimiter=',')
        sequences = initial_dataset[:, :self.n_dim]
        values = initial_dataset[:, self.n_dim]

        # Get indices and prepare data, and adjust values based on optimization type (negate for minimization)
        initial_actions = [self.find_index_sequence(seq) for seq in sequences]
        target_values = -values if self.opt_type == "min" else values

        # Create policy
        self.policy = physbo.search.discrete.policy(test_X=self.candidate_space, initial_data=[initial_actions, target_values])
        self.policy.set_seed(self.seed)

        # Save state to batch_0 directory
        self.policy.save(file_history=f'{save_dir}/history_0.npz',
                         file_training=f'{save_dir}/training_0.npz',
                         file_predictor=f'{save_dir}/predictor_0.dump')

        print(f"Initialized with {len(initial_actions)} data points")

    def execute_batch(self, batch_id):
        """Execute a batch of Bayesian optimization"""
        # Create output directory
        batch_dir = f'./batch_{batch_id}'
        output_file = f"{batch_dir}/suggested_batch_{batch_id}.csv"

        if os.path.exists(output_file):
            raise ValueError(f"Batch {batch_id} already executed.\n"
                             f"If you want to re-run this batch, please delete or rename the existing file first.")

        os.makedirs(batch_dir, exist_ok=True)


        # Load previous batch results if this isn't the initialization
        prev_batch = batch_id - 1
        prev_file = f"./batch_{prev_batch}/suggested_batch_{prev_batch}.csv"

        history_file = f'./batch_{prev_batch}/history_{prev_batch}.npz'
        training_file = f'./batch_{prev_batch}/training_{prev_batch}.npz'
        predictor_file = f'./batch_{prev_batch}/predictor_{prev_batch}.dump'

        if not all(os.path.exists(f) for f in [history_file, training_file, predictor_file]):  # check and make sure the last state exists
            raise FileNotFoundError(f"Previous batch {prev_batch} state files not found. Please run batch {prev_batch} first.")

        self.policy = physbo.search.discrete.policy(test_X=self.candidate_space)
        self.policy.load(file_history=history_file,
                         file_training=training_file,
                         file_predictor=predictor_file)

        # If previous results exist, update the model

        if os.path.exists(prev_file):
            try:
                with open(prev_file, 'r') as f:
                    lines = f.readlines()
                if not lines:
                    raise ValueError(f"File '{prev_file}' is empty, please check!")
                for i, line in enumerate(lines):
                    parts = line.strip().split(',')
                    if len(parts) <= self.n_dim or parts[self.n_dim].strip() == '':
                        raise ValueError(f"Line {i + 1} missing evaluation result")

                # Read the CSV with results as string to handle potentially empty cells
                added_dataset = np.loadtxt(prev_file, delimiter=',')
                added_sequences = added_dataset[:, :self.n_dim].astype(int)
                added_values = added_dataset[:, self.n_dim]
                added_actions = [self.find_index_sequence(seq) for seq in added_sequences]
                added_target_values = -added_values if self.opt_type == "min" else added_values

                self.policy.write(added_actions, added_target_values)

                print(f"Updated model with {len(added_actions)} results from batch {prev_batch}")

            except Exception as e:
                raise ValueError(f"Could not process previous batch results, because {str(e)} within batch_{prev_batch} folder!") from e

        # Execute Bayesian search
        actions = self.policy.bayes_search(
            max_num_probes=1,
            num_search_each_probe=self.num_search_each_probe,
            simulator=None,  # return the actions, otherwise return history. 'None' means you should fill your results from cal. or exp.
            score=self.score,
            interval=self.interval,
            num_rand_basis=self.num_rand_basis
        )

        # Get selected sequences
        selected_candidates = self.candidate_space[actions]

        # Save suggested sequences to CSV
        with open(output_file, 'w') as f:
            for seq in selected_candidates:
                row_str = ",".join(map(str, seq)) + ","
                f.write(row_str + "\n")

        # Save current state
        self.policy.save(
            file_history=f'{batch_dir}/history_{batch_id}.npz',
            file_training=f'{batch_dir}/training_{batch_id}.npz',
            file_predictor=f'{batch_dir}/predictor_{batch_id}.dump'
        )

        print(f"Batch {batch_id} complete. Suggested sequences in {output_file}")
        print(f"Note: Add evaluation results after the last comma on each line before running the next batch!")

        return actions, selected_candidates


    def visualization(self, batch_id):
        """Visualize the batch of Bayesian optimization"""
        history_file = f'./batch_{batch_id}/history_{batch_id}.npz'
        training_file = f'./batch_{batch_id}/training_{batch_id}.npz'
        predictor_file = f'./batch_{batch_id}/predictor_{batch_id}.dump'
        if not all(os.path.exists(f) for f in [history_file, training_file, predictor_file]):
            raise FileNotFoundError(f"Batch_{batch_id} state files not found. Please run batch {batch_id} first.")

        policy = physbo.search.discrete.policy(test_X=self.candidate_space)
        policy.load(file_history=f'./batch_{batch_id}/history_{batch_id}.npz',
                    file_training=f'./batch_{batch_id}/training_{batch_id}.npz',
                    file_predictor=f'./batch_{batch_id}/predictor_{batch_id}.dump')

        # extract the info. for plotting
        num_run_nonzero = np.compress(policy.history.terminal_num_run != 0, policy.history.terminal_num_run)
        count_diff = np.diff(np.insert(num_run_nonzero, 0, 0))
        num_batch = np.repeat(np.arange(len(num_run_nonzero)), count_diff)
        num_batch_best = np.unique(num_batch)
        fx = np.compress(policy.history.fx != 0, policy.history.fx)

        best_fx, best_action = policy.history.export_sequence_best_fx()
        best_sequence = self.candidate_space[best_action]
        print(f"The best sequence after {batch_id} batches is: {best_sequence[-1]}")

        if self.opt_type == "min":
            fx *= -1
            best_fx *= -1

        # Create figure
        plt.figure(figsize=(5, 4))

        # Plot best values Line chart
        plt.scatter(num_batch, fx, label='Dataset', color='C0', alpha=0.5, edgecolors='None')
        plt.plot(num_batch_best, best_fx, label='Best value', color='C1', lw=3)
        plt.yscale('log')
        # Add title and labels
        opt_direction = "Minimization" if self.opt_type == "min" else "Maximization"
        plt.xlabel('Batch ID', fontsize=12)
        plt.ylabel(f'Objective Value ({opt_direction})', fontsize=12)

        # Show plot
        plt.xlim(-1, 31)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()

        np.savez("plot_bo.npz", num_batch=num_batch, fx=fx, num_batch_best=num_batch_best, best_fx=best_fx)


def main():
    try:
        batch_to_run = int(input("Enter batch number (0=init, positive=bo, negative=visualize): "))
        print()

        # Create optimizer
        bo = BayesianOptimizer(
            n_dim=13,  # the dimension of your binary sequence
            seed=0,  # random seed of policy for reproductivity
            initial_data_file="./batch_0/Initial_dataset.csv",  # the file path of your initial data
            opt_type="min",  # do you want to max or min the objective value
            score="TS",  # sampling method: "TS", "EI", or "PI"
            interval=0,  # number of learning the hyper parameter: 0 means learning only at the first step
            num_rand_basis=5000,  # number of basis function
            NUM_SELECTION=8  # how many selected candidates for next step
        )

        # Handle visualization request (-n: means results after n batches)
        if batch_to_run < 0:
            bo.visualization(abs(batch_to_run))
            return

        # Handle initialization
        if batch_to_run == 0:
            bo.initialize_optimizer()
            print("Initialized optimizer. Run batch 1 for first suggestions.")
            return

        # Execute batch
        bo.execute_batch(batch_id=batch_to_run)
        print(f"Next: Perform and add results in batch_{batch_to_run}/suggested_batch_{batch_to_run}.csv")
        print(f"Then run batch {batch_to_run + 1}")

    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()