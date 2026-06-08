import pandas as pd


class Observer:
    def __init__(self):
        # Initialize an empty DataFrame with the desired columns
        self.data = pd.DataFrame(
            columns=[
                "episode_index",
                "time_step",
                "agent_id",
                "actions",
                "observations",
                "rnn_states",
                "rewards",
                "infos",
                "masks",
            ]
        )

    def record(self, episode_index, time_step, data):
        # This method will be called to add data to the DataFrame for all agents
        temp_data = {"episode_index": episode_index, "time_step": time_step, **data}
        self.data = pd.concat([self.data, pd.DataFrame([temp_data])], ignore_index=True)

    def save(self, filename="render_data.pkl"):
        # Method to save the DataFrame to a CSV file
        self.data.to_pickle(filename, index=False)

    def get_data(self):
        # Method to retrieve the current state of the DataFrame
        return self.data
