import time
import numpy as np
import torch
# from onpolicy.runner.shared.base_runner import Runner

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from ..base_runner import Runner # load module from parent folder
from base_runner_shared import Runner
from utils_general import plot_train_metrics

import imageio

from onpolicy.instrument.observer import Observer
from datetime import datetime
import string
import random
import tqdm


def _t2n(x):
    return x.detach().cpu().numpy()


class MAZFishRunner(Runner):
    """Runner class to perform training, evaluation, and data collection"""

    def __init__(self, config):
        super(MAZFishRunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = (
            int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        )

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                ) = self.collect(step)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                data = (
                    obs,
                    rewards,
                    dones,
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                # Insert data into buffer
                self.insert(data)

            # Compute return and update network
            self.compute()
            train_infos = self.train()

            # Post process
            total_num_steps = (
                (episode + 1) * self.episode_length * self.n_rollout_threads
            )

            # Save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save(episode)

            # Log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    "\n Env [{}], Algo {}, Exp {}, updates {}/{} ep, timesteps {}/{}, FPS {}".format(
                        self.env_name,
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    )
                )

                env_infos = {}
                for agent_id in range(self.num_agents):
                    idv_rews = []
                    for info in infos:
                        if "individual_reward" in info[agent_id].keys():
                            idv_rews.append(info[agent_id]["individual_reward"])
                    agent_k = "agent%i/individual_rewards" % agent_id
                    env_infos[agent_k] = idv_rews

                train_infos["average_episode_rewards"] = (
                    np.mean(self.buffer.rewards) * self.episode_length
                )
                print(
                    "average episode rewards is {}".format(
                        train_infos["average_episode_rewards"]
                    )
                )
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # Eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

        plot_train_metrics(self.log_dir)

    def warmup(self):
        # Reset env
        obs = self.envs.reset()

        # Replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        '''
        '''
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic = (
            self.trainer.policy.get_actions(
                np.concatenate(self.buffer.share_obs[step]),
                np.concatenate(self.buffer.obs[step]),
                np.concatenate(self.buffer.rnn_states[step]),
                np.concatenate(self.buffer.rnn_states_critic[step]),
                np.concatenate(self.buffer.masks[step]),
            )
        )
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads)
        )
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_states_critic), self.n_rollout_threads)
        )
        # Rearrange action
        actions_env = actions

        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions_env,
        )

    def insert(self, data):
        (
            obs,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
        )

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
        )

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True,
            )
            eval_actions = np.array(
                np.split(_t2n(eval_action), self.n_eval_rollout_threads)
            )
            eval_rnn_states = np.array(
                np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads)
            )

            eval_actions_env = eval_actions

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(
                eval_actions_env
            )
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones(
                (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
            )
            eval_masks[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), 1), dtype=np.float32
            )

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos["eval_average_episode_rewards"] = np.sum(
            np.array(eval_episode_rewards), axis=0
        )
        eval_average_episode_rewards = np.mean(
            eval_env_infos["eval_average_episode_rewards"]
        )
        print(
            "eval average episode rewards of agent: "
            + str(eval_average_episode_rewards)
        )
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        observer = Observer()  # Initialize the observer

        if hasattr(self.all_args, "run_name"):
            run_name = self.all_args.run_name
        else:
            run_name = "".join(
                random.choices(string.ascii_letters + string.digits, k=8)
            )
            # because args are saved at runner initialization,
            # this is not actually getting saved to all_args.json
            # but is available in the self.all_args object
            self.all_args.run_name = run_name

        """Visualize the env."""
        envs = self.envs
        seeds = [envs.envs[i].env_seed for i in range(self.n_rollout_threads)]

        video_writer = None
        for episode in range(self.all_args.render_episodes):
            episode_seeds = [
                seed + 197 * episode + 1 for seed in seeds
            ]  # 197x and 1 are arbitrary offsets
            obs = envs.reset_with_seeds(episode_seeds)
            if self.all_args.save_vids and episode < self.all_args.num_vids_to_save:
                image = envs.render("rgb_array")[0]  # [0]
                if video_writer is None:
                    os.makedirs(self.output_dir, exist_ok=True)
                    video_writer_path = f"{self.output_dir}/MAFish_behavior_{self.all_args.timestamp}_{run_name}_{episode}.mp4"
                    video_writer = imageio.get_writer(
                        video_writer_path,
                        format="ffmpeg",
                        mode="I",
                        fps=int(1.0 / self.all_args.ifi),
                    )
                    print(f"Opened video writer at {video_writer_path}")
                video_writer.append_data(image)
            # else:
            #     envs.render("human")

            rnn_states = np.zeros(
                (
                    self.n_rollout_threads,
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            masks = np.ones(
                (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
            )

            episode_rewards = []

            for step in tqdm.tqdm(range(self.episode_length)):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(
                    np.concatenate(obs),
                    np.concatenate(rnn_states),
                    np.concatenate(masks),
                    deterministic=True,
                )
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(
                    np.split(_t2n(rnn_states), self.n_rollout_threads)
                )

                actions_env = actions

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                if hasattr(self.trainer.policy.actor.base, "last_attn_mask"):
                    attn_mask = (
                        self.trainer.policy.actor.base.last_attn_mask
                    )  # shape [batch_size * num_agents, obs_dim]
                    try:
                        attn_mask = attn_mask.reshape(
                            self.n_rollout_threads, self.num_agents, -1
                        ).numpy()
                    except Exception as e:
                        print(
                            f"[attn_mask reshape error] Expected shape: ({self.n_rollout_threads * self.num_agents}, -1), got: {attn_mask.shape}"
                        )
                        attn_mask = None
                else:
                    attn_mask = None
                # print("obs.shape", obs.shape)
                # print("rnn_states.shape", rnn_states.shape)
                # print("attn_mask.shape", attn_mask.shape if attn_mask is not None else None)


                # Record the data
                observer.record(
                    episode_index=episode,
                    time_step=step,
                    data={
                        "agent_id": list(range(self.num_agents)),
                        "actions": actions_env,
                        "observations": obs.tolist(),
                        "rnn_states": rnn_states.tolist(),
                        # "rnn_states_critic": rnn_states_critic.tolist(), # TODO
                        "rewards": rewards.tolist(),
                        "infos": infos,
                        # "masks": masks.tolist(),
                        "attn_mask": (
                            attn_mask.tolist() if attn_mask is not None else None
                        ),
                    },
                )

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones(
                    (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
                )
                masks[dones == True] = np.zeros(
                    ((dones == True).sum(), 1), dtype=np.float32
                )

                if self.all_args.save_vids and episode < self.all_args.num_vids_to_save:
                    image = envs.render("rgb_array")[0]  # [0] to match above :)
                    video_writer.append_data(image)

                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    # if elapsed < self.all_args.ifi:
                    #     time.sleep(self.all_args.ifi - elapsed)
                # else:
                #     envs.render("human")

            # Save the Observer log pkl
            outfile = f"{self.output_dir}/MAZFish_neural_{self.all_args.timestamp}_{run_name}_{episode}_raw.pkl"
            env_args = self._get_env_args()
            observer.save(outfile, metadata=(self.all_args, env_args))
            print("Saved", outfile)
            observer = Observer()  # Use a new observer for each episode

            if self.all_args.save_vids and episode < self.all_args.num_vids_to_save:
                video_writer.close()
                print(f"Saved video to {video_writer_path}")
                video_writer = None # Required to reset for the next episode

            print(
                "average episode rewards is: "
                + str(np.mean(np.sum(np.array(episode_rewards), axis=0)))
            )

        return run_name

        # """Visualize the env."""
        # envs = self.envs

        # all_frames = []
        # for episode in range(self.all_args.render_episodes):
        #     print(f"Rendering episode {episode + 1}/{self.all_args.render_episodes}")
        #     obs = envs.reset()
        #     if self.all_args.save_gifs:
        #         image = envs.render("rgb_array")[0] # [0]
        #         all_frames.append(image)
        #     else:
        #         envs.render("human")

        #     rnn_states = np.zeros(
        #         (
        #             self.n_rollout_threads,
        #             self.num_agents,
        #             self.recurrent_N,
        #             self.hidden_size,
        #         ),
        #         dtype=np.float32,
        #     )
        #     masks = np.ones(
        #         (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
        #     )

        #     episode_rewards = []

        #     for step in range(self.episode_length):
        #         calc_start = time.time()

        #         self.trainer.prep_rollout()
        #         action, rnn_states = self.trainer.policy.act(
        #             np.concatenate(obs),
        #             np.concatenate(rnn_states),
        #             np.concatenate(masks),
        #             deterministic=True,
        #         )
        #         # print(f"in forage_runner_shared.py, rnn_shape: {rnn_states.shape}")
        #         actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        #         rnn_states = np.array(
        #             np.split(_t2n(rnn_states), self.n_rollout_threads)
        #         )

        #         actions_env = actions

        #         # Obser reward and next obs
        #         obs, rewards, dones, infos = envs.step(actions_env)
        #         episode_rewards.append(rewards)

        #         # print(f"rnn_states: {rnn_states.shape}")
                
        #         # Record the data
        #         observer.record(
        #             episode_index=episode,
        #             time_step=step,
        #             data={
        #                 "agent_id": list(range(self.num_agents)),
        #                 "actions": actions_env,
        #                 "observations": obs.tolist(),
        #                 "rnn_states": rnn_states.tolist(),
        #                 # "rnn_states_critic": rnn_states_critic.tolist(), # TODO
        #                 "rewards": rewards.tolist(),
        #                 "infos": infos,
        #                 # "masks": masks.tolist(),
        #             },
        #         )

        #         rnn_states[dones == True] = np.zeros(
        #             ((dones == True).sum(), self.recurrent_N, self.hidden_size),
        #             dtype=np.float32,
        #         )
        #         masks = np.ones(
        #             (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
        #         )
        #         masks[dones == True] = np.zeros(
        #             ((dones == True).sum(), 1), dtype=np.float32
        #         )

        #         if self.all_args.save_gifs:
        #             image = envs.render("rgb_array")[0] # [0] to match above :)
        #             all_frames.append(image)
        #             calc_end = time.time()
        #             elapsed = calc_end - calc_start
        #             if elapsed < self.all_args.ifi:
        #                 time.sleep(self.all_args.ifi - elapsed)
        #         else:
        #             envs.render("human")

        #     print(
        #         "average episode rewards is: "
        #         + str(np.mean(np.sum(np.array(episode_rewards), axis=0)))
        #     )

        # if self.all_args.save_gifs:
        #     random_string = "".join(
        #         random.choices(string.ascii_letters + string.digits, k=8)
        #     )
        #     if self.all_args.max_gif_frames is not None and len(all_frames) > self.all_args.max_gif_frames:
        #         trimmed_frames = all_frames[:self.all_args.max_gif_frames]
        #     else:
        #         trimmed_frames = all_frames
        
        #     outfile = f"{self.output_dir}/rollouts_{self.all_args.timestamp}_{self.num_agents}agents_{random_string}.gif"
        #     imageio.mimsave(outfile, trimmed_frames, duration=self.all_args.ifi)
        #     print(
        #         "Saved",
        #         outfile,
        #         "with",
        #         len(all_frames),
        #         "frames of size",
        #         all_frames[0].shape,
        #     )

        #     env_args = self._get_env_args()
        #     observer.add_metadata(self.all_args, env_args)
        #     outfile = f"{self.output_dir}/rollouts_{self.all_args.timestamp}_{self.num_agents}agents_{random_string}.pkl"
        #     observer.save(outfile)
        #     print("Saved", outfile)
