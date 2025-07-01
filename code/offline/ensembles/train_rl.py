from copy import copy

import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from offline.ensembles.rl_models import *
import sys
sys.path.append("../..")
from simulator.simulator import Simulation


# Define the policy gradient agent
class PolicyGradientAgent:
    def __init__(self, num_models, learning_rate=0.01, gamma=0.99, exploration_prob=0.1, hidden_sizes=[128, 64], exploration_decay=0.95):
        self.gamma = gamma
        self.exploration_prob = exploration_prob
        self.exploration_decay = exploration_decay

        # init networks
        self.route_network = RoutingModel(num_models,  hidden_sizes=hidden_sizes)
        self.thresh_network = ThreshModel(num_models, hidden_sizes=hidden_sizes)

        # init optimizers
        self.route_optimizer = optim.Adam(self.route_network.parameters(), lr=learning_rate)
        self.thresh_optimizer = optim.Adam(self.thresh_network.parameters(), lr=learning_rate)


    def select_routing(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.route_network(state_tensor)

        # Random exploration
        if np.random.uniform(0, 1) < self.exploration_prob:
            action = np.random.choice(len(action_probs[0]))
        else:
            action = torch.multinomial(action_probs, num_samples=1).item()

        return action, action_probs


    def select_threshold(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.thresh_network(state_tensor)

        # Random exploration
        if np.random.uniform(0, 1) < self.exploration_prob:
            action = np.random.choice(len(action_probs[0]))
        else:
            action = torch.multinomial(action_probs, num_samples=1).item()

        return action, action_probs


    def update_policy(self, route_log_probs, thresh_log_probs, rewards):
        # decay exploration rate
        self.exploration_prob *= self.exploration_decay

        # normalize rewards
        max_len = max([len(i) for i in rewards])
        rewards_pad = np.array([np.pad(i, (0, max_len-len(i))) for i in rewards])
        if len(rewards) > 1:
            rewards_norm = (rewards_pad - np.mean(rewards_pad, axis=0))/(np.std(rewards_pad, axis=0) + 1e-8)
        else:
            rewards_norm = rewards_pad

        # get relevant rewards for both NNs
        rewards_route = []
        rewards_thresh = []
        for i, episode_rewards in enumerate(rewards):
            route_episode_rewards_norm = []
            thresh_episode_rewards_norm = []
            for j, _ in enumerate(episode_rewards):
                route_episode_rewards_norm.append(rewards_norm[i,j])
                if 0 < j < len(episode_rewards)-1:
                    thresh_episode_rewards_norm.append(rewards_norm[i,j])
            rewards_route.extend(self.calculate_discounted_rewards(route_episode_rewards_norm))
            if len(thresh_episode_rewards_norm) > 0:
                rewards_thresh.extend(self.calculate_discounted_rewards(thresh_episode_rewards_norm))

        assert len(rewards_route) == len(route_log_probs)
        assert len(rewards_thresh) == len(thresh_log_probs)

        # print()
        # print("REWARDS")
        # print(rewards_route)
        # print(rewards_thresh)

        # update
        self.update_network(self.route_network, self.route_optimizer, route_log_probs, rewards_route)
        if len(rewards_thresh) > 0:
            self.update_network(self.thresh_network, self.thresh_optimizer, thresh_log_probs, rewards_thresh)


    def update_network(self, model, optimizer, log_probs, discounted_rewards):
        # discounted_rewards = self.calculate_discounted_rewards(rewards)
        # print("REWARD:")
        # print(discounted_rewards)
        loss = self.calculate_policy_loss(log_probs, discounted_rewards)

        optimizer.zero_grad()
        loss.backward()

        # gradient clipping (probably not necessary
        # max_gradient_norm = 1.0
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)

        optimizer.step()


    def calculate_discounted_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
        running_reward = 0
        for t in reversed(range(len(rewards))):
            running_reward = self.gamma * running_reward + rewards[t]
            discounted_rewards[t] = running_reward
        # print("rewards:", discounted_rewards)
        return discounted_rewards


    def calculate_policy_loss(self, log_probs, rewards):
        policy_loss = []
        for log_prob, reward in zip(log_probs, rewards):
            policy_loss.append(-log_prob * reward)
        return torch.sum(torch.stack(policy_loss))


# Main training loop
def train_policy_gradients(flop_thresh, batch_size=20, max_episodes=100, learning_rate=0.01, exploration_prob=0.1, hidden_sizes=[128,64], alpha=0.5):
    env = Simulation(pred_dir="../../simulator/nn_preds/hellaswag",
                     profiling_file="../../simulator/profiling/llama/llama_prof_tmp.json",
                     flop_thresh=flop_thresh,
                     alpha = alpha)

    num_models = env.num_models

    agent = PolicyGradientAgent(num_models, learning_rate=learning_rate, exploration_prob=exploration_prob, hidden_sizes=hidden_sizes)

    reward_running_avg = np.zeros(env.num_models)
    total_reward_history = []
    acc_history = []
    flop_history = []
    model_history = []
    thresh_history = []

    rewards = []
    route_log_probs = []
    thresh_log_probs = []

    for episode in range(max_episodes):
        state = env.reset()
        total_reward = 0
        episode_rewards = []

        # first model has no threshold, all samples are routed to it
        routing_action, routing_probs = agent.select_routing(state)
        _, done = env.add_model(routing_action)
        next_state, reward, _, _ = env.step()
        route_log_prob = torch.log(routing_probs)[0, routing_action]
        route_log_probs.append(route_log_prob)
        episode_rewards.append(reward)
        total_reward += reward
        state = next_state

        # steps
        while not done:
            # predict action and apply action in simulation

            # get routing
            routing_action, routing_probs = agent.select_routing(state)
            thresh_input, done = env.add_model(routing_action)
            route_log_prob = torch.log(routing_probs)[0, routing_action]
            route_log_probs.append(route_log_prob)

            # get threshold
            if not done:
                thresh_action, thresh_probs = agent.select_threshold(thresh_input)
                env.add_thresh(thresh_action)
                thresh_log_prob = torch.log(thresh_probs)[0, thresh_action]
                thresh_log_probs.append(thresh_log_prob)

            # get reward and next input for routeNet
            next_state, reward, _, _ = env.step()
            episode_rewards.append(reward)

            total_reward += reward
            state = next_state

        rewards.append(episode_rewards)

        # logging progress
        final_acc, final_flops = env.get_goal(True)
        total_reward_history.append(total_reward)
        acc_history.append(final_acc)
        flop_history.append(final_flops)
        model_history.append(copy(env.models))
        thresh_history.append(copy(env.threshs))

        env.print_ensemble()
        env.print_goal()
        print(f"Episode {episode + 1}/{max_episodes}, Total Reward: {total_reward}")

        # batch complete, update
        if (episode + 1) % batch_size == 0 or (episode + 1) == max_episodes:

            # print()
            # print()
            # print(total_reward_history)
            # print(acc_history)
            # print(flop_history)
            # print()
            # print()

            assert len(rewards) == batch_size

            # update policy ned
            agent.update_policy(route_log_probs, thresh_log_probs, rewards)

            # empty lists for new batch
            rewards.clear()
            thresh_log_probs.clear()
            route_log_probs.clear()



    return total_reward_history, acc_history, flop_history, model_history, thresh_history


def add_to_plot(axs, filename):
    rewards = np.load(filename + "_rewards.npy")
    flop = np.load(filename + "_flop.npy")
    acc = np.load(filename + "_acc.npy")

    axs[0].plot(range(len(rewards)), rewards, marker='o', linestyle='-')
    axs[1].plot(range(len(acc)), acc, marker='o', linestyle='-')
    axs[2].plot(range(len(flop)), flop, marker='o', linestyle='-')

    return axs


if __name__ == "__main__":
    rewards, acc, flop, models, threshs = train_policy_gradients(max_episodes=10,
                                                batch_size=10,
                                                flop_thresh=1,
                                                learning_rate=0.001,
                                                exploration_prob=0.5,
                                                hidden_sizes=[128, 256, 128, 64, 32],
                                                alpha=0.5)

    print("acc =", acc)
    print("flop =", flop)
    print()
    print("models = ", models)
    print()
    print("threshs = ", threshs)

    # plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Plot for the first subplot
    axs[0].plot(range(len(rewards)), rewards, marker='o', linestyle='-')
    axs[0].set_title('Reward')
    axs[0].set_xlabel('Index')
    axs[0].set_ylabel('Reward')

    # Plot for the second subplot
    axs[1].plot(range(len(acc)), acc, marker='o', linestyle='-')
    axs[1].set_title('Accuracy')
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel('Accuracy')

    # Plot for the third subplot
    axs[2].plot(range(len(flop)), flop, marker='o', linestyle='-')
    axs[2].set_title('FLOPs')
    axs[2].set_xlabel('Index')
    axs[2].set_ylabel('Relative to goal')

    # add previous runs to plot
    #axs = add_to_plot(axs, "results/playaround_")

    # Show the plots
    plt.tight_layout()
    plt.show()

    # save results
    filename = "results/playaround10"
    np.save(filename + "_rewards", rewards)
    np.save(filename + "_acc", acc)
    np.save(filename + "_flop", flop)