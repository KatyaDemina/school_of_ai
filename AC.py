import torch
import gym
import numpy as np
import torch.nn.functional as F
import random
from collections import deque


class ActorCritic:
    def __init__(self, env):
        self.env = env
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 1.0
        self.gamma = .99
        self.tau = .125

        # ===================================================================== #
        #                               Actor Model                             #
        # Chain rule: we try to maximize Q value, which depends on policy at    #
        # the state. dQ/da *da/dw is weights gradient. We can set -Q as loss    #
        # and make step  only for actor model net to change its weights         #
        # ===================================================================== #

        self.memory = deque(maxlen=2000)
        self.a_optimizer, self.actor_model = self.create_actor_model()
        self.target_a_optimizer, self.target_actor_model = self.create_actor_model()
        # ===================================================================== #
        #                              Critic Model                             #
        # ===================================================================== #

        self.c_optimizer, self.critic_model = self.create_critic_model()
        self.target_c_optimizer, self.target_critic_model = self.create_critic_model()
        self.loss = torch.nn.MSELoss()


    # ========================================================================= #
    #                              Model Definitions                            #
    # ========================================================================= #

    def create_actor_model(self):

        model = torch.nn.Sequential(
            torch.nn.Linear(self.env.observation_space.shape[0],128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,self.env.action_space.shape[0]),
            torch.nn.ReLU()
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        return optimizer, model


    def create_critic_model(self):

        model = CriticNet(self.env.observation_space.shape[0], self.env.action_space.shape[0])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        return optimizer, model


    # ========================================================================= #
    #                               Model Training                              #
    # ========================================================================= #

    def remember(self, cur_state, action, reward, new_state, done):
        self.memory.append([cur_state, action, reward, new_state, done])

    def _train_actor(self, samples):
        for sample in samples:
            self.actor_model.zero_grad()
            cur_state, action, reward, new_state, _ = sample
            predicted_action = self.actor_model(torch.FloatTensor(cur_state))
            minus_predicted_Q = -self.critic_model(torch.FloatTensor(cur_state), predicted_action)
            minus_predicted_Q.backward()
            self.a_optimizer.step()


    def _train_critic(self, samples):
        for sample in samples:

            self.c_optimizer.zero_grad()
            cur_state, action, reward, new_state, done = sample
            if not done:
                target_action = self.target_actor_model(torch.FloatTensor(new_state))
                future_reward = self.target_critic_model(
                    torch.FloatTensor(new_state),target_action).detach().numpy()[0][0]
                reward += self.gamma * future_reward

            output = self.critic_model(torch.FloatTensor(cur_state.reshape(1,24)), torch.FloatTensor(action.reshape(1,4)))
            loss = self.loss(output, torch.FloatTensor(np.array(reward).reshape(1,1)))
            loss.backward()
            self.c_optimizer.step()


    def train(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return
        samples = random.sample(self.memory, batch_size)
        self._train_critic(samples)
        self._train_actor(samples)

    # ========================================================================= #
    #                         Target Model Updating                             #
    # ========================================================================= #

    def _update_actor_target(self):
        self.target_actor_model = self.actor_model


    def _update_critic_target(self):
        self.target_critic_model.update_weights(self.critic_model)

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()

    # ========================================================================= #
    #                              Model Predictions                            #
    # ========================================================================= #

    def act(self, cur_state):
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return self.actor_model(torch.FloatTensor(cur_state.reshape(1, 24))).detach().numpy()

    def load_weights(self):

        self.actor_model.load_state_dict(torch.load('actor_net.pkl'))
        self.actor_model.eval()



class CriticNet(torch.nn.Module):
    def __init__(self,state_shape, action_shape):
        super().__init__()
        self.state_h1 = torch.nn.Linear(state_shape, 32)
        self.state_h2 = torch.nn.Linear(32, 64)

        self.action_h1 = torch.nn.Linear(action_shape, 64)

        self.merged_h1 = torch.nn.Linear(128, 32)

        self.output = torch.nn.Linear(32, 1)

    def forward(self, state, action):
        s = state
        s = F.relu(self.state_h1(s))
        s = F.relu(self.state_h2(s))

        a = action
        a = F.relu(self.action_h1(a))

        m = torch.cat((s, a), 1)
        m = F.relu(self.merged_h1(m))

        return self.output(m)

    def update_weights(self, net):
        self.state_h1.weight = net.state_h1.weight
        self.state_h2.weight = net.state_h2.weight
        self.action_h1.weight = net.action_h1.weight

        self.merged_h1.weight = net.merged_h1.weight
        self.output.weight = net.output.weight






def main(num_trials=1000, trial_len=500):

    env = gym.make('BipedalWalker-v2')
    actor_critic = ActorCritic(env)

    for i in range(num_trials):
        cur_state = env.reset()
        actor_critic.epsilon = 1/np.log(i+np.e)
        for t in range(trial_len):
            r = 0

            env.render()
            cur_state = cur_state.reshape((1, env.observation_space.shape[0]))
            action = actor_critic.act(cur_state)
            action = action.reshape((1, env.action_space.shape[0]))

            new_state, reward, done, _ = env.step(action.reshape(action.shape[1]))
            r+= actor_critic.gamma ** t * reward
            new_state = new_state.reshape((1, env.observation_space.shape[0]))

            actor_critic.remember(cur_state, action, reward, new_state, done)
            actor_critic.train()
            if done:
                break

            cur_state = new_state
        print('trial: ', i, 'step:', t, 'reward:', r, 'epsilon: ', actor_critic.epsilon)
        actor_critic.update_target()

    torch.save(actor_critic.target_actor_model.state_dict(), 'actor_net.pkl')
    torch.save(actor_critic.target_critic_model.state_dict(), 'critic_net.pkl')



def walk(num_trials, trial_len):
    env = gym.make('BipedalWalker-v2')
    actor_critic = ActorCritic(env)
    actor_critic.load_weights()
    actor_critic.epsilon = 0.0

    for i in range(num_trials):
        cur_state = env.reset()
        for t in range(trial_len):
            r = 0

            env.render()
            cur_state = cur_state.reshape((1, env.observation_space.shape[0]))
            action = actor_critic.act(cur_state)

            new_state, reward, done, _ = env.step(action.reshape(action.shape[1]))
            r += actor_critic.gamma ** t * reward

            if done:
                break

            cur_state = new_state
        print('trial: ', i, 'step:', t, 'reward:', r)




if __name__ == "__main__":
    main()
    # walk(10, 500)
