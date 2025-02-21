import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np
from dynamics import vehicle_dynamic


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
            nn.Tanh(),
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.action_var = torch.full((action_dim,), action_std * action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()

    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        rewards=rewards.float()

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss=loss.mean()
            loss.backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())


def main():
    rewdic = []
    solved_reward = 300 
    log_interval = 20  
    max_episodes = 2000 
    max_timesteps = 1500  
    update_timestep = 500  
    action_std = 0.5  
    K_epochs = 80  
    eps_clip = 0.2 
    gamma = 0.99 

    lr = 0.0003
    betas = (0.9, 0.999)

    random_seed = None
    state_dim = 16
    action_dim =2 

    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
    print(lr, betas)

    # logging variables
    running_reward = 0
    time_step = 0

    max_reward=0
    for i_episode in range(1, max_episodes + 1):
        x = 0
        y = 0
        vx = 3
        vy = 0
        phi =0
        phidot =0
        xb = -30
        yb = 0
        vbx = 0
        vby = 0
        phib=0
        phibdot=0
        reward=0
        xb1=50
        yb1=0
        vbx1=0
        vby1=0
        state=[x,y,vx,vy,phi,phidot,xb,yb,vbx,vby,xb1,yb1,vbx1,vby1,phib,phibdot]
        state=np.array(state)
        print(f"Current episode {i_episode}")

        for t in range(max_timesteps):
            xb = xb - 1
            time_step += 1
            action = ppo.select_action(state, memory)
            vx=vx+abs(action[0])
            action[1] = action[1] + np.random.normal(0, 0.5, 1) 
            next_state=vehicle_dynamic(y,vy,phi,phidot,vx,action[1])
            phi = phi + next_state[2]
            vy=next_state[0]+next_state[1]
            phidot = next_state[2]
            phi = phi + phidot
            x=x+vx*1
            y=y+vy*1
            state = [x, y, vx, vy, phi, phidot, xb, yb, vbx, vby,xb1,yb1,vbx1,vby1, phib, phibdot]
            state = np.array(state)
            xd=x-xb
            yd=y-yb

            re1=100.0-50*abs(y)

            if abs(y) <= 4:
                re3 = 500
            elif abs(y) > 5:
                re3 = -8000
            else:
                re3 = 10

            if abs(y+vy)<abs(y) and x>=75:
                re4 = 300
            elif abs(y+vy)>abs(y) and x>=75:
                re4 = -300
            else:
                re4=0

            if (x - xb1) * (x - xb1) / 4 + (y - yb1) * (y - yb1) < 9:
                re2=-10000.0
            else:
                re2 = 10

            tempreward =re1+re2+re3 +re4
            reward= reward+tempreward
            done=0
            if x>200 or abs(y)>100:
                done=1

            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            flag=1
            if time_step % update_timestep == 0  and flag:
                print("update")
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
            running_reward += reward
            if done:
                break

        print(f"reward: {reward}")
        rewdic.append(reward)

        if reward > 10:
            print("Problem Solved!")
            if reward > max_reward:
                torch.save(ppo.policy.state_dict(), r'./straightobs/PPO_continuous_solved_{}.pth')
                max_reward=reward

if __name__ == '__main__':
    main()

