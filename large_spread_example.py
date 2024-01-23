import numpy as np
from .._mpe_utils.core import World, Agent, Landmark
from .._mpe_utils.scenario import BaseScenario
import random

class Scenario(BaseScenario):
    def make_world(self, groups, cooperative=False, shuffle_obs=False):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = sum(groups)
        num_landmarks = len(groups)
        world.collaborative = True

        self.shuffle_obs = shuffle_obs

        self.cooperative = cooperative
        self.groups = groups
        self.group_indices = [a * [i] for i, a in enumerate(self.groups)]
        self.group_indices = [
            item for sublist in self.group_indices for item in sublist
        ]
        # generate colors:
        self.colors = [np.random.random(3) for _ in groups]

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = "agent_{}".format(i)
            agent.collide = False
            agent.silent = True
            agent.size = 0.15

        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in zip(self.group_indices, world.agents):
            agent.color = self.colors[i]

        # random properties for landmarks
        for landmark, color in zip(world.landmarks, self.colors):
            landmark.color = color

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-3, +3, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                for a in world.agents
            ]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    # the collision penalty is cancelled for agents in the same group
                    if self.group_indices[world.agents.index(a)] == self.group_indices[world.agents.index(agent)]:
                        continue
                    else:
                        rew -= 1
                        collisions += 1
        # origin version                        
        # if agent.collide:
        #     for a in world.agents:
        #         if self.is_collision(a, agent):
        #             rew -= 1
        #             collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0

        i = world.agents.index(agent)
        rew = -np.sqrt(
            np.sum(
                np.square(
                    agent.state.p_pos
                    - world.landmarks[self.group_indices[i]].state.p_pos
                )
            )
        )
        # 1. Define the distance threshold
        judge_distance_threshold = agent.size

        # 2. Check if all agents of the same color satisfy the distance condition
        same_color_agents = [a for idx, a in zip(self.group_indices, world.agents) if idx == self.group_indices[i]]
        all_within_threshold = all(
            np.sqrt(np.sum(np.square(a.state.p_pos - world.landmarks[self.group_indices[i]].state.p_pos))) < judge_distance_threshold
            for a in same_color_agents
        )

        # 3. If all agents of the same color are within the threshold, calculate additional reward based on their mutual distances
        if all_within_threshold:
            temp_rew = 0 
            for a1 in same_color_agents:
                for a2 in same_color_agents:
                    if a1 != a2:
                        dist = np.sqrt(np.sum(np.square(a1.state.p_pos - a2.state.p_pos)))
                        temp_rew += 0.1 / (dist + 0.05)
            rew += temp_rew/len(same_color_agents)

        if self.cooperative:
            return 0
        else:
            return rew

    def global_reward(self, world):
        rew = 0

        for i, a in zip(self.group_indices, world.agents):
            l = world.landmarks[i]
            rew -= np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))

        if self.cooperative:
            return rew
        else:
            return 0
        

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # Find the distance between the agent and the landmark of the same color
        # key = 0
        key = self.group_indices[world.agents.index(agent)]
        customized_pos = world.landmarks[key].state.p_pos - agent.state.p_pos
        customized_distance = np.array([np.sqrt(np.sum(np.square(customized_pos)))])
        # x = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + [customized_pos])
        x = np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + [customized_distance])
        if self.shuffle_obs:
            x = list(x)
            random.Random(self.group_indices[world.agents.index(agent)]).shuffle(x)
            x = np.array(x)
        return x
    

