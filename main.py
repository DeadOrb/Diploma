import sc2
import math

from typing import Tuple, List
from sc2 import Race, Difficulty
from sc2 import unit
from sc2.constants import *
from sc2.data import Result
from sc2.game_info import GameInfo
from sc2.player import Bot, Computer
from sc2.player import Human
from sc2.constants import MARINE, MARAUDER
from sc2.unit import Unit
from sc2.units import Units
from sc2.position import Point2, Point3
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Size
import numpy as np
import torch
import time
from torch import nn
from torch import optim
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, action_size):
        super(ActorNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.network(x)

    def select_action(self, x):
        return torch.multinomial(self(x), 1).detach().numpy()

class ValueNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ValueNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)

class SergeantAgent:
    def __init__(self, gamma=0.97):
        # TODO
        self.gamma = gamma

        self.batch_size = 8
        self.episode_counter = 0
        self.epoch = 0
        
        # TODO: check number input size
        self.value_network = ValueNetwork(6, 24, 1)
        self.actor_network = ActorNetwork(6, 24, 8)

        # state(x, y, x_a, y_a, x_c, y_c, status) -> 

        self.value_network_optimizer = optim.Adam(self.value_network.parameters())
        self.actor_network_optimizer = optim.Adam(self.actor_network.parameters())

        self.all_rewards = []


    def _returns_advantages(self, rewards, values, next_value):
        returns = np.append(np.zeros_like(rewards), [next_value], axis=0)
        
        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma * returns[t + 1]
            
        returns = returns[:-1]
        advantages = returns - values
        return returns, advantages

    def next_episode(self, state):
        if self.episode_counter == 0:
            self.actions = np.empty((self.batch_size,), dtype=np.int)
            self.rewards, self.values = np.empty((2, self.batch_size), dtype=np.float)
            self.states = np.empty((self.batch_size, 6), dtype=np.float)

        self.states[self.episode_counter] = np.array(state)
        self.values[self.episode_counter] = self.value_network(torch.tensor(state, dtype=torch.float)).detach().numpy()
        policy = self.actor_network(torch.tensor(state, dtype=torch.float))
        self.actions[self.episode_counter] = torch.multinomial(policy, 1).detach().numpy()
        return self.actions[self.episode_counter]

    def save_reward(self, action, reward):
        self.all_rewards.append(reward)
        self.rewards[self.episode_counter] = reward
        self.episode_counter += 1

        reward_file = open("reward.txt", 'a')
        reward_file.write("Epoch " + str(self.epoch) + " " + str(reward) + '\n')
        reward_file.close()

        if self.episode_counter == 8:
            self.episode_counter = 0
            self.epoch += 1

            next_value = 0

            returns, advantages = self._returns_advantages(self.rewards, self.values, next_value)

            self.optimize_model(self.states, self.actions, returns, advantages)

    def optimize_model(self, states, actions, returns, advantages):
        actions = F.one_hot(torch.tensor(actions).to(torch.int64), 8)
        returns = torch.tensor(returns[:, None], dtype=torch.float)
        advantages = torch.tensor(advantages, dtype=torch.float)
        states = torch.tensor(states, dtype=torch.float)

        # MSE for the values
        self.value_network_optimizer.zero_grad()
        values = self.value_network(states)
        loss_value = 1 * F.mse_loss(values, returns)
        loss_value.backward()
        self.value_network_optimizer.step()

        # Actor loss
        self.actor_network_optimizer.zero_grad()
        policies = self.actor_network(states)
        loss_policy = ((actions.float() * policies.log()).sum(-1) * advantages).mean()
        loss_entropy = - (policies * policies.log()).sum(-1).mean()
        loss_actor = - loss_policy - 0.0001 * loss_entropy  
        loss_actor.backward()
        self.actor_network_optimizer.step()
        
        return loss_value, loss_actor




class HBot(sc2.BotAI):
    def __init__(self, marine_agent, marauder_agent, game=0) -> None:
        self.marine_agent = marine_agent
        self.marauder_agent = marauder_agent
        self.game = game
        # radius area for attacking
        self.radius = 15

        # counterclock-wise (one step with a radius of approximately 2\sqrt{2} ~ 3) 
        self.sergeant_actions_list = [[0, 3], [2, 2], [3, 0], [2, -2], [0, -3], [-2, -2], [-3, 0], [-2, 2 ]]
        self.commander_action_list = [[0, 15], [10, 10], [15, 0], [10, -10], [0, -15], [-10, -10], [-15, 0], [-10, 10]]

        # flags
        self.mar_is_exist_group = False
        self.mar_if_exist_center_coordinate = False
        self.mar_is_active = False
        self.mar_in_action = False

        self.is_start = True

        self.mrd_is_exist_group = False
        self.mrd_if_exist_center_coordinate = False
        self.mrd_is_active = False
        self.mrd_in_action = False
        self.commander_is_active = False

        self.start_num_mariners = 0
        self.start_num_marauder = 0
        self.target = [0, 0]
        self.previous_attack_center = [0, 0]
        self.attack_center = [0, 0]

        # counter for make reward 
        self.enemy_units_count = 0

        self.enemy_units_group = []

    async def on_step(self, iteration: int):
        # Since work with asynchronous programming is taking place here,
        # it is impossible to divide everything into classes, so all the code here is divided into parts

        self.iteration = iteration
        # SCRIPTED PART OF BOT
        # command center
        ccs: Units = self.townhalls
        # random command center
        cc: Unit = ccs.random

        if self.is_start:
            self.is_start = False
        elif iteration % 60 == 0:
            self.previous_attack_center = self.attack_center
            self.attack_center = self.select_target()

        # Build more SCVs until 70
        if self.can_afford(UnitTypeId.SCV) and self.supply_workers < 70 and cc.is_idle:
            cc.train(UnitTypeId.SCV)

        # Manage idle scvs, would be taken care by distribute workers aswell
        if self.townhalls:
            for w in self.workers.idle:
                th: Unit = self.townhalls.closest_to(w)
                mfs: Units = self.mineral_field.closer_than(10, th)
                if mfs:
                    mf: Unit = mfs.closest_to(w)
                    w.gather(mf)

        await self.distribute_workers()

        # Make scvs until 22, usually you only need 1:1 mineral:gas ratio for reapers, but if you don't lose any then you will need additional depots (mule income should take care of that)
        # Stop scv production when barracks is complete but we still have a command center (priotize morphing to orbital command)
        if (
            self.can_afford(UnitTypeId.SCV)
            and self.supply_left > 0
            and self.supply_workers < 70
            and (
                self.structures(UnitTypeId.BARRACKS).ready.amount < 1
                and self.townhalls(UnitTypeId.COMMANDCENTER).idle
                or self.townhalls(UnitTypeId.ORBITALCOMMAND).idle
            )
        ):
            for th in self.townhalls.idle:
                th.train(UnitTypeId.SCV)

        # Expand if we can afford (400 minerals) and have less than 2 bases
        if (
            1 <= self.townhalls.amount < 3
            and self.already_pending(UnitTypeId.COMMANDCENTER) == 0
            and self.can_afford(UnitTypeId.COMMANDCENTER)
        ):
            # get_next_expansion returns the position of the next possible expansion location where you can place a command center
            location: Point2 = await self.get_next_expansion()
            if location:
                # Now we "select" (or choose) the nearest worker to that found location
                worker: Unit = self.select_build_worker(location)
                if worker and self.can_afford(UnitTypeId.COMMANDCENTER):
                    # The worker will be commanded to build the command center
                    worker.build(UnitTypeId.COMMANDCENTER, location)


        # Send workers back to mine if they are idle
        for scv in self.workers.idle:
            scv.gather(self.mineral_field.closest_to(cc)) 

        # Saturate gas
        for refinery in self.gas_buildings:
            if refinery.assigned_harvesters < refinery.ideal_harvesters:
                worker: Units = self.workers.closer_than(10, refinery)
                if worker:
                    worker.random.gather(refinery)

        # Build more supply depots
        if (
            self.supply_left < 5
            and self.townhalls
            and self.supply_used >= 14
            and self.can_afford(UnitTypeId.SUPPLYDEPOT)
            and self.already_pending(UnitTypeId.SUPPLYDEPOT) < 1
        ):
            workers: Units = self.workers.gathering
            # If workers were found
            if workers:
                worker: Unit = workers.furthest_to(workers.center)
                location: Point2 = await self.find_placement(UnitTypeId.SUPPLYDEPOT, worker.position, placement_step=3)
                # If a placement location was found
                if location:
                    # Order worker to build exactly on that location
                    worker.build(UnitTypeId.SUPPLYDEPOT, location)

        # Build up to 4 barracks if we can afford them
        # Check if we have a supply depot (tech requirement) before trying to make barracks
        barracks_tech_requirement: float = self.tech_requirement_progress(UnitTypeId.BARRACKS)
        if (
            barracks_tech_requirement == 1
            and self.structures(UnitTypeId.BARRACKS).ready.amount + self.already_pending(UnitTypeId.BARRACKS) < 4
            and self.can_afford(UnitTypeId.BARRACKS)
        ):
            workers: Units = self.workers.gathering
            if (
                workers and self.townhalls
            ):  # need to check if townhalls.amount > 0 because placement is based on townhall location
                worker: Unit = workers.furthest_to(workers.center)
                # I chose placement_step 4 here so there will be gaps between barracks hopefully
                location: Point2 = await self.find_placement(
                    UnitTypeId.BARRACKS, self.townhalls.random.position, placement_step=5
                )
                if location:
                    worker.build(UnitTypeId.BARRACKS, location)

        # Build refineries (on nearby vespene) when at least one barracks is in construction
        if (
            self.structures(UnitTypeId.BARRACKS).ready.amount + self.already_pending(UnitTypeId.BARRACKS) > 0
            and self.already_pending(UnitTypeId.REFINERY) < 1
        ):
            # Loop over all townhalls that are 100% complete
            for th in self.townhalls.ready:
                # Find all vespene geysers that are closer than range 10 to this townhall
                vgs: Units = self.vespene_geyser.closer_than(10, th)
                for vg in vgs:
                    if await self.can_place_single(UnitTypeId.REFINERY, vg.position) and self.can_afford(
                        UnitTypeId.REFINERY
                    ):
                        workers: Units = self.workers.gathering
                        if workers:  # same condition as above
                            worker: Unit = workers.closest_to(vg)
                            # Caution: the target for the refinery has to be the vespene geyser, not its position!
                            worker.build(UnitTypeId.REFINERY, vg)

                            # Dont build more than one each frame
                            break

        # Build refineries
        if self.tech_requirement_progress(UnitTypeId.BARRACKS) == 1:
            if self.structures(UnitTypeId.BARRACKS) and self.gas_buildings.amount < self.townhalls.amount * 2:
                if self.can_afford(UnitTypeId.REFINERY):
                    vgs: Units = self.vespene_geyser.closer_than(20, cc)
                    for vg in vgs:
                        if self.gas_buildings.filter(lambda unit: unit.distance_to(vg) < 1):
                            break

                        worker: Unit = self.select_build_worker(vg.position)
                        if worker is None:
                            break

                        worker.build(UnitTypeId.REFINERY, vg)
                        break

        def points_to_build_addon(sp_position: Point2) -> List[Point2]:
            """ Return all points that need to be checked when trying to build an addon. Returns 4 points. """
            addon_offset: Point2 = Point2((2.5, -0.5))
            addon_position: Point2 = sp_position + addon_offset
            addon_points = [
                (addon_position + Point2((x - 0.5, y - 0.5))).rounded for x in range(0, 2) for y in range(0, 2)
            ]
            return addon_points

        # Build barracks reactor or lift if no room to build techlab
        if self.tech_requirement_progress(UnitTypeId.BARRACKS) == 1:
            for sp in self.structures(UnitTypeId.BARRACKS).ready.idle:
                if not sp.has_add_on and self.can_afford(UnitTypeId.BARRACKSREACTOR) and self.structures(UnitTypeId.BARRACKSREACTOR).amount < 2:
                    addon_points = points_to_build_addon(sp.position)
                    if all(
                        self.in_map_bounds(addon_point)
                        and self.in_placement_grid(addon_point)
                        and self.in_pathing_grid(addon_point)
                        for addon_point in addon_points
                    ):
                        sp.build(UnitTypeId.BARRACKSREACTOR)
                    else:
                        sp(AbilityId.LIFT)

        # Build barracks reactor or lift if no room to build techlab
        if self.tech_requirement_progress(UnitTypeId.BARRACKS) == 1:
            for sp in self.structures(UnitTypeId.BARRACKS).ready.idle:
                if (not sp.has_add_on and self.can_afford(UnitTypeId.BARRACKSTECHLAB) 
                    and self.structures(UnitTypeId.BARRACKSTECHLAB).amount < 2
                   ):
                    addon_points = points_to_build_addon(sp.position)
                    if all(
                        self.in_map_bounds(addon_point)
                        and self.in_placement_grid(addon_point)
                        and self.in_pathing_grid(addon_point)
                        for addon_point in addon_points
                    ):
                        sp.build(UnitTypeId.BARRACKSTECHLAB)
                    else:
                        sp(AbilityId.LIFT)

        def land_positions(sp_position: Point2) -> List[Point2]:
            """ Return all points that need to be checked when trying to land at a location where there is enough space to build an addon. Returns 13 points. """
            land_positions = [(sp_position + Point2((x, y))).rounded for x in range(-1, 2) for y in range(-1, 2)]
            return land_positions + points_to_build_addon(sp_position)

        # Find a position to land for a flying barraks so that it can build an addon
        for sp in self.structures(UnitTypeId.BARRACKSFLYING).idle:
            possible_land_positions_offset = sorted(
                (Point2((x, y)) for x in range(-10, 10) for y in range(-10, 10)),
                key=lambda point: point.x ** 2 + point.y ** 2,
            )
            offset_point: Point2 = Point2((-0.5, -0.5))
            possible_land_positions = (sp.position.rounded + offset_point + p for p in possible_land_positions_offset)
            for target_land_position in possible_land_positions:
                land_and_addon_points: List[Point2] = land_positions(target_land_position)
                if all(
                    self.in_map_bounds(land_pos) and self.in_placement_grid(land_pos) and self.in_pathing_grid(land_pos)
                    for land_pos in land_and_addon_points
                ):
                    sp(AbilityId.LAND, target_land_position)
                    break

        # Make marauder if we can afford them and we have supply remaining
        if self.supply_left > 0:
            # Loop through all idle barracks
            for rax in self.structures(UnitTypeId.BARRACKS).idle:
                if self.can_afford(UnitTypeId.MARAUDER) and rax.has_techlab:
                    rax.train(UnitTypeId.MARAUDER) 

        # Make mariners if we can afford them and we have supply remaining
        if self.supply_left > 0:
            # Loop through all idle barracks
            for rax in self.structures(UnitTypeId.BARRACKS).idle:
                if self.can_afford(UnitTypeId.MARINE) and rax.has_reactor:
                    rax.train(UnitTypeId.MARINE)
                    rax.train(UnitTypeId.MARINE)




        # MARINER SERGEANT
        # making group for simple using orders
        self.MARINERS_GROUP = Units(self.units(MARINE), self) 
        if self.MARINERS_GROUP.amount > 0:
            self.MARINE_CENTER = self.take_center(self.MARINERS_GROUP)

            if ((math.sqrt((self.MARINE_CENTER[0] - self.attack_center[0]) ** 2 + (self.MARINE_CENTER[1] - self.attack_center[1]) ** 2) > self.radius)
                or self.previous_attack_center != self.attack_center):
                self.attack(self.attack_center, self.MARINERS_GROUP)
                self.mar_is_active = False
            elif not self.mar_is_active:
                self.stop(self.MARINERS_GROUP)
                self.mar_is_active = True
            
            if self.mar_is_active and not self.mar_in_action:
                self.ACTUAL_MARINER_GROUP = self.MARINERS_GROUP
                self.mar_distance = math.sqrt((self.MARINE_CENTER[0] - self.attack_center[0]) ** 2 + (self.MARINE_CENTER[1] - self.attack_center[1]) ** 2)
                self.mar_enemy_units_group = self.enemy_units().closer_than(self.radius, self.attack_center)
                if self.mar_enemy_units_group == []:
                    self.MARINE_CENTER = self.take_center(self.ACTUAL_MARINER_GROUP)
                    state = [self.MARINE_CENTER[0], self.MARINE_CENTER[1], self.attack_center[0], self.attack_center[1], self.attack_center[0], self.attack_center[1]]
                else:
                    self.MARINE_CENTER = self.take_center(self.ACTUAL_MARINER_GROUP)
                    enemy_center = self.mar_enemy_units_group.center
                    state = [self.MARINE_CENTER[0], self.MARINE_CENTER[1], self.attack_center[0], self.attack_center[1], enemy_center[0], enemy_center[1]]

                self.mar_action = self.marine_agent.next_episode(state)
                self.MARINE_CENTER = self.take_center(self.ACTUAL_MARINER_GROUP)
                self.attack([self.MARINE_CENTER[0] + self.sergeant_actions_list[self.mar_action][0], self.MARINE_CENTER[1] + self.sergeant_actions_list[self.mar_action][1]], 
                self.ACTUAL_MARINER_GROUP)
                self.mar_in_action = True
            elif self.mar_in_action:
                if self.units.tags_in(self.ACTUAL_MARINER_GROUP.tags).idle:
                    new_enemy_units = self.units.tags_in(self.mar_enemy_units_group.tags)
                    kills = len(self.mar_enemy_units_group) - len(new_enemy_units)
                    if kills < 0:
                        kills = 0
                    self.MARINE_CENTER = self.take_center(self.ACTUAL_MARINER_GROUP)

                    local_kills_file = open("local_kills_mariners.txt", "a")
                    local_kills_file.write("Epoch " + str(self.marine_agent.epoch) + " " + str(kills) + '\n')
                    local_kills_file.close()
                    
                    reward = kills * 10 + math.e ** (self.mar_distance - math.sqrt((self.MARINE_CENTER[0] - self.attack_center[0]) ** 2 + (self.MARINE_CENTER[1] - self.attack_center[1]) ** 2))
                    
                    self.marine_agent.save_reward(self.mar_action, reward)
                    self.mar_in_action = False




        # MARINER SERGEANT
        # making group for simple using orders
        self.MARAUDER_GROUP = Units(self.units(MARAUDER), self)
        if self.MARAUDER_GROUP.amount > 0:
            self.MARAUDER_CENTER = self.take_center(self.MARAUDER_GROUP)

            if ((math.sqrt((self.MARAUDER_CENTER[0] - self.attack_center[0]) ** 2 + (self.MARAUDER_CENTER[1] - self.attack_center[1]) ** 2) > self.radius)
                or self.previous_attack_center != self.attack_center):
                self.attack(self.attack_center, self.MARAUDER_GROUP)
                self.mrd_is_active = False
            elif not self.mrd_is_active:
                self.stop(self.MARAUDER_GROUP)
                self.mrd_is_active = True
            
            if self.mrd_is_active and not self.mrd_in_action:
                self.ACTUAL_MARADUDER_GROUP = self.MARAUDER_GROUP
                self.MARAUDER_NUMBER = self.ACTUAL_MARADUDER_GROUP.amount
                self.mrd_distance = math.sqrt((self.MARAUDER_CENTER[0] - self.attack_center[0]) ** 2 + (self.MARAUDER_CENTER[1] - self.attack_center[1]) ** 2)
                self.mrd_enemy_units_group = self.enemy_units().closer_than(self.radius, self.attack_center)
                if self.mrd_enemy_units_group == []:
                    self.MARAUDER_CENTER = self.take_center(self.ACTUAL_MARADUDER_GROUP)
                    state = [self.MARAUDER_CENTER[0], self.MARAUDER_CENTER[1], self.attack_center[0], self.attack_center[1], self.attack_center[0], self.attack_center[1]]
                else:
                    self.MARAUDER_CENTER = self.take_center(self.ACTUAL_MARADUDER_GROUP)
                    enemy_center = self.mrd_enemy_units_group.center
                    state = [self.MARAUDER_CENTER[0], self.MARAUDER_CENTER[1], self.attack_center[0], self.attack_center[1], enemy_center[0], enemy_center[1]]

                self.mrd_action = self.marauder_agent.next_episode(state)
                self.MARAUDER_CENTER = self.take_center(self.ACTUAL_MARADUDER_GROUP)
                self.attack([self.MARAUDER_CENTER[0] + self.sergeant_actions_list[self.mrd_action][0], self.MARAUDER_CENTER[1] + self.sergeant_actions_list[self.mrd_action][1]], 
                self.ACTUAL_MARADUDER_GROUP)
                self.mrd_in_action = True
            elif self.mrd_in_action:
                if self.units.tags_in(self.ACTUAL_MARADUDER_GROUP.tags).idle:
                    new_enemy_units = self.units.tags_in(self.mrd_enemy_units_group.tags)
                    kills = len(self.mrd_enemy_units_group) - len(new_enemy_units)
                    if kills < 0:
                        kills = 0
                    self.MARAUDER_CENTER = self.take_center(self.ACTUAL_MARADUDER_GROUP)
                    reward = kills * 10 + math.e ** (self.mrd_distance - math.sqrt((self.MARAUDER_CENTER[0] - self.attack_center[0]) ** 2 + (self.MARAUDER_CENTER[1] - self.attack_center[1]) ** 2))
                    
                    local_kills_file = open("local_kills_marauder.txt", "a")
                    local_kills_file.write("Epoch " + str(self.marauder_agent.epoch) + " " + str(kills) + '\n')
                    local_kills_file.close()

                    self.marauder_agent.save_reward(self.mrd_action, reward)
                    self.mrd_in_action = False

        # total_numbers_warrior_units_file = open("total_numbers_warrior_units.txt", "a")
        # total_numbers_warrior_units_file.write("Game " + str(self.game) + str(Units(self.units(MARAUDER), self).amount + Units(self.units(MARINE), self).amount) + '\n')
        # total_numbers_warrior_units_file.close()

            
    def select_target(self) -> Point2:
        # Pick a random enemy structure's position
        targets = self.enemy_structures
        if targets:
            return targets.random.position
        # Pick a random enemy unit's position
        targets = self.enemy_units
        if targets:
            return targets.random.position
        # Pick enemy start location if it has no friendly units nearby
        if (min([unit.distance_to(self.enemy_start_locations[0]) for unit in self.units]) > 5 and (self.MARINERS_GROUP.amount + self.MARAUDER_GROUP.amount > 12) 
            and self.iteration > 700):
            return self.enemy_start_locations[0]
        # Pick a random mineral field on the map
        return self.mineral_field.random.position


    def take_center(self, group):
        CENTER = self.units.tags_in(group.tags).center
        return CENTER

    def stop(self, group):
        for unit in group:
            unit.stop()
        
    def attack(self, target: list, group):
        attack_pose = Point2((target[0], target[1]))

        for unit in group:
            unit.attack(attack_pose) 

    def on_end(self, game_result):
        winners = open("winners.txt", "a")
        if game_result == Result.Victory:
            winners.write(str(1) + '\n')
        else:
            winners.write(str(0) + '\n')
        winners.close()
        print(game_result)

marine_agent = SergeantAgent()
marauder_agent = SergeantAgent()
game = 0
while True:
    sc2.run_game(
        sc2.maps.get("AcropolisLE"),
        [Bot(Race.Terran, HBot(marine_agent, marauder_agent, game)), Computer(Race.Zerg, Difficulty.Hard)],
        realtime=False
    )
    game += 1

    # torch.save(marine_agent.actor_network.state_dict(), "marine_actor")
    # torch.save(marine_agent.value_network.state_dict(), "marine_value")
    # torch.save(marauder_agent.actor_network.state_dict(), "marauder_actor")
    # torch.save(marauder_agent.value_network.state_dict(), "marauder_value")