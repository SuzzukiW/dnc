# src/environment/sumo_env.py

import traci
import numpy as np

class SumoEnv:
    def __init__(self, config):
        self.config = config
        self.intersection_id = config["intersection_id"]
        self.sumo_binary = config["sumo_binary"]
        self.sumo_config_file = config["sumo_config_file"]

    def start(self):
        if not traci.isLoaded():
            traci.start([self.sumo_binary, "-c", self.sumo_config_file])

    def reset(self):
        # Start the SUMO simulation if it hasn't started already
        if not traci.isLoaded():
            self.start()
        else:
            traci.load([self.sumo_binary, "-c", self.sumo_config_file])
        return self.get_state()

    def get_state(self):
        vehicle_count = traci.lanearea.getLastStepVehicleNumber(self.intersection_id)
        waiting_time = traci.lanearea.getLastStepMeanSpeed(self.intersection_id)
        return np.array([vehicle_count, waiting_time])

    def step(self, action):
        self.perform_action(action)
        traci.simulationStep()
        next_state = self.get_state()
        reward = -next_state[1]  # Example reward: negative of waiting time
        done = traci.simulation.getMinExpectedNumber() <= 0  # Simulation end condition
        return next_state, reward, done

    def perform_action(self, action):
        phases = ["GGrrGG", "rrGrrr", "rGGrrG", "rrrGrr"]
        traci.trafficlight.setRedYellowGreenState(self.intersection_id, phases[action])

    def close(self):
        traci.close()