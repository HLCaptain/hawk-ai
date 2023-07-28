from datetime import datetime
import random
import time
import pandas as pd
import os

from data_generation_strategy import DataGenerationStrategy
from beamngpy import Scenario, Vehicle, BeamNGpy
from beamngpy.sensors import AdvancedIMU

class ImuDataGenerationStrategy(DataGenerationStrategy):

    def __init__(self, bng: BeamNGpy, number_of_vehicles: int):
        super().__init__()
        self.bng = bng
        self.traffic = bng.traffic
        self.number_of_vehicles = number_of_vehicles
        self.imus: list[AdvancedIMU] = []
        self.aggressions = {}
        self.imu_ids = {}
        self.imu_update_time = 0.01
        self.data = self._get_default_data_frame()
        os.makedirs('data/imu', exist_ok=True)
        self.data_file_name = 'data/imu/imu_data_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.parquet'

    def _get_default_data_frame(self) -> pd.DataFrame:
        return pd.DataFrame(columns=[
            'imuId',
            'vehicleAggression',
            'time',
            'pos',
            'dirX',
            'dirY',
            'dirZ',
            'angVel',
            'angAccel',
            'mass',
            'accRaw',
            'accSmooth'
            ])

    def setup_vehicle(self, vehicle: Vehicle, aggression: float, mode: str = 'traffic'):
        vehicle.ai.set_mode(mode)
        vehicle.ai.set_aggression(aggression)
        vehicle.ai.drive_in_lane(True)
        # vehicle.ai.set_speed(speed = 36.0, mode = 'limit')

    def setup_scenario(self, scenario: Scenario) -> None:
        self.spawn_random_vehicles(
            bng=self.bng,
            scenario=scenario,
            number_of_vehicles=self.number_of_vehicles
        )

        for vehicle in scenario.vehicles.values():
            vehicle_imu_name = vehicle.vid + '_imu'
            imu = AdvancedIMU(
                name=vehicle_imu_name,
                bng=self.bng,
                vehicle=vehicle,
                physics_update_time=self.imu_update_time,
                # is_send_immediately=True
            )
            self.imus.append(imu)
            self.aggressions[vehicle.vid] = random.uniform(0.2, 1.0)
            self.setup_vehicle(
                vehicle=vehicle,
                aggression=self.aggressions[vehicle.vid]
            )

        # Spawning traffic
        self.traffic.start(list(scenario.vehicles.values()))
        self.traffic.spawn(
            max_amount=self.number_of_vehicles,
            extra_amount=2,
            parked_amount=0
        )

    def clean_scenario(self, scenario: Scenario) -> None:
        # for imu in self.imus:
        #     imu.remove()
        self.imus.clear()
        for vehicle in list(scenario.vehicles.values()):
            scenario.remove_vehicle(vehicle)
        self.traffic.reset()

    def monitor_data(self, monitor_data_length: int, iteration: int) -> None:
        print('Driving around, polling the advanced IMU sensor at regular intervals...')
        counter = 0
        imu_ids = {}
        for imu in self.imus:
            imu_ids[imu.name] = iteration * self.number_of_vehicles + counter
        rows = []
        while len(rows) < self.number_of_vehicles / self.imu_update_time * monitor_data_length:
            for imu in self.imus:
                imu_data = imu.poll()  # Fetch the latest readings from the sensor.
                for item in imu_data:
                    rows.append(
                        {
                            'imuId': imu_ids[imu.name],
                            'vehicleAggression': self.aggressions[imu.vehicle.vid],
                            'time': item['time'],
                            'pos': item['pos'],
                            'dirX': item['dirX'],
                            'dirY': item['dirY'],
                            'dirZ': item['dirZ'],
                            'angVel': item['angVel'],
                            'angAccel': item['angAccel'],
                            'mass': item['mass'],
                            'accRaw': item['accRaw'],
                            'accSmooth': item['accSmooth'],
                        }
                    )

        self.data = pd.concat([self.data, pd.DataFrame(rows)], ignore_index=True)
        print('IMU data recorded during iteration: ', len(rows))

    def finish_iteration(self) -> None:
        # Read in the data if exists
        if os.path.exists(self.data_file_name):
            current_data = pd.read_parquet(self.data_file_name)
        else:
            current_data = self._get_default_data_frame()

        # Concat new data
        current_data = pd.concat([current_data, self.data], ignore_index=True)
        current_data.to_parquet(self.data_file_name)
        # Reset data
        self.data = self._get_default_data_frame()
