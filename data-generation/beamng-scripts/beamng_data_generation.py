import random
import time

from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from data_generation_strategy import DataGenerationStrategy
from imu_data_generation_strategy import ImuDataGenerationStrategy
from image_data_generation_strategy import ImageDataGenerationStrategy
from image_in_x_data_generation_strategy import ImageInXDataGenerationStrategy
from typing import Callable

def generate_data(
    beamng: BeamNGpy,
    get_scenario: Callable[[], Scenario],
    strategy: DataGenerationStrategy,
    number_of_iterations: int,
    number_of_simulations: int,
    monitor_data_length: int):
    """
    Generates data using the BeamNGpy simulator.

    Args:
        beamng (BeamNGpy): The BeamNGpy instance used to connect to the simulator.
        get_scenario (Callable[[], Scenario]): A function that returns a new Scenario object for each simulation.
        strategy (DataGenerationStrategy): The data generation strategy to use.
        number_of_iterations (int): The number of iterations to perform for each simulation.
        number_of_simulations (int): The number of simulations to run.
        monitor_data_length (int): The length of data to monitor for each iteration.

    Returns:
        None
    """

    vehicle_main_name = 'vehicle_main'
    vehicle_main = Vehicle(
        vid=vehicle_main_name,
        model='etk800',
        license='HawkAI',
        color='Red'
    )

    for simulation in range(number_of_simulations):
        print('Starting simulation:', simulation)
        bng = beamng.open(launch=True)
        bng.settings.set_deterministic(100) # Set simulator to 60hz temporal resolution
        scenario = get_scenario()
        scenario.add_vehicle(vehicle_main, pos=(-717.121, 101, 118.675), rot_quat=(0, 0, 0.3826834, 0.9238795))
        scenario.make(bng)
        bng.load_scenario(scenario)
        scenario.remove_vehicle(vehicle_main)
        bng.start_scenario()

        for iteration in range(number_of_iterations):
            print('Setting up scenario for iteration', number_of_iterations * simulation + iteration)
            strategy.setup_scenario(scenario)

            print('Monitoring data for iteration', number_of_iterations * simulation)
            strategy.monitor_data(monitor_data_length, number_of_iterations * simulation + iteration)

            print('Scenario cleanup for iteration', number_of_iterations * simulation + iteration)
            strategy.clean_scenario(scenario)

            print('Ended data generation for iteration', number_of_iterations * simulation + iteration)
            strategy.finish_iteration()

        print('Closing simulation:', simulation)
        bng.close()
        time.sleep(5)

def main():
    set_up_simple_logging()

    number_of_simulations = 2
    number_of_iterations = 20

    # Image generation
    number_of_vehicles_in_traffic = 6
    image_per_iteration = 20
    # 4 simulations * 50 iterations * 20 images = 4000 images generated overall

    # IMU generation
    number_of_vehicles = 2
    iteration_duration_in_seconds = 60

    get_scenario = lambda: Scenario('west_coast_usa', 'data_generation', description='Generating data iteratively')

    random.seed(1337 + time.time_ns())
    beamng = BeamNGpy('localhost', 64256)

    image_generation_strategy = ImageDataGenerationStrategy(beamng, number_of_vehicles_in_traffic)
    imu_generation_strategy = ImuDataGenerationStrategy(beamng, number_of_vehicles)

    generate_data(
        beamng=beamng,
        get_scenario=get_scenario,
        strategy=image_generation_strategy,
        number_of_iterations=number_of_iterations,
        number_of_simulations=number_of_simulations,
        monitor_data_length=iteration_duration_in_seconds,
    )

if __name__ == '__main__':
    main()