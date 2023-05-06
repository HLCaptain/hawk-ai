import random
import time

from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from data_generation_strategy import DataGenerationStrategy
from imu_data_generation_strategy import ImuDataGenerationStrategy
from image_data_generation_strategy import ImageDataGenerationStrategy
    
def generate_data(
    bng: BeamNGpy,
    scenario: Scenario,
    strategy: DataGenerationStrategy,
    number_of_iterations: int,
    monitor_data_length: int):

    vehicle_main_name = 'vehicle_main'
    vehicle_main = Vehicle(
        vid=vehicle_main_name,
        model='etk800',
        license='HawkAI',
        color='Red'
    )

    scenario.add_vehicle(vehicle_main, pos=(-717.121, 101, 118.675), rot_quat=(0, 0, 0.3826834, 0.9238795))
    scenario.make(bng)
    bng.load_scenario(scenario)
    scenario.remove_vehicle(vehicle_main)
    bng.start_scenario()

    for iteration in range(number_of_iterations):
        print('Setting up scenario for iteration', iteration)
        strategy.setup_scenario(scenario)
        
        print('Monitoring data for iteration', iteration)
        strategy.monitor_data(monitor_data_length, iteration)
        
        print('Scenario cleanup for iteration', iteration)
        strategy.clean_scenario(scenario)

    print('Ended data generation for iteration', iteration)
    strategy.finish()

def main():
    set_up_simple_logging()

    number_of_iterations = 200
    
    # Image generation
    number_of_vehicles_in_traffic = 12
    image_per_iteration = 20
    
    # IMU generation
    number_of_vehicles = 2
    iteration_duration_in_seconds = 45
    
    random.seed(1337 + time.time_ns())
    beamng = BeamNGpy('localhost', 64256)
    bng = beamng.open(launch=True)
    bng.settings.set_deterministic(100) # Set simulator to 60hz temporal resolution

    scenario = Scenario('west_coast_usa', 'data_generation', description='Generating data iteratively')
    
    image_generation_strategy = ImageDataGenerationStrategy(bng, number_of_vehicles_in_traffic)
    imu_generation_strategy = ImuDataGenerationStrategy(bng, number_of_vehicles)

    generate_data(
        bng=bng,
        scenario=scenario,
        strategy=image_generation_strategy,
        number_of_iterations=number_of_iterations,
        monitor_data_length=image_per_iteration,
    )
    
    bng.close()

if __name__ == '__main__':
    main()