import random
import time
from datetime import datetime
import pandas as pd

from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import AdvancedIMU

def setup_vehicle(vehicle: Vehicle, aggression: float, mode: str = 'traffic'):
    vehicle.ai.set_mode(mode)
    vehicle.ai.set_aggression(aggression)
    vehicle.ai.drive_in_lane(True)
    # vehicle.ai.set_speed(speed = 36.0, mode = 'limit')
    
def get_random_vehicle(name: str, model_names: list):
    return Vehicle(
            vid=name,
            model=random.choice(model_names),
            license='HawkAI',
            color='Red'
            )
    
def generate_data(
    number_of_iterations: int,
    iteration_duration_in_seconds: int,
    vehicle_model_names: list,
    number_of_vehicles: int,
    data: pd.DataFrame):

    random.seed(1337 + time.time_ns())
    beamng = BeamNGpy('localhost', 64256)
    bng = beamng.open(launch=True)
    bng.settings.set_deterministic(60) # Set simulator to 60hz temporal resolution

    scenario = Scenario('west_coast_usa', 'advanced_IMU_demo', description='Spanning the map with an advanced IMU sensor')
    vehicle_main_name = 'vehicle_main'
    vehicle_main = get_random_vehicle(vehicle_main_name, vehicle_model_names)

    scenario.add_vehicle(vehicle_main, pos=(-717.121, 101, 118.675), rot_quat=(0, 0, 0.3826834, 0.9238795))
    scenario.make(bng)
    bng.load_scenario(scenario)
    scenario.remove_vehicle(vehicle_main)
    
    are_roads_loading = True
    while are_roads_loading:
        print('Road data loading...')
        time.sleep(1.0)
        roads = bng.scenario.get_roads()
        are_roads_loading = len(roads) == 0
    print('Road data loaded!')
    
    bng.start_scenario()
    bng.spawn_traffic(
            max_amount=number_of_vehicles,
            extra_amount=2,
            parked_amount=0
        )

    for iteration in range(number_of_iterations):
        # fetch road data from the game
        print('Iteration:', iteration)
        road_ids = random.choices(tuple(roads), k=number_of_vehicles)
        # road_ids = random.choices(tuple(roads), k = number_of_vehicles - 1)
        road_edges = {}

        for road_id in road_ids:
            road_edges[road_id] = beamng.scenario.get_road_edges(road_id)

        vehicles_in_scenario = {}
        for r_id, r_edges in road_edges.items():
            vehicle_name = 'vehicle_' + str(iteration) + '_' + r_id
            vehicle = get_random_vehicle(vehicle_name, vehicle_model_names)
            middle = r_edges[int(len(r_edges) / 2)]['middle']
            scenario.add_vehicle(
                vehicle=vehicle,
                pos=(middle[0], middle[1] + 0.5, middle[2]),
                rot_quat=(0, 0, 0.3826834, 0.9238795)
                )
            vehicles_in_scenario[r_id] = vehicle

        scenario.update()
        bng.reset_traffic()
        bng.start_traffic(list(scenario.vehicles.values()))

        # vehicle_main_imu_name = 'vehicle_' + str(iteration) + '_main_imu'
        # imu_vehicles = {vehicle_main_imu_name: vehicle_main_name}
        imu_vehicles = {}
        imu_update_time = 0.01
        imus = []

        aggressions = {}
        # aggressions[vehicle_main_name] = random.uniform(0.2, 0.6)
        # vehicle_main.ai.set_aggression(aggressions[vehicle_main_name])

        for vehicle_name, vehicle in scenario.vehicles.items():
            vehicle_imu_name = vehicle_name + '_' + str(iteration) + '_imu'
            aggressions[vehicle_name] = random.uniform(0.2, 1.0)
            setup_vehicle(
                vehicle=vehicle,
                aggression=aggressions[vehicle_name],
            )
            # vehicle.ai.set_aggression(aggressions[vehicle_name])
            imu_vehicles[vehicle_imu_name] = vehicle_name
            imus.append(AdvancedIMU(vehicle_imu_name, bng, vehicle, gfx_update_time = imu_update_time))

        imu_ids = {}
        counter = 0
        for imu in imus:
            imu_ids[imu.name] = iteration * number_of_vehicles + counter
            counter += 1

        print('Driving around, polling the advanced IMU sensor at regular intervals...')

        rows = []

        while len(rows) < number_of_vehicles / imu_update_time * iteration_duration_in_seconds:
            for imu in imus:
                # t1 = time.time()
                imu_data = imu.poll()  # Fetch the latest readings from the sensor.
                # print('Poll time:', time.time() - t1)
                for item in imu_data:
                    rows.append(
                        {
                            'imuId': imu_ids[imu.name],
                            'vehicleAggression': aggressions[imu_vehicles[imu.name]],
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

        data = pd.concat([data, pd.DataFrame(rows)], ignore_index=True)

        print('IMU data recorded during iteration: ', len(rows))
        print('Ended data generation')
        bng.stop_traffic()
        for imu in imus:
            imu.remove()
        for vehicle in list(vehicles_in_scenario.values()):
            scenario.remove_vehicle(vehicle)

    bng.close()
    return data

def main():
    set_up_simple_logging()

    number_of_vehicles = 2
    number_of_iterations = 36
    iteration_duration_in_seconds = 45
    vehicle_model_names = [
        'autobello',
        'midtruck',
        'bastion',
        'legran',
        'moonhawk',
        'burnside',
        'vivace',
        'bolide',
        'scintilla',
        'etk800',
        'etki',
        'etkc',
        'barstow',
        'bluebuck',
        'pickup',
        'fullsize',
        'van',
        'roamer',
        'semi',
        'sbr',
        'covet',
        'hopper',
        'miramar',
        'pessima',
        'midsize',
        'wendover'
        ]
    
    parquet_name = 'imu_data_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.parquet'
    data = pd.DataFrame(columns=[
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

    data = generate_data(
        number_of_iterations=number_of_iterations,
        iteration_duration_in_seconds=iteration_duration_in_seconds,
        vehicle_model_names=vehicle_model_names,
        number_of_vehicles=number_of_vehicles,
        data=data
    )
    
    data.to_parquet(parquet_name)

if __name__ == '__main__':
    main()