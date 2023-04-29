import random
import time
import csv

from datetime import datetime
from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import AdvancedIMU

def setup_vehicle(vehicle: Vehicle, aggression: float, mode: str = 'span'):
    vehicle.ai.set_mode(mode)
    vehicle.ai.set_aggression(aggression)
    vehicle.ai.drive_in_lane(True)
    vehicle.ai.set_speed(speed = 130.0, mode = 'limit')
    
def get_random_vehicle(name: str, model_names: list):
    return Vehicle(
            vid=name,
            model=random.choice(model_names),
            license='HawkAI',
            color='Red'
            )
    
def run_iteration(
    iteration: str,
    iteration_duration_in_seconds: int,
    vehicle_model_names: list,
    number_of_vehicles: int,
    writer: csv.DictWriter):

    random.seed(1337 + time.time_ns())
    beamng = BeamNGpy('localhost', 64256)
    bng = beamng.open(launch=True)
    bng.settings.set_deterministic(60) # Set simulator to 60hz temporal resolution

    print('Iteration:', iteration)
    scenario = Scenario('west_coast_usa', 'advanced_IMU_demo', description='Spanning the map with an advanced IMU sensor')
    vehicle_main_name = 'vehicle_' + iteration + '_main'
    vehicle_main = get_random_vehicle(vehicle_main_name, vehicle_model_names)

    scenario.add_vehicle(vehicle_main, pos=(-717.121, 101, 118.675), rot_quat=(0, 0, 0.3826834, 0.9238795))
    scenario.make(bng)
    bng.scenario.load(scenario)

    are_roads_loading = True
    while are_roads_loading:
        print('Road data loading...')
        time.sleep(1.0)
        roads = bng.scenario.get_roads()
        are_roads_loading = len(roads) == 0
    print('Road data loaded!')

    # fetch road data from the game
    road_ids = random.choices(tuple(roads), k = number_of_vehicles - 1)
    road_edges = {}

    for road_id in road_ids:
        road_edges[road_id] = beamng.scenario.get_road_edges(road_id)
    
    aggressions = {}
    aggressions[vehicle_main_name] = random.uniform(0.2, 1.0)
    setup_vehicle(
        vehicle = vehicle_main,
        aggression = aggressions[vehicle_main_name]
        )

    vehicles_in_scenario = {}
    print(road_edges)
    for r_id, r_edges in road_edges.items():
        vehicle_name = 'vehicle_' + iteration + '_' + r_id
        vehicle = get_random_vehicle(vehicle_name, vehicle_model_names)
        middle = r_edges[int(len(r_edges) / 2)]['middle']
        scenario.add_vehicle(
            vehicle,
            pos = (middle[0], middle[1] + 0.5, middle[2]),
            rot_quat = (0, 0, 0.3826834, 0.9238795)
            )
        aggressions[vehicle_name] = random.uniform(0.2, 1.0)
        setup_vehicle(
            vehicle = vehicle,
            aggression = aggressions[vehicle_name]
            )
        vehicles_in_scenario[r_id] = vehicle

    scenario.update()
    bng.scenario.start()

    vehicle_main_imu_name = 'vehicle_' + iteration + '_main_imu'
    imu_vehicles = {vehicle_main_imu_name: vehicle_main_name}

    imu_update_time = 0.01
    imus = [AdvancedIMU(vehicle_main_imu_name, bng, vehicle_main, gfx_update_time = imu_update_time)]
    for vehicle_id, vehicle in vehicles_in_scenario.items():
        vehicle_name = 'vehicle_' + iteration + '_' + vehicle_id
        vehicle_imu_name = vehicle_name + '_' + iteration + '_imu'
        imu_vehicles[vehicle_imu_name] = vehicle_name
        imus.append(AdvancedIMU(vehicle_imu_name, bng, vehicle, gfx_update_time = imu_update_time))
        
    print('Driving around, polling the advanced IMU sensor at regular intervals...')
    t0 = time.time()
    while time.time() - t0 < iteration_duration_in_seconds:
        for imu in imus:
            data = imu.poll() # Fetch the latest readings from the sensor.
            for _, item in enumerate(data):
                writer.writerow(
                    {
                        'iterationId': iteration,
                        'imuId': imu.name,
                        'vehicleId': imu_vehicles[imu.name],
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

    print('Ended data generation')
    for imu in imus:
        imu.remove()

    bng.close()

def main():
    set_up_simple_logging()

    number_of_vehicles = 10
    iterations = 4
    iteration_duration_in_seconds = 120
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
    
    csv_name = 'imu_data_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.csv'
    with open(file=csv_name, mode='w', newline='') as csvfile:
        
        fieldnames = [
            'iterationId',
            'imuId',
            'vehicleId',
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
            ]
        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
            delimiter=';'
            )
        writer.writeheader()

        for iteration in range(iterations):
            run_iteration(
                iteration=str(iteration),
                iteration_duration_in_seconds=iteration_duration_in_seconds,
                vehicle_model_names=vehicle_model_names,
                number_of_vehicles=number_of_vehicles,
                writer=writer
                )
            time.sleep(5.0)

if __name__ == '__main__':
    main()