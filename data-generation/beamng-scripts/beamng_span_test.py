import random
import time
import csv

from beamngpy import BeamNGpy, Scenario, Vehicle, set_up_simple_logging
from beamngpy.sensors import AdvancedIMU
from datetime import datetime

def setup_vehicle(vehicle: Vehicle):
    vehicle.ai.set_mode('span')
    vehicle.ai.set_aggression(0.2)
    vehicle.ai.drive_in_lane(True)
    
def get_random_vehicle(name: str, model_names: list):
    return Vehicle(
            vid=name,
            model=random.choice(model_names),
            license='HawkAI',
            color='Red'
        )

def main():
    set_up_simple_logging()

    number_of_vehicles = 10
    iterations = 1
    iteration_duration_in_seconds = 30
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
    with open(csv_name, 'w', newline='') as csvfile:
        
        fieldnames = ['vehicleId', 'time', 'x', 'y', 'z', 'mass', 'accRaw', 'accSmooth']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
    
        for iteration in range(1, iterations):
            
            random.seed(1337 + time.time_ns())
            beamng = BeamNGpy('localhost', 64256)
            bng = beamng.open(launch=True)
            bng.settings.set_deterministic(60) # Set simulator to 60hz temporal resolution
            
            print('Iteration:', iteration)
            scenario = Scenario('west_coast_usa', 'advanced_IMU_demo', description='Spanning the map with an advanced IMU sensor')
            vehicle_main = get_random_vehicle('vehicle_main', vehicle_model_names)
            
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
            road_ids = random.choices(tuple(roads), k=number_of_vehicles-1)
            road_edges = {}

            for id in road_ids:
                road_edges[id] = beamng.scenario.get_road_edges(id)

            vehicles_in_scenario = {}
            print(road_edges)  
            for r_id in road_edges.keys():
                r_edges = road_edges[r_id]
                vehicle_name = 'vehicle_' + r_id
                vehicle = get_random_vehicle(vehicle_name, vehicle_model_names)
                middle = r_edges[int(len(r_edges)/2)]['middle']
                scenario.add_vehicle(
                    vehicle,
                    pos=(
                        middle[0],
                        middle[1] + 0.5,
                        middle[2]
                    ),
                    rot_quat=(0, 0, 0.3826834, 0.9238795)
                )
                setup_vehicle(vehicle)
                vehicles_in_scenario[r_id] = vehicle

            setup_vehicle(vehicle_main)
            scenario.update()
            bng.scenario.start()

            imus = [AdvancedIMU('vehicle_main_imu', bng, vehicle_main, gfx_update_time=0.01)]
            for id in vehicles_in_scenario:
                vehicle = vehicles_in_scenario[id]
                imus.append(AdvancedIMU('vehicle_' + id + '_imu', bng, vehicle, gfx_update_time=0.01))        

            print('Driving around, polling the advanced IMU sensor at regular intervals...')
            t0 = time.time()
            while time.time() - t0 < iteration_duration_in_seconds:
                for imu in imus:
                    data = imu.poll() # Fetch the latest readings from the sensor.
                    for i in range(0, len(data)):
                        # print(data[i])
                        writer.writerow(
                            {
                                'vehicleId': imu.name,
                                'time': data[i]['time'],
                                'x': data[i]['pos'][0],
                                'y': data[i]['pos'][1],
                                'z': data[i]['pos'][2],
                                'mass': data[i]['mass'],
                                'accRaw': data[i]['accRaw'],
                                'accSmooth': data[i]['accSmooth']
                            }
                        )

            print('Ended data generation')
            for imu in imus:
                imu.remove()

            bng.close()
            time.sleep(5.0)

if __name__ == '__main__':
    main()