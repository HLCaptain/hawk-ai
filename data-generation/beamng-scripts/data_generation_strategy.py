import time
import random
from abc import ABC, abstractmethod
from beamngpy import Scenario, BeamNGpy, Vehicle

class DataGenerationStrategy(ABC):
    
    def __init__(self):
        self.vehicle_model_names = [
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
    
    @abstractmethod
    def setup_scenario(self, scenario: Scenario) -> None:
        pass
    
    @abstractmethod
    def clean_scenario(self, scenario: Scenario) -> None:
        pass

    @abstractmethod
    def monitor_data(self, iteration_duration: float, iteration: int) -> None:
        pass
    
    @abstractmethod
    def finish(self) -> None:
        pass
    
    def spawn_random_vehicles(
        self,
        bng: BeamNGpy,
        scenario: Scenario,
        number_of_vehicles: int = 1,
        models: list[str] = []):
        
        if len(models) == 0:
            models = self.vehicle_model_names

        are_roads_loading = True
        while are_roads_loading:
            print('Road data loading...')
            time.sleep(1.0)
            roads = bng.scenario.get_roads()
            are_roads_loading = len(roads) == 0
        print('Road data loaded!')
        
        drivable_road_ids = []
        for r_id, r_inf in roads.items():
            if r_inf['drivability'] != '-1':
                drivable_road_ids.append(r_id)

        drivable_road_ids = random.choices(drivable_road_ids, k=number_of_vehicles)
        
        road_edges = {}
        for road_id in drivable_road_ids:
            road_edges[road_id] = bng.scenario.get_road_edges(road_id)
        
        for r_id, r_edges in road_edges.items():
            vehicle_name = 'vehicle_' + r_id
            vehicle = Vehicle(
                vid=vehicle_name,
                model=random.choice(models),
                color='RED',
                license='HawkAI'
            )

            middle = r_edges[int(len(r_edges) / 2)]['middle']
            scenario.add_vehicle(
                vehicle=vehicle,
                pos=(middle[0], middle[1] + 0.5, middle[2]),
                rot_quat=(0, 0, 0.3826834, 0.9238795)
            )

        scenario.update()
