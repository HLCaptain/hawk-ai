import time
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from lxml import etree
from beamngpy import Scenario, BeamNGpy, Vehicle
from beamngpy.sensors import Camera
from data_generation_strategy import DataGenerationStrategy

class ImageDataGenerationStrategy(DataGenerationStrategy):
    
    def __init__(self, bng: BeamNGpy, number_of_vehicles_in_traffic: int):
        self.bng = bng
        self.number_of_vehicles_in_traffic = number_of_vehicles_in_traffic
        self.traffic = bng.traffic
        self.cameras: list[Camera] = []
        self.bboxes = []
        self.images = []
        self.vehicle_cameras: dict[Vehicle, Camera] = {}
    
    def setup_scenario(self, scenario: Scenario) -> None:

        self.spawn_random_vehicles(
            bng=self.bng,
            scenario=scenario,
            number_of_vehicles=1,
            models=['etk800']
        )
        
        list(scenario.vehicles.values())[0].switch()
        
        for vehicle in scenario.vehicles.values():
            vehicle.ai.set_mode('traffic')
            vehicle.ai.drive_in_lane(True)
            vehicle.ai.set_aggression(0.5)
            front_camera_pos = (0.0, -4, 0.8)
            front_camera_dir = (0, -1, 0)
            camera = Camera(vehicle.vid + '_front_camera', self.bng, vehicle=vehicle,
                requested_update_time=-1.0, is_using_shared_memory=True,
                pos=front_camera_pos, dir=front_camera_dir, field_of_view_y=90,
                near_far_planes=(0.1, 100), resolution=(1024, 1024),
                is_render_annotations=True, is_render_instance=True, is_render_depth=True
            )
            self.cameras.append(camera)
            self.vehicle_cameras[vehicle] = camera
            
        scenario.update()

        self.traffic.start(list(scenario.vehicles.values()))
        self.traffic.spawn(
            max_amount=self.number_of_vehicles_in_traffic,
            extra_amount=0,
            parked_amount=0
        )
        self.traffic.reset()
    
    def _snap_camera(self, camera: Camera, class_data: dict):
        images = camera.get_full_poll_request()
        self.images.append(images['colour'])
        bounding_boxes = Camera.extract_bounding_boxes(images['annotation'], images['instance'], class_data)
        self.bboxes.append(bounding_boxes)
        
    def _show_most_recent_image(self):
        image_with_boxes = Camera.draw_bounding_boxes(self.bboxes[-1], self.images[-1], width=3)
        plt.clf()
        plt.figure(figsize=(15, 15))
        plt.imshow(np.asarray(image_with_boxes.convert('RGB')))
        plt.show()
    
    def monitor_data(self, iteration_duration: float, iteration: int) -> None:
        annotations = self.bng.camera.get_annotations()                     # Gets a dictionary of RGB colours, indexed by material names.
        class_data = self.bng.camera.get_annotation_classes(annotations)    # Gets a dictionary of material names, indexed by RGB colours (encoded as 32-bit).

        front_camera_pos = (0.0, -4, 0.8)
        front_camera_dir = (0, -1, 0)
        rear_camera_pos = (0.0, 4, 0.8)
        rear_camera_dir = (0, 1, 0)
        right_camera_pos = (-2.0, 0, 0.8)
        right_camera_dir = (-1, 0, 0)
        left_camera_pos = (2.0, 0, 0.8)
        left_camera_dir = (1, 0, 0)
        camera_positions = [front_camera_pos, rear_camera_pos, right_camera_pos, left_camera_pos]
        camera_directions = [front_camera_dir, rear_camera_dir, right_camera_dir, left_camera_dir]
        
        time.sleep(1)
        t0 = time.time()
        while t0 + iteration_duration > time.time():
            for vehicle, camera in self.vehicle_cameras.items():
                print(vehicle)
                print(camera)
                for (camera_pos, camera_dir) in zip(camera_positions, camera_directions):
                    camera.set_position(camera_pos)
                    camera.set_direction(camera_dir)
                    self._snap_camera(camera, class_data)

        # self._show_most_recent_image()

    def clean_scenario(self, scenario: Scenario) -> None:
        for camera in self.cameras:
            camera.remove()
        self.cameras.clear()
        for vehicle in list(scenario.vehicles.values()):
            scenario.remove_vehicle(vehicle)
        self.traffic.reset()
        self.vehicle_cameras.clear()
    
    def finish(self) -> None:
        # Save images and annotations
        image_folder = 'images'
        annotations_folder = 'annotations'
        # Remove folders if they exist
        if os.path.exists(image_folder):
            for filename in os.listdir(image_folder):
                os.remove(os.path.join(image_folder, filename))
        if os.path.exists(annotations_folder):
            for filename in os.listdir(annotations_folder):
                os.remove(os.path.join(annotations_folder, filename))
        os.makedirs(image_folder, exist_ok=True)
        os.makedirs(annotations_folder, exist_ok=True)

        for i, (image_data, bboxes) in enumerate(zip(self.images, self.bboxes)):
            # Save image
            image_filename = f"image_{i:04d}.png"
            plt.imsave(os.path.join(image_folder, image_filename), np.asarray(image_data.convert('RGB')))
            # Create XML annotations
            annotation_xml = Camera.export_bounding_boxes_xml(bboxes, filename=image_filename, size=(1024, 1024, 3))

            # Save XML annotation
            annotation_filename = f"annotation_{i:04d}.xml"
            with open(file=os.path.join(annotations_folder, annotation_filename), mode='w', encoding='utf-8') as file:
                file.write(annotation_xml)
