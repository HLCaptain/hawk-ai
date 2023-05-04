import time
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from lxml import etree
from beamngpy import Scenario, BeamNGpy
from beamngpy.sensors import Camera
from data_generation_strategy import DataGenerationStrategy

class ImageDataGenerationStrategy(DataGenerationStrategy):
    
    def __init__(self, bng: BeamNGpy, number_of_vehicles: int):
        self.bng = bng
        self.number_of_vehicles = number_of_vehicles
        self.traffic = bng.traffic
        self.cameras: list[Camera] = []
        self.bboxes = []
        self.images = []
    
    def setup_scenario(self, scenario: Scenario) -> None:

        self.spawn_random_vehicles(
            bng=self.bng,
            scenario=scenario,
            number_of_vehicles=self.number_of_vehicles,
            models=['etk800']
        )
        
        for vehicle in scenario.vehicles.values():
            vehicle.ai.set_mode('traffic')
            front_camera = Camera(vehicle.vid + '_front_camera', self.bng, vehicle=vehicle,
                requested_update_time=-1.0, is_using_shared_memory=False,
                pos=(0.0, -4, 0.8), dir=(0, -1, 0), field_of_view_y=90,
                near_far_planes=(0.1, 100), resolution=(256, 256),
                is_render_annotations=True, is_render_instance=True, is_render_depth=True
            )
            rear_camera = Camera(vehicle.vid + '_rear_camera', self.bng, vehicle=vehicle,
                requested_update_time=-1.0, is_using_shared_memory=False,
                pos=(0.0, 4, 0.8), dir=(0, 1, 0), field_of_view_y=90,
                near_far_planes=(0.1, 100), resolution=(256, 256),
                is_render_annotations=True, is_render_instance=True, is_render_depth=True
            )
            left_camera = Camera(vehicle.vid + '_left_camera', self.bng, vehicle=vehicle,
                requested_update_time=-1.0, is_using_shared_memory=False,
                pos=(2.0, 0, 0.8), dir=(1, 0, 0), field_of_view_y=90,
                near_far_planes=(0.1, 100), resolution=(256, 256),
                is_render_annotations=True, is_render_instance=True, is_render_depth=True
            )
            right_camera = Camera(vehicle.vid + '_right_camera', self.bng, vehicle=vehicle,
                requested_update_time=-1.0, is_using_shared_memory=False,
                pos=(-2.0, 0, 0.8), dir=(-1, 0, 0), field_of_view_y=90,
                near_far_planes=(0.1, 100), resolution=(256, 256),
                is_render_annotations=True, is_render_instance=True, is_render_depth=True
            )
            self.cameras.append(front_camera)
            self.cameras.append(rear_camera)
            self.cameras.append(left_camera)
            self.cameras.append(right_camera)
            
        scenario.update()

        self.traffic.start(list(scenario.vehicles.values()))
        self.traffic.spawn(
            max_amount=self.number_of_vehicles * 4,
            extra_amount=2,
            parked_amount=0
        )
    
    def monitor_data(self, iteration_duration: float, iteration: int) -> None:
        annotations = self.bng.camera.get_annotations()                     # Gets a dictionary of RGB colours, indexed by material names.
        class_data = self.bng.camera.get_annotation_classes(annotations)    # Gets a dictionary of material names, indexed by RGB colours (encoded as 32-bit).
        
        t0 = time.time()
        while t0 + iteration_duration > time.time():
            for camera in self.cameras:
                images = camera.get_full_poll_request()
                print(images)
                self.images.append(np.asarray(images['colour'].convert('RGB')))
                bounding_boxes = Camera.extract_bounding_boxes(images['annotation'], images['instance'], class_data)
                self.bboxes.append(bounding_boxes)
        
        print(self.bboxes)
        print(self.images)
        image_with_boxes = Camera.draw_bounding_boxes(self.bboxes[-1], self.images[-1], width=3)
        plt.clf()
        plt.figure(figsize=(15, 15))
        plt.imshow(np.asarray(image_with_boxes.convert('RGB')))
        plt.show()
    
    def clean_scenario(self, scenario: Scenario) -> None:
        for camera in self.cameras:
            camera.remove()
        self.cameras.clear()
        for vehicle in list(scenario.vehicles.values()):
            scenario.remove_vehicle(vehicle)
        self.traffic.reset()
    
    def finish(self) -> None:
        # Save images and annotations
        image_folder = 'images'
        annotations_folder = 'annotations'
        os.rmdir(image_folder)
        os.rmdir(annotations_folder)
        os.makedirs(image_folder, exist_ok=True)
        os.makedirs(annotations_folder, exist_ok=True)

        for i, (image_data, bboxes) in enumerate(zip(self.images, self.bboxes)):
            # Save image
            image_filename = f"image_{i:04d}.jpg"
            plt.imsave(os.path.join(image_folder, image_filename), image_data)
            # Create XML annotations
            annotation_xml = Camera.export_bounding_boxes_xml(bboxes, filename=image_filename, size=(256, 256, 3))

            # Save XML annotation
            annotation_filename = f"annotation_{i:04d}.xml"
            with open(os.path.join(annotations_folder, annotation_filename), 'wb') as f:
                f.write(etree.tostring(annotation_xml, pretty_print=True))
