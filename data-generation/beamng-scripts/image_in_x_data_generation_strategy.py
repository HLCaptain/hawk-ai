import time
import os
import numpy as np
import matplotlib.pyplot as plt
import threading
import math
from beamngpy import Scenario, BeamNGpy, Vehicle
from beamngpy.sensors import Camera
from beamngpy.types import Float3
from data_generation_strategy import DataGenerationStrategy
from PIL import Image
from beamngpy.logging import BNGValueError

class ImageInXDataGenerationStrategy(DataGenerationStrategy):
    """
    A data generation strategy that captures images from multiple cameras placed on vehicles in a BeamNG.drive scenario.
    The captured images are saved along with their annotations and natureness values.

    Args:
        bng (BeamNGpy): The BeamNGpy instance used to interact with the BeamNG.drive simulator.
        number_of_vehicles_in_traffic (int): The number of vehicles to spawn in traffic.
        image_resolution (tuple[int, int], optional): The resolution of the captured images. Defaults to (1024, 1024).
    """

    def __init__(self, bng: BeamNGpy, number_of_vehicles_in_traffic: int, image_resolution: tuple[int, int] = (1024, 1024)):
        super().__init__()
        self.bng = bng
        self.number_of_vehicles_in_traffic = number_of_vehicles_in_traffic
        self.image_resolution = image_resolution
        self.traffic = bng.traffic
        self.cameras: list[Camera] = []
        self.bbox_cache = []
        self.image_cache = []
        self.images_saved = 0
        self.vehicle_cameras: dict[Vehicle, Camera] = {}
        self.natureness_of_images: list[float] = []
        self.scenario: Scenario

        # All object's positions and types on the map
        self.map_objects: list[tuple[Float3, str]] = []

        # Negative weight: probably in City, positive weight: probably outside of City
        self.object_type_weights = {
            'building': -1,
            'nature': 1,
            'rock': 0.4,
            'car': -0.25,
            'pole': -0.1,
        }

    def setup_scenario(self, scenario: Scenario) -> None:
        self.scenario = scenario
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
            camera = Camera(vehicle.vid + '_camera', self.bng, vehicle=vehicle,
                requested_update_time=-1.0, is_using_shared_memory=True, update_priority=1,
                pos=front_camera_pos, dir=front_camera_dir, field_of_view_y=90,
                near_far_planes=(0.1, 100), resolution=self.image_resolution,
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
        print('Snapping camera')
        images = camera.get_full_poll_request()

        # Add image to cache
        self.image_cache.append(images['colour'])
        bounding_boxes = Camera.extract_bounding_boxes(images['annotation'], images['instance'], class_data)
        self.bbox_cache.append(bounding_boxes)

    def _calculate_places_of_images(self, camera: Camera):
        def print_objects(objects):
            for obj in objects:
                print(f'Object Name: {obj.name} Type: {obj.type} Pos: {obj.pos} Rot: {obj.rot} Scale: {obj.scale} Options: {obj.opts}')
        # Get items from the map around the car in 25m radius
        for vehicle, cam in self.vehicle_cameras.items():
            if cam == camera:
                current_position = vehicle.state['pos']
        total_weight = 0
        # Get all model position and types
        if not self.map_objects:
            try:
                print('Loading static instances from scenario...')
                static_objects = self.scenario.bng.scenario.find_objects_class('TSStatic')
                for obj in static_objects:
                    self.map_objects.append((obj.pos, obj.type))
                # First few objects
                print('Loaded objects! First 5 object:')
                print_objects(static_objects[:5])
                print('Types of annotations:')
                # get object.opts['annotation'] for all objects and create a set of them
                annotations = set([obj.opts['annotation'] for obj in static_objects])
                print(annotations)
                print(f'Total number of objects: {len(static_objects)}')
            except BNGValueError as e:
                print(f'Could not load static instances from scenario: {e}')
        for obj in self.map_objects:
            if math.dist(obj[0], current_position) < 25:
                if obj[1] in self.object_type_weights:
                    total_weight += self.object_type_weights[obj[1]]

        # Inverse tan and normalize to 0-1
        natureness = (math.atan(total_weight) + math.pi / 2)
        self.natureness_of_images.append(natureness)


    def _show_most_recent_image(self):
        image_with_boxes = Camera.draw_bounding_boxes(self.bbox_cache[-1], self.image_cache[-1], width=3)
        plt.clf()
        plt.figure(figsize=(15, 15))
        plt.imshow(np.asarray(image_with_boxes.convert('RGB')))
        plt.show()

    def monitor_data(self, monitor_data_length: int, iteration: int) -> None:
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
        old_number_of_images = len(self.image_cache)
        while monitor_data_length > len(self.image_cache) - old_number_of_images: # Loop until the number of images taken is equal to the number of seconds in the iteration
            for camera in self.vehicle_cameras.values():
                for (camera_pos, camera_dir) in zip(camera_positions, camera_directions):
                    camera.set_position(camera_pos)
                    camera.set_direction(camera_dir)
                    self._snap_camera(camera, class_data)
                    self._calculate_places_of_images(camera)

        # self._show_most_recent_image()
        new_number_of_images = len(self.image_cache)
        print(f'Number of images taken during iteration {iteration}: {new_number_of_images - old_number_of_images}')

    def clean_scenario(self, scenario: Scenario) -> None:
        for camera in self.cameras:
            camera.remove()
        self.cameras.clear()
        for vehicle in list(scenario.vehicles.values()):
            scenario.remove_vehicle(vehicle)
        self.traffic.reset()
        self.vehicle_cameras.clear()

    def finish_iteration(self) -> None:
        # Save images and annotations in a separate thread
        save_thread = threading.Thread(target=self.save_images_and_annotations)
        save_thread.start()

    def save_images_and_annotations(self) -> None:
        for i, (image_data, bboxes) in enumerate(zip(self.image_cache, self.bbox_cache)):
            image_folder = 'data/images'
            annotations_folder = 'data/annotations'
            os.makedirs(image_folder, exist_ok=True)
            os.makedirs(annotations_folder, exist_ok=True)

            # Save image, overwriting previous images
            image_filename = f"image_{self.images_saved:04d}.webp"
            image_filepath = os.path.join(image_folder, image_filename)
            if os.path.exists(image_filepath):
                os.remove(image_filepath)
            image_data = image_data.convert('RGB')
            image_data.save(fp=image_filepath, format='WebP', lossless=False, method=6, quality=80)

            # Create XML annotations
            annotation_xml = Camera.export_bounding_boxes_xml(bboxes, filename=image_filename, size=(*self.image_resolution, 3))

            # Save XML annotation, overwriting previous images
            annotation_filename = f"annotation_{self.images_saved:04d}.xml"
            annotation_filepath = os.path.join(annotations_folder, annotation_filename)
            if os.path.exists(annotation_filepath):
                os.remove(annotation_filepath)
            with open(file=annotation_filepath, mode='w', encoding='utf-8') as file:
                file.write(annotation_xml)

            self.images_saved += 1

            # Print progress
            if i % 25 == 0 and i != 0:
                print(f'Finished saving {self.images_saved}/{len(self.image_cache) + self.images_saved - i - 1} images')

        print(f'Finished saving {self.images_saved}/{self.images_saved} images')
        self.image_cache.clear()
        self.bbox_cache.clear()

    def save_natureness(self) -> None:
        csv_filename = 'data/natureness.csv'
        if os.path.exists(csv_filename):
            os.remove(csv_filename)
        with open(file=csv_filename, mode='w', encoding='utf-8') as file:
            for index, natureness in enumerate(self.natureness_of_images):
                file.write(f'{index};{natureness}\n')
