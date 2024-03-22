import time
import os
import numpy as np
import matplotlib.pyplot as plt
import threading
from beamngpy import Scenario, BeamNGpy, Vehicle
from beamngpy.sensors import Camera
from data_generation_strategy import DataGenerationStrategy
from PIL import Image

class ImageDataGenerationStrategy(DataGenerationStrategy):
    """
    A data generation strategy for capturing images and annotations using BeamNGpy and a set of vehicle cameras.

    Args:
        bng (BeamNGpy): The BeamNGpy instance.
        number_of_vehicles_in_traffic (int): The number of vehicles in traffic.
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

        self.image_cache.append(images['colour'])
        bounding_boxes = Camera.extract_bounding_boxes(images['annotation'], images['instance'], class_data)
        self.bbox_cache.append(bounding_boxes)

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
