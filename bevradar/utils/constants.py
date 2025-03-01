LIDAR_NAME = 'LIDAR_TOP'

RADAR_NAMES = ('RADAR_FRONT_LEFT', 'RADAR_FRONT', 'RADAR_FRONT_RIGHT',
               'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT')

CAMERA_NAMES = ('CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT')

NAME_TO_CLASS = {
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}

CLASSES = (
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
)

# https://github.com/nutonomy/nuscenes-devkit/blob/4df2701feb3436ae49edaf70128488865a3f6ff9/python-sdk/nuscenes/utils/color_map.py
COLOR_PALETTE = {
    "car": (255, 158, 0),                       # Orange.
    "truck": (255, 99, 71),                     # Tomato.
    "trailer": (255, 140, 0),                   # Dark Orange.
    "bus": (255, 69, 0),                        # Red Orange.
    "construction_vehicle": (233, 150, 70),     # Dark Salmon.
    "bicycle": (220, 20, 60),                   # Crimson.
    "motorcycle": (255, 61, 99),                # Red.
    "pedestrian": (0, 0, 230),                  # Blue.
    "traffic_cone": (47, 79, 79),               # Dark Slate Gray.
    "barrier": (112, 128, 144),                 # Slate Gray.
}

SINGLE_NAME_TO_CLASS = {
    "vehicle.bicycle": "vehicle",
    "vehicle.bus.bendy": "vehicle",
    "vehicle.bus.rigid": "vehicle",
    "vehicle.car": "vehicle",
    "vehicle.construction": "vehicle",
    "vehicle.motorcycle": "vehicle",
    "vehicle.trailer": "vehicle",
    "vehicle.truck": "vehicle",
}

SINGLE_CLASSES = ("vehicle",)

SINGLE_COLOR_PALETTE = {
    "vehicle": (118, 185, 0),                   # GPU company Green.
}

MAP_CLASSES = (
    "drivable_area",
    "ped_crossing",
    "walkway",
    "stop_line",
    "carpark_area",
    "divider"
)

MAP_COLOR_PALETTE = {
    "drivable_area": (166, 206, 227),
    "road_segment": (31, 120, 180),
    "road_block": (178, 223, 138),
    "lane": (51, 160, 44),
    "ped_crossing": (251, 154, 153),
    "walkway": (227, 26, 28),
    "stop_line": (253, 191, 111),
    "carpark_area": (255, 127, 0),
    "road_divider": (202, 178, 214),
    "lane_divider": (106, 61, 154),
    "divider": (106, 61, 154),
}