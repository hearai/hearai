import sys


def create_heads_dict(classification_mode):
    if classification_mode == "gloss":
        num_classes_dict = {
            "gloss": {
                "num_class": 2400,
                "loss_weight": 1
            }
        }  # number of classes for each head
    elif classification_mode == "hamnosys":  
        num_classes_dict = {
            "symmetry_operator": {
                "num_class": 9,
                "loss_weight": 0,
            },
            "hand_shape_base_form": {
                "num_class": 12,
                "loss_weight": 1,
            },
            "hand_shape_thumb_position": {
                "num_class": 4,
                "loss_weight": 0,
            },
            "hand_shape_bending": {
                "num_class": 6,
                "loss_weight": 0,
            },
            "hand_position_finger_direction": {
                "num_class": 18,
                "loss_weight": 0,
            },
            "hand_position_palm_orientation": {
                "num_class": 8,
                "loss_weight": 0,
            },
            "hand_location_x": {
                "num_class": 5,
                "loss_weight": 0,
            },
            "hand_location_y": {
                "num_class": 37,
                "loss_weight": 0,
            },
        }  # number of classes for each head
    elif classification_mode == "hamnosys-less":
        num_classes_dict = {
            "hand_shape_base_form": {
                "num_class": 12,
                "loss_weight": 0.5,
            },
            "hand_shape_thumb_position": {
                "num_class": 4,
                "loss_weight": 0.5,
            },
        }  # number of classes for each head
    else:
        sys.exit("Wrong classification_mode passed to pipeline")

    return num_classes_dict
