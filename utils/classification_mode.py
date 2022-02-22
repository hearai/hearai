import sys


def create_heads_dict(classification_mode):
    if classification_mode == "gloss":
        num_classes_dict = {"gloss": 2400}  # number of classes for each head
    elif classification_mode == "hamnosys":
        num_classes_dict = {
            "symmetry_operator": 12,
            "hand_shape_base_form": 13,
            "hand_shape_thumb_position": 4,
            "hand_shape_bending": 6,
            "hand_position_finger_direction": 18,
            "hand_position_palm_orientation": 8,
            "hand_location_x": 5,
            "hand_location_y": 36,
        }  # number of classes for each head
    else:
        sys.exit("Wrong classification_mode passed to pipeline")

    return num_classes_dict
