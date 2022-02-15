import sys


def create_heads_dict(classification_mode):
    if classification_mode == "gloss":
        num_classes_dict = {"gloss": 2400}  # number of classes for each head
    elif classification_mode == "hamnosys": # dla annotacji test_hamnosys2.txt
        num_classes_dict = {
            "symmetry_operator": 12,
            "hand_shape_base_form": 13,
            "hand_shape_thumb_position": 4,
            "hand_shape_bending": 6,
            "hand_position_finger_direction": 18,
            "hand_position_palm_orientation": 8,
            "hand_location_frontal_plane_LR": 5,
            "hand_location_frontal_plane_TB": 36,
            "distance": 7,
        }  # number of classes for each head
    elif classification_mode == "toy_hamnosys":  # dla annotacji toy_hamnosys.txt
        num_classes_dict = {
            "hand_shape_base_form": 6,
            "hand_shape_thumb_position": 3,
            "hand_shape_bending": 4,
            "hand_position_finger_direction": 18,
            "hand_position_palm_orientation": 8,
            "hand_location_x": 14,
            "hand_location_y": 5,
        }  # number of classes for each head
    else:
        sys.exit("Wrong classification_mode passed to pipeline")

    return num_classes_dict
