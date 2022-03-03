import sys


def create_heads_dict(classification_mode):
    if classification_mode == "gloss":
        num_classes_dict = {"gloss": [2400, 1]}  # number of classes for each head
    elif classification_mode == "hamnosys":  
        num_classes_dict = {
            "symmetry_operator": [9, 0],
            "hand_shape_base_form": [12, 1],
            "hand_shape_thumb_position": [4, 0],
            "hand_shape_bending": [6, 0],
            "hand_position_finger_direction": [18, 0],
            "hand_position_palm_orientation": [8, 0],
            "hand_location_x": [5, 0] ,
            "hand_location_y": [37, 0],
        }  # number of classes for each head
    elif classification_mode == "hamnosys-less":
        num_classes_dict = {
            "hand_shape_base_form": [12, 0.5],
            "hand_shape_thumb_position": [4, 0.5],
        }  # number of classes for each head
    else:
        sys.exit("Wrong classification_mode passed to pipeline")

    return num_classes_dict
