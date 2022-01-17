def create_heads_dict(classification_mode):
    if classification_mode == "gloss":
        num_classes_dict = {"gloss": 2400}  # number of classes for each head
    elif classification_mode == "hamnosys":
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
        num_classes_dict = {"gloss": 2400}  # number of classes for each head

    return num_classes_dict
