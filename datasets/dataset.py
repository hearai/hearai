

class JsonlDataset(Dataset):
    def __init__(self, data_path):
        """Dummy custom dataset example of not-existing JSONL datafile with hand coordinates and face mesh.
        Args:
            data_path ([type]): Path to Jsonl file
        """        loaded_json = []
        with open(data_path) as f:
            for line in f:
                loaded_json.append(json.loads(line))
        data_pairs = []
        for row in loaded_json[:max_len]:
            hand_coordinate = row["hand_coordinate"]
            face_mesh = row["face_mesh"]
            data_pairs.append([hand_coordinate, face_mesh])

        self.samples = data_pairs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = {}
        sample["hand_coordinate"] = self.samples[idx][0]
        sample["face_mesh"] = self.samples[idx][1]
        return sample
