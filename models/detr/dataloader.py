import torchvision
from utils import get_paths


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
            self,
            folder,
            image_processor,
    ):
        annotation_file_path, image_directory_path = get_paths(folder)

        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)

        self.image_processor = image_processor

    def __getitem__(self, idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}
        encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target
