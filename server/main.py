from threading import Thread

from server.image_processor import image_process
from server.image_processor import masking
from server.utils import vision


def init_process_result():
    processor = image_process.ImageProcess(gpu=False)
    assert processor is not None, "We need test implemented."

    print("Start process")

    images = vision.read_images_from_folder(vision.root_dir + 'images')
    outputs = processor.detect(images, visualize=False)
    masks = masking.mask_centroids(outputs, score_threshold=.5, category_dict=processor.category_index)
    masking.draw(images, masks, draw_centroid=True, label_and_score=True)
    return True


if __name__ == '__main__':
    t1 = Thread(name="test", target=init_process_result)
    t1.start()
    while t1.is_alive():
        print("Processing image", end='\r')
