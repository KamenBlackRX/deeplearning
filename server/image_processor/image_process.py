import os
import tarfile

import numpy as np
import six.moves.urllib as urllib
import tensorflow as tf
from matplotlib import pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import ops as util_ops
from object_detection.utils import visualization_utils as visual


class ImageProcess:
    """
    Process images using tensorflow model.
    """
    def __init__(self, model='mask_rcnn_inception_v2_coco_2018_01_28', labels='mscoco_label_map.pbtxt'):
        """
        Constructor of ImageProcessor class, default model is mask RCNN inception v2.

            :param model: model that is going to be used in the processing, default is mask_rcnn_inception_v2_coco_2018_01_28, see https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md for details on the available pre trained models.
            :type model: str
            :param labels: file(.pbtxt) containing the labels of the dataset, usually found in models/research/object_detection/data folder for pre trained tensorflow models.
            :type labels: str
        """
        # root_dir is the repository directory.
        self.root_dir = os.getcwd().split('boidez')[0] + 'boidez/'

        self.pb_path = model + '/frozen_inference_graph.pb'
        self.category_index = None
        try:
            os.chdir(self.root_dir + 'pre_trained_models')
        except IOError:
            print('pre_trained_models folder not found at repository root. Creating...')
            os.mkdir(self.root_dir + 'pre_trained_models')
            os.chdir(self.root_dir + 'pre_trained_models')

        if labels is not None:
            labels_path = os.path.join(self.root_dir + 'models/research/object_detection/data', labels)
            self.load_labels(labels_path)

        downloaded = False
        for file in os.listdir(os.getcwd()):
            if os.path.isdir(file) and file == model:
                downloaded = True
        if downloaded is False:
            self.download_model(model)

    def download_model(self, model):
        """
        Downloads model from tensorflow object detection api.

            :param model: name of the model to be downloaded.
            :type model: str
        """
        model_file = model + ".tar.gz"
        download_base = 'http://download.tensorflow.org/models/object_detection/'

        print('Downloading model ' + model)
        opener = urllib.request.URLopener()
        opener.retrieve(download_base + model_file, model_file)
        tar_file = tarfile.open(model_file)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd())

    def load_labels(self, labels_path, num_classes=90):
        """
        Load labels from .pbtxt file.

            :param labels_path: path to labels file(.pbtxt).
            :type labels_path: str
            :param num_classes: number of classes in the labels file.
            :type num_classes: int
        """
        label_map = label_map_util.load_labelmap(labels_path)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

    def load_model(self):
        """
        Load frozen TensorFlow model from .pb file into memory. The model loaded is the one set in the constructor.

            :return: tensorflow graph from loaded model.
        """
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.pb_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def __run_inference_for_single_image(self, image, **kwargs):
        """
        Run inference for a single image, detecting the features in the image.
        :param image: image to be processed.
        :type image: numpy array
        :param graph: graph to be used as model
        :type graph: tensorflow graph
        :keyword gpu: (bool)run tensorflow-gpu mode
        :return: output dictionary containing the detection.
        """
        config = None
        graph = self.load_model()
        if kwargs is not None and 'gpu' in kwargs.keys() and kwargs['gpu'] is False:
            config = tf.ConfigProto(device_count={'GPU': 0})
        with graph.as_default():
            with tf.Session(config=config) as sess:
                operations = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in operations for output in op.outputs}
                tensor_dict = {}
                for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes',
                            'detection_masks']:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
                if 'detection_masks' in tensor_dict:
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])

                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = util_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)

                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, axis=0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

    def detect(self, images, **kwargs):
        """

        :param images: list of images or single image to be processed.
        :type images: list[numpy array] or numpy array
        :param graph: model to process the images.
        :type graph: tensorflow graph
        :keyword visualize: (bool) activate visualization of output.
        :keyword gpu: (bool) run tensorflow with gpu.
        :return:
        """
        assert type(images) is list or type(images) is np.ndarray, 'Images must be either a list of np.array' \
                                                                   ' or a single np.array.'
        visualize = False
        gpu = True
        if kwargs is not None:
            if 'visualize' in kwargs.keys():
                assert type(kwargs['visualize']) is bool, 'Keyword visualize must be of type bool.'
                visualize = kwargs['visualize']
            if 'gpu' in kwargs.keys():
                assert type(kwargs['gpu']) is bool, 'Keyword gpu must be of type bool.'
                gpu = kwargs['gpu']

        if type(images) is type(np.array):
            images = [images]

        outputs = []
        for image in images:
            outputs.append(self.__run_inference_for_single_image(image, gpu=gpu))
            if visualize is True:
                visual.visualize_boxes_and_labels_on_image_array(image,
                                                                 outputs[-1]['detection_boxes'],
                                                                 outputs[-1]['detection_classes'],
                                                                 outputs[-1]['detection_scores'],
                                                                 self.category_index,
                                                                 instance_masks=outputs[-1]['detection_masks'],
                                                                 use_normalized_coordinates=True,
                                                                 line_thickness=6)
                plt.imshow(image)
                plt.show()
        return outputs
