import os
import cv2
import tensorflow as tf
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

from pre_process import tensor_to_image
import numpy as np


content_layers = ['block5_conv2']

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

style_weight=1e-3
content_weight=1e3
train_times = 10
epochs = 100

num_style_layers = len(style_layers)


def scaleSize(w , h, max_w, max_h):
    if (float(h) / w) <= (float(max_h) / max_w) :
        h = h / float(w) * max_w
        w = max_w
    else:
        w = w / float(h) * max_h
        h = max_h
    return int(w), int(h)

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('lijc,lijd->lcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        inputs = inputs*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        
        # put style and content image into CNN to extract feature maps
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                            outputs[self.num_style_layers:])

        # use gram matrix to calculate correlation between feature maps
        style_outputs = [gram_matrix(style_output)
                            for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                        for style_name, value
                        in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}

class transfer_Process:
    progressChange = pyqtSignal(int)
    def __init__(self, content, style):
        self.content_img = content
        self.style_img = style
        self.extractor = StyleContentModel(style_layers, content_layers)
        self.style_targets = self.extractor(self.style_img)['style']
        self.content_targets = self.extractor(self.content_img)['content']
        self.opt = None

    def style_content_loss(self, outputs):
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        # mean square error for loss in style and content
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-self.style_targets[name])**2)
                            for name in style_outputs.keys()])
        style_loss *= style_weight / num_style_layers

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-self.content_targets[name])**2)
                                for name in content_outputs.keys()])
        content_loss *= content_weight / 2
        loss = style_loss + content_loss
        return loss

    @tf.function()
    def train_step(self, image):
        with tf.GradientTape() as tape:
            outputs = self.extractor(image)
            loss = self.style_content_loss(outputs)
        # calculate gradient of loss function with respect to the result image
        grad = tape.gradient(loss, image)
        # apply gradient descent on result image
        self.opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    def run(self, main_process):
        # use Adam for optimizing
        main_process.received.emit(0)
        self.opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
        image = tf.Variable(self.content_img)
        step = 0
        current_percent = 0
        for n in range(train_times):
            for m in range(epochs):
                if not main_process.generating:
                    break
                step += 1
                current_percent = 100 * step / (train_times * epochs)
                self.train_step(image)
                main_process.received.emit(int(current_percent))
            new_image = tensor_to_image(image)
            main_process.result = new_image
            main_process.ui.newImg.setPixmap(main_process.genPixmap(main_process.ui.newImg.width(),
                                                         main_process.ui.newImg.height(), main_process.result))
            main_process.ui.newImg.setStyleSheet("")
        main_process.ui.horizontalLayoutWidget.hide()
        main_process.generating = False
    