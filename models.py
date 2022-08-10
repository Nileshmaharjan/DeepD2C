import numpy as np
import tensorflow as tf

import models
import utils
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *


class D2CEncoder(Layer):
    def __init__(self, height, width):
        super(D2CEncoder, self).__init__()
        self.secret_dense = Dense(4096, activation='relu', kernel_initializer='he_normal')

        self.conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv4 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv5 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv6 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        # self.conv7 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')

        self.conva = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.convb = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.convc = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.convd = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conve = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.convf = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')


        self.conv7 = Conv2D(16, 1, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv8 = Conv2D(16, 1, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv9 = Conv2D(3, 1, padding='same', kernel_initializer='he_normal')

    def call(self, inputs):
        secret, image = inputs
        secret = secret
        image = image

        secret = self.secret_dense(secret)
        secret = Reshape((64, 64, 1))(secret)
        secret_enlarged = UpSampling2D(size=(4, 4))(secret)

        conv1_a = self.conva(secret_enlarged)
        conv2_a = self.convb(conv1_a)
        conv3_a = self.convc(conv2_a + conv1_a)
        conv4_a = self.convd(conv3_a + conv2_a + conv1_a)
        conv5_a = self.conve(conv4_a + conv3_a + conv2_a + conv1_a)
        conv6_a = self.convf(conv5_a + conv4_a + conv3_a + conv2_a + conv1_a)
        # conv7_a = self.convg(conv6_a)

        conv1_b = self.conv1(image)
        hyb_conv1 = concatenate([conv1_a, conv1_b], axis=3)
        conv2_b = self.conv2(hyb_conv1)
        hyb_conv2 = concatenate([conv2_a, conv2_b, conv1_b], axis=3)
        conv3_b = self.conv3(hyb_conv2)
        hyb_conv3 = concatenate([conv3_a, conv3_b, conv2_b, conv1_b], axis=3)
        conv4_b = self.conv4(hyb_conv3)
        hyb_conv4 = concatenate([conv4_a, conv4_b, conv3_b, conv2_b, conv1_b], axis=3)
        conv5_b = self.conv5(hyb_conv4)
        hyb_conv5 = concatenate([conv5_a, conv5_b,  conv4_b, conv3_b, conv2_b, conv1_b], axis=3)
        conv6_b = self.conv6(hyb_conv5)
        hyb_conv6 = concatenate([conv6_a, conv6_b, conv5_b,  conv4_b, conv3_b, conv2_b, conv1_b], axis=3)
        conv7 = self.conv7(hyb_conv6)
        output = concatenate([image, conv7])
        conv8 = self.conv8(output)
        # output = concatenate([image, conv8])
        conv9 = self.conv9(conv8)
        return conv9


class D2CDecoder(Layer):
    def __init__(self, height, width):
        super(D2CDecoder, self).__init__()
        self.decoder = Sequential([
            Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
            Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
            Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
            Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
            Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
            Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
            Conv2D(3, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
            Conv2D(1, 1, padding='same', kernel_initializer='he_normal'),
            Flatten(),
            Dense(200)
        ])

    def call(self, image):
        image = image
        return self.decoder(image)


class Discriminator(Layer):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = Sequential([
            Conv2D(8, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(16, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(32, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(64, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(1, (3, 3), activation=None, padding='same')
        ])

    def call(self, image):
        x = image
        x = self.model(x)
        output = tf.reduce_mean(x)
        return output, x


class BuildModel:
    def __init__(self, encoder, decoder, discriminator, secret_input, image_input, l2_edge_gain, borders, secret_size,
                 M, loss_scales, yuv_scales, args, global_step):
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.secret_input = secret_input
        self.image_input = image_input
        self.l2_edge_gain = l2_edge_gain
        self.borders = borders
        self.secret_size = secret_size
        self.M = M
        self.loss_scales = loss_scales
        self.yuv_scales = yuv_scales
        self.args = args
        self.global_step = global_step

    def __call__(self, encoder, decoder, discriminator, secret_input, image_input, l2_edge_gain, borders, secret_size,
                 M, loss_scales, yuv_scales, args, global_step):
        print(M[:, 1, :], "projective_transform_matrix")

        input_warped = tf.contrib.image.transform(image_input, M[:, 1, :], interpolation='BILINEAR')

        mask_warped = tf.contrib.image.transform(tf.ones_like(input_warped), M[:, 1, :], interpolation='BILINEAR')
        input_warped += (1 - mask_warped) * image_input

        residual_warped = encoder((secret_input, input_warped))
        encoded_warped = residual_warped + input_warped
        # encoded_warped = residual_warped
        residual = tf.contrib.image.transform(residual_warped, M[:, 0, :], interpolation='BILINEAR')

        if borders == 'no_edge':
            encoded_image = image_input + residual
        elif borders == 'black':
            encoded_image = residual_warped + input_warped
            encoded_image = tf.contrib.image.transform(encoded_image, M[:, 0, :], interpolation='BILINEAR')
        elif borders.startswith('random'):
            mask = tf.contrib.image.transform(tf.ones_like(residual), M[:, 0, :], interpolation='BILINEAR')
            encoded_image = residual_warped + input_warped
            encoded_image = tf.contrib.image.transform(encoded_image, M[:, 0, :], interpolation='BILINEAR')
            ch = 3 if borders.endswith('rgb') else 1
            encoded_image += (1 - mask) * tf.ones_like(residual) * tf.random.uniform([ch])
        elif borders == 'white':
            mask = tf.contrib.image.transform(tf.ones_like(residual), M[:, 0, :], interpolation='BILINEAR')
            encoded_image = residual_warped + input_warped
            encoded_image = tf.contrib.image.transform(encoded_image, M[:, 0, :], interpolation='BILINEAR')
            encoded_image += (1 - mask) * tf.ones_like(residual)
        elif borders == 'image':
            mask = tf.contrib.image.transform(tf.ones_like(residual), M[:, 0, :], interpolation='BILINEAR')
            encoded_image = residual_warped + input_warped
            encoded_image = tf.contrib.image.transform(encoded_image, M[:, 0, :], interpolation='BILINEAR')
            encoded_image += (1 - mask) * tf.manip.roll(image_input, shift=1, axis=0)
        if borders == 'no_edge':
            D_output_real, _ = discriminator(image_input)
            D_output_fake, D_heatmap = discriminator(encoded_image)
        else:
            D_output_real, _ = discriminator(input_warped)
            D_output_fake, D_heatmap = discriminator(encoded_warped)

        transformation_network = models.TransformNet(encoded_image, args, global_step)

        transformed_image, transform_summaries, image_after_gaussian_blur, image_after_gaussian_noise, image_after_color_transformation = transformation_network(
            encoded_image, args, global_step)

        decoded_secret = decoder(transformed_image)

        getsecretaccuracy = models.GetSecretAccuracy(secret_input, decoded_secret)

        bit_acc, str_acc = getsecretaccuracy(secret_input, decoded_secret)
        bit_loss = 1 - bit_acc

        secret_loss_op = tf.losses.sigmoid_cross_entropy(secret_input, decoded_secret)

        size = (int(image_input.shape[1]), int(image_input.shape[2]))
        gain = 10
        falloff_speed = 4  # Cos dropoff that reaches 0 at distance 1/x into image
        falloff_im = np.ones(size)
        for i in range(int(falloff_im.shape[0] / falloff_speed)):
            falloff_im[-i, :] *= (np.cos(4 * np.pi * i / size[0] + np.pi) + 1) / 2
            falloff_im[i, :] *= (np.cos(4 * np.pi * i / size[0] + np.pi) + 1) / 2
        for j in range(int(falloff_im.shape[1] / falloff_speed)):
            falloff_im[:, -j] *= (np.cos(4 * np.pi * j / size[0] + np.pi) + 1) / 2
            falloff_im[:, j] *= (np.cos(4 * np.pi * j / size[0] + np.pi) + 1) / 2
        falloff_im = 1 - falloff_im
        falloff_im = tf.convert_to_tensor(falloff_im, dtype=tf.float32)
        falloff_im *= l2_edge_gain

        # convert image from RGB to YUV format (better format to store color information of the image)
        encoded_image_yuv = tf.image.rgb_to_yuv(encoded_image)
        image_input_yuv = tf.image.rgb_to_yuv(image_input)
        im_diff = encoded_image_yuv - image_input_yuv
        im_diff += im_diff * tf.expand_dims(falloff_im, axis=[-1])
        yuv_loss_op = tf.reduce_mean(tf.square(im_diff), axis=[0, 1, 2])
        image_loss_op = tf.tensordot(yuv_loss_op, yuv_scales, axes=1)

        D_loss = D_output_real - D_output_fake
        G_loss = D_output_fake

        loss_op = loss_scales[0] * image_loss_op + loss_scales[2] * secret_loss_op
        if not args.no_gan:
            loss_op += loss_scales[3] * G_loss

        summary_op = tf.summary.merge([
                                          tf.summary.scalar('bit_acc', bit_acc, family='train'),
                                          tf.summary.scalar('bit_loss', bit_loss, family='train'),
                                          tf.summary.scalar('str_acc', str_acc, family='train'),
                                          tf.summary.scalar('loss', loss_op, family='train'),
                                          tf.summary.scalar('image_loss', image_loss_op, family='train'),
                                          tf.summary.scalar('G_loss', G_loss, family='train'),
                                          tf.summary.scalar('secret_loss', secret_loss_op, family='train'),
                                          tf.summary.scalar('dis_loss', D_loss, family='train'),
                                          tf.summary.scalar('Y_loss', yuv_loss_op[0], family='color_loss'),
                                          tf.summary.scalar('U_loss', yuv_loss_op[1], family='color_loss'),
                                          tf.summary.scalar('V_loss', yuv_loss_op[2], family='color_loss'),
                                      ] + transform_summaries)

        image_input_summary = models.ImageToSummary(image_input, 'image_input', family='input')
        input_warped_summary = models.ImageToSummary(input_warped, 'input_warped', family='input')
        encoded_warped_summary = models.ImageToSummary(encoded_warped, 'encoded_warped', family='encoded')
        residual_warped_summary = models.ImageToSummary(residual_warped, 'residual_warped', family='encoded')
        encoded_image_summary = models.ImageToSummary(encoded_image, 'encoded_image', family='encoded')
        transformed_image_summary = models.ImageToSummary(transformed_image, 'transformed_image', family='transformed')
        image_after_gaussian_blur_summary = models.ImageToSummary(image_after_gaussian_blur, 'image_after_gaussian_blur', family='transformed')
        image_after_gaussian_noise_summary = models.ImageToSummary(image_after_gaussian_noise, 'image_after_gaussian_noise', family='transformed')
        image_after_color_transformation_summary = models.ImageToSummary(image_after_color_transformation, 'image_after_color_transformation', family='transformed')
        D_heatmap_summary = models.ImageToSummary(D_heatmap, 'discriminator', family='transformed')


        image_summary_op = tf.summary.merge([
            image_input_summary(image_input, 'image_input', family='input'),
            input_warped_summary(input_warped, 'input_warped', family='input'),
            encoded_warped_summary(encoded_warped, 'encoded_warped', family='encoded'),
            residual_warped_summary(residual_warped + .5, 'residual_warped', family='encoded'),
            encoded_image_summary(encoded_image, 'encoded_image', family='encoded'),
            transformed_image_summary(transformed_image, 'transformed_image', family='transformed'),
            image_after_gaussian_blur_summary(image_after_gaussian_blur, 'image_after_gaussian_blur', family='transformed'),
            image_after_gaussian_noise_summary(image_after_gaussian_noise, 'image_after_gaussian_noise', family='transformed'),
            image_after_color_transformation_summary(image_after_color_transformation, 'image_after_color_transformation', family='transformed'),
            D_heatmap_summary(D_heatmap, 'discriminator', family='transformed'),
        ])

        return loss_op, secret_loss_op, D_loss, summary_op, image_summary_op, bit_acc


class TransformNet:
    def __init__(self, encoded_image, args, global_step):
        self.encoded_image = encoded_image
        self.args = args
        self.global_step = global_step

    def __call__(self, encoded_image, args, global_step):
        sh = tf.shape(encoded_image)

        ramp_fn = lambda ramp: tf.minimum(tf.to_float(global_step) / ramp, 1.)

        rnd_bri = ramp_fn(args.rnd_bri_ramp) * args.rnd_bri
        rnd_hue = ramp_fn(args.rnd_hue_ramp) * args.rnd_hue
        rnd_brightness = utils.get_rnd_brightness_tf(rnd_bri, rnd_hue, args.batch_size)

        # JPEG quality
        jpeg_quality = 100. - tf.random.uniform([]) * ramp_fn(args.jpeg_quality_ramp) * (100. - args.jpeg_quality)
        jpeg_factor = tf.cond(tf.less(jpeg_quality, 50), lambda: 5000. / jpeg_quality,
                              lambda: 200. - jpeg_quality * 2) / 100. + .0001

        rnd_noise = tf.random.uniform([]) * ramp_fn(args.rnd_noise_ramp) * args.rnd_noise

        contrast_low = 1. - (1. - args.contrast_low) * ramp_fn(args.contrast_ramp)
        contrast_high = 1. + (args.contrast_high - 1.) * ramp_fn(args.contrast_ramp)
        contrast_params = [contrast_low, contrast_high]

        rnd_sat = tf.random.uniform([]) * ramp_fn(args.rnd_sat_ramp) * args.rnd_sat

        # Blurring
        f = utils.random_blur_kernel(probs=[.25, .25], N_blur=7,
                                     sigrange_gauss=[1., 3.], sigrange_line=[.25, 1.], wmin_line=3)
        encoded_image = tf.nn.conv2d(encoded_image, f, [1, 1, 1, 1], padding='SAME')
        image_after_gaussian_blur = tf.reshape(encoded_image, [-1, 256, 256, 3])

        # Adding gaussian noise
        noise = tf.random_normal(shape=tf.shape(encoded_image), mean=0.0, stddev=rnd_noise, dtype=tf.float32)
        encoded_image = encoded_image + noise
        encoded_image = tf.clip_by_value(encoded_image, 0, 1)
        image_after_gaussian_noise = tf.reshape(encoded_image, [-1, 256, 256, 3])

        # Color transformation
        contrast_scale = tf.random_uniform(shape=[tf.shape(encoded_image)[0]], minval=contrast_params[0],
                                           maxval=contrast_params[1])
        contrast_scale = tf.reshape(contrast_scale, shape=[tf.shape(encoded_image)[0], 1, 1, 1])

        encoded_image = encoded_image * contrast_scale
        encoded_image = encoded_image + rnd_brightness
        encoded_image = tf.clip_by_value(encoded_image, 0, 1)

        encoded_image_lum = tf.expand_dims(tf.reduce_sum(encoded_image * tf.constant([.3, .6, .1]), axis=3), 3)
        encoded_image = (1 - rnd_sat) * encoded_image + rnd_sat * encoded_image_lum

        encoded_image = tf.reshape(encoded_image, [-1, 256, 256, 3])
        image_after_color_transformation = tf.reshape(encoded_image, [-1, 256, 256, 3])

        if not args.no_jpeg:
            encoded_image = utils.jpeg_compress_decompress(encoded_image, rounding=utils.round_only_at_0,
                                                           factor=jpeg_factor, downsample_c=True)

            summaries = [tf.summary.scalar('transformer/rnd_bri', rnd_bri),
                         # writing values of scalar tensor that changes over time or iterations
                         tf.summary.scalar('transformer/rnd_hue', rnd_hue),
                         tf.summary.scalar('transformer/rnd_noise', rnd_noise),
                         tf.summary.scalar('transformer/contrast_low', contrast_low),
                         tf.summary.scalar('transformer/contrast_high', contrast_high),
                         tf.summary.scalar('transformer/jpeg_quality', jpeg_quality)]
        return encoded_image, summaries, image_after_gaussian_blur, image_after_gaussian_noise, image_after_color_transformation


class GetSecretAccuracy:
    def __init__(self, secret_true, secret_pred):
        self.secret_true = secret_true
        self.secret_pred = secret_pred

    def __call__(self, secret_true, secret_pred):
        with tf.variable_scope("acc"):
            secret_pred = tf.round(tf.sigmoid(secret_pred))
            correct_pred = tf.to_int64(tf.shape(secret_pred)[1]) - tf.count_nonzero(secret_pred - secret_true, axis=1)

            str_acc = 1.0 - tf.count_nonzero(correct_pred - tf.to_int64(tf.shape(secret_pred)[1])) / tf.size(
                correct_pred,
                out_type=tf.int64)

            bit_acc = tf.reduce_sum(correct_pred) / tf.size(secret_pred, out_type=tf.int64)
            return bit_acc, str_acc


class ImageToSummary():
    def __init__(self, image, name, family):
        self.image = image
        self.name = name
        self.family = family

    def __call__(self, image, name, family="train"):
        image = tf.clip_by_value(image, 0, 1)
        image = tf.cast(image * 255, dtype=tf.uint8)
        summary = tf.summary.image(name, image, max_outputs=1, family=family)
        return summary


def prepare_deployment_hiding_graph(encoder, secret_input, image_input):
    residual = encoder((secret_input, image_input))
    encoded_image = residual + image_input
    encoded_image = tf.clip_by_value(encoded_image, 0, 1)

    return encoded_image, residual


def prepare_deployment_reveal_graph(decoder, image_input):
    decoded_secret = decoded_secret = decoder(image_input)

    return tf.round(tf.sigmoid(decoded_secret))
