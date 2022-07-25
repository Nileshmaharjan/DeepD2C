import bchlib
import glob
import os
from PIL import Image,ImageOps
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tqdm import tqdm
from PIL import Image
import numpy as np
import time
start_time = time.time()
from dotenv import load_dotenv
import os
import datetime

BCH_POLYNOMIAL = 137
BCH_BITS = 7


def main():
    # read env files from environment
    root_directory = os.getenv('root_directory')
    model_directory = os.getenv('model_directory')
    test_image_directory = os.getenv('test_image_directory')
    secret_message = os.getenv('secret')
    secret_size = os.getenv('secret_size')

    # create experiment directory and within it checkerboard, encoded image directory and binary txt file directory
    date = datetime.datetime.now()
    experiment_directory_name = date.strftime("%b") + "-" + date.strftime("%d") + "-" + date.strftime("%H") + "-" + date.strftime("%M") + "-" + date.strftime("%S") + "-" + date.strftime("%p")

    experiment_directory_path = root_directory + experiment_directory_name
    checkerboard_file_directory_name = "/checkerboard-{}".format(int(time.time()))
    checkerboard_file_directory_path = experiment_directory_path + checkerboard_file_directory_name
    encoded_file_directory_name = "/encoded-{}".format(int(time.time()))
    encoded_file_directory_path = experiment_directory_path + encoded_file_directory_name
    print("Checkerboard file directory path: ", checkerboard_file_directory_path)
    print("Encoded file directory path: ", encoded_file_directory_path)
    binary_input_file_name = "binary_input.txt"
    binary_input_file_path = experiment_directory_path + '/' + binary_input_file_name


    if not os.path.exists(experiment_directory_path):
        os.makedirs(experiment_directory_path)
        open(binary_input_file_path, 'a').close()
        if not os.path.exists(checkerboard_file_directory_path):
            os.makedirs(checkerboard_file_directory_path)
        if not os.path.exists(encoded_file_directory_path):
            os.makedirs(encoded_file_directory_path)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=model_directory)
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--images_dir', type=str, default=test_image_directory)
    parser.add_argument('--save_dir', type=str, default=encoded_file_directory_path)
    parser.add_argument('--secret', type=str, default=secret_message)
    parser.add_argument('--secret_size', type=int, default=secret_size)
    parser.add_argument('--checkerboard_save_dir', type=str, default=checkerboard_file_directory_path)
    args = parser.parse_args()

    if args.image is not None:
        files_list = [args.image]
    elif args.images_dir is not None:
        files_list = glob.glob(args.images_dir + '/*')
    else:
        print('Missing input image')
        return

    sess = tf.InteractiveSession(graph=tf.Graph())

    model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], args.model)

    input_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['secret'].name
    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_secret = tf.get_default_graph().get_tensor_by_name(input_secret_name)
    input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_stegastamp_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['stegastamp'].name
    output_residual_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['residual'].name
    output_stegastamp = tf.get_default_graph().get_tensor_by_name(output_stegastamp_name)
    output_residual = tf.get_default_graph().get_tensor_by_name(output_residual_name)


    width = 256
    height = 256

    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)
    print(len(args.secret))
    if len(args.secret) > 70:
        print('Error: Can only encode 56bits (7 characters) with ECC')
        return

    # data = bytearray(args.secret + ' '*(7-len(args.secret)), 'utf-8')
    data = bytearray(args.secret, 'utf-8')

    #BCH encoding
    ecc = bch.encode(data)
    packet = data + ecc
    print("data len:", len(data), ", ecc len:", len(ecc))
    packet_binary = ''.join(format(x, '08b') for x in packet)

    secret = [int(x) for x in packet_binary]
    np.savetxt(binary_input_file_path, secret, delimiter=', ', fmt='% 4d')
    secret_array = np.array(secret)
    print(f"The secret array shape is {secret_array.shape}")
    reshaped_array = np.reshape(secret_array, (-1, 2))
    print(f"The secret array shape is {reshaped_array.shape}")

    print(f'data : {data}\necc : {ecc}\npacket : {packet}\npacket_binary : {packet_binary}\nsecret : {secret}')

    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        size = (width, height)
        counter = 1
        for filename in tqdm(files_list):
            image = Image.open(filename).convert("RGB")
            image = np.array(ImageOps.fit(image,size),dtype=np.float32)
            image /= 255.

            feed_dict = {input_secret:[secret],
                         input_image:[image]}

            hidden_img, residual = sess.run([output_stegastamp, output_residual],feed_dict=feed_dict)
            #residual : data checkerboard image
            residual = np.squeeze(residual, axis=0)
            residual = residual[:, :, 0]
            img = Image.fromarray(residual, 'L')
            print('Counter', counter)
            # Saving checkerboard input image
            img.save(checkerboard_file_directory_path+ f"/{counter}" + "." + "png")
            counter += 1

            #hidden_img : data embedded image
            rescaled = (hidden_img[0] * 255).astype(np.uint8)

            raw_img = (image * 255).astype(np.uint8)

            #what is this??
            residual = residual[0]+.5

            residual = (residual * 255).astype(np.uint8)

            save_name = filename.split('\\')[-1].split('.')[0]

            im = Image.fromarray(np.array(rescaled))
            im.save(args.save_dir + '/'+save_name+'_encoded_.png')


if __name__ == "__main__":
    main()

print("--- %s seconds ---" % (time.time() - start_time))