import bchlib
import glob
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import tensorflow.contrib.image
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tqdm import tqdm

import time
start_time = time.time()

from dotenv import load_dotenv
import os

BCH_POLYNOMIAL = 137
BCH_BITS = 7


def main():
    # read env files from environment
    model_directory = os.getenv('model_directory')
    encoded_file_directory_path = os.getenv('encoded_file_directory_path')
    secret_size = os.getenv('secret_size')
    binary_file_path = os.getenv('binary_file_directory_path')
    experiment_directory_path = os.getenv('experiment_directory_path')
    ber_file_name = "ber.txt"
    ber_file_path = experiment_directory_path + '/' + ber_file_name


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,  default=model_directory)
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--images_dir', type=str, default=encoded_file_directory_path)
    parser.add_argument('--secret_size', type=int, default=200)
    args = parser.parse_args()

    if args.image is not None:
        files_list = [args.image]
    elif args.images_dir is not None:
        files_list = glob.glob(args.images_dir + '/*')
    else:
        print('Missing input image')
        return

    binary_input = np.loadtxt(binary_file_path)
    input_data = np.asarray(binary_input, dtype=int).tolist()

    sess = tf.InteractiveSession(graph=tf.Graph())

    model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], args.model)

    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['decoded'].name
    output_secret = tf.get_default_graph().get_tensor_by_name(output_secret_name)

    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    ber_list = []
    index = 1  # used in ber calculation
    for i, filename in enumerate(tqdm(files_list)):
        image = Image.open(filename).convert("RGB")
        image = np.array(ImageOps.fit(image,(256, 256)),dtype=np.float32)
        image /= 255.

        feed_dict = {input_image:[image]}

        decoded_data = sess.run([output_secret],feed_dict=feed_dict)[0][0]

        # value  changed compared to stega stamp here?
        packet_binary = "".join([str(int(bit)) for bit in decoded_data[:16]])
        packet = bytes(int(packet_binary[i : i + 8], 2) for i in range(0, len(packet_binary), 8))
        packet = bytearray(packet)

        data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]

        decoded = np.asarray(decoded_data, dtype=int).tolist()

        err = 0
        n = 200
        for i in range(n):
            print(i)
            if input_data[i] == decoded[i]:
                err = err + 0
            else:
                err = err + 1
        ber = err/n
        ber_list.append(ber)
        index = index + 1
        print('Total error', err)

    avg_ber = sum(ber_list) / index
    f = open(ber_file_path, "w+")
    f.write(f"The average ber value from {index - 1} images is: {avg_ber} ")

    print('-------Finished-------')

if __name__ == "__main__":
    main()
print("--- %s seconds ---" % (time.time() - start_time))