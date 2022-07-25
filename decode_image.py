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

    sess = tf.InteractiveSession(graph=tf.Graph())

    model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], args.model)

    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

    output_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['decoded'].name
    output_secret = tf.get_default_graph().get_tensor_by_name(output_secret_name)

    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    ber_list = []
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

        # bitflips = bch.decode_inplace(data, ecc)
        #er = np.logical_xor(input_data, decoded_data)
        #er = np.count_nonzero(er)
        #ber = er/len(input_data)
        #ber_list.append(ber)

        print('\nDecoded bits :', np.asarray(decoded_data, dtype=int).tolist())
        #print('Input bits   :', np.asarray(input_data, dtype=int).tolist())

        #print(f'{i+1}_Bit Error Rate : {ber}')


        # if bitflips != -1:
        #     try:
        #         code = data.decode("utf-8")
        #         print(filename, code)
        #         continue
        #     except:
        #         continue
        # print(filename, 'Failed to decode')

    print('-------Finished-------')
    #print(f'BER Average : {np.mean(ber_list)}')

if __name__ == "__main__":
    main()
print("--- %s seconds ---" % (time.time() - start_time))