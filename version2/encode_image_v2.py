import bchlib
import glob
import os
from PIL import Image,ImageOps
import numpy as np
import tensorflow as tf
# import tensorflow.contrib.image
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tqdm import tqdm
from PIL import Image
import numpy as np
import time
start_time = time.time()

BCH_POLYNOMIAL = 137
BCH_BITS = 7

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--images_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--secret', type=str, default=None)
    parser.add_argument('--secret_size', type=int, default=20)
    parser.add_argument('--checkerboard_save_dir', type=str, default=None)
    args = parser.parse_args()

    if args.image is not None:
        files_list = [args.image]
    elif args.images_dir is not None:
        files_list = glob.glob(args.images_dir + '/*')
    else:
        print('Missing input image')
        return

    sess = tf.compat.v1.InteractiveSession(graph=tf.Graph())

    model = tf.compat.v1.saved_model.loader.load(sess, [tag_constants.SERVING], args.model)

    input_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['secret'].name
    input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
    input_secret = tf.compat.v1.get_default_graph().get_tensor_by_name(input_secret_name)
    input_image = tf.compat.v1.get_default_graph().get_tensor_by_name(input_image_name)

    output_stegastamp_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['stegastamp'].name
    output_residual_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['residual'].name
    output_stegastamp = tf.compat.v1.get_default_graph().get_tensor_by_name(output_stegastamp_name)
    output_residual = tf.compat.v1.get_default_graph().get_tensor_by_name(output_residual_name)


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
    np.savetxt('C:/Research/COdes/DeepD2C/21DEC04/input_bits/binary_input.txt', secret, delimiter=', ', fmt='% 4d')
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
            # Saving checkerboard input image
            img.save("C:/Research/COdes/DeepD2C/21DEC04/checkerboard_input_images" + f"/{counter}" + "." + "png")
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