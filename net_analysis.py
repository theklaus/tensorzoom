import time

import skimage
import skimage.io
import skimage.transform
import tensorflow as tf
import os
import glob
from PIL import Image

from tensorzoom_net import TensorZoomNet


def load_image(path, height=None, width=None):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]
    return skimage.transform.resize(img, (ny, nx), mode='constant', anti_aliasing=True)


def create_tiles(large, height, width, num):
    h_stride = int(height / num)
    w_stride = int(width / num)
    t_tiles = []
    for y in range(num):
        row = []
        for x in range(num):
            t_tile = tf.slice(large, [0, y * h_stride, x * w_stride, 0], [1, h_stride, w_stride, 3])
            row.append(t_tile)
        t_tiles.append(row)
    return t_tiles


def render(pb_path, img_path):
    with tf.Session() as sess:
        img = load_image(img_path)
        contents = tf.expand_dims(tf.constant(img, tf.float32), 0)

        net = TensorZoomNet(pb_path, False)
        net.build(contents)
        fast_output = net.output

        start_time = time.time()
        output = sess.run(fast_output)
        duration = time.time() - start_time
        print ("############")
        print ("output calculated: %.10f sec" % duration)
        print ("############")

        # print image
        _, pb_name = os.path.split(pb_path)
        pb_name, _ = os.path.splitext(pb_name)
        name, ext = os.path.splitext(img_path)
        out_path = name + "_" + pb_name + ext
        skimage.io.imsave(out_path, output[0])
        print ("img saved:", out_path)


def render_sliced(pb_path, img_path, side_num):
    with tf.Session() as sess:
        img = load_image(img_path)
        contents = tf.expand_dims(tf.constant(img, tf.float32), 0)

        # use stitch training method, slice the image into tiles and concat as batches
        tiles = create_tiles(contents, img.shape[0], img.shape[1], side_num)
        batch = tf.concat([tf.concat(tiles[y], 0) for y in range(side_num)], 0)  # row1, row2, ...

        net = TensorZoomNet(pb_path, False)
        net.build(batch)

        # stitch the tiles back together after split the batches
        split = tf.split(net.output, side_num ** 2, 0)
        fast_output = tf.concat([
            tf.concat([split[x] for x in range(side_num * y, side_num * y + side_num)], 2)
            for y in range(side_num)], 1)

        start_time = time.time()
        output = sess.run(fast_output)
        duration = time.time() - start_time
        print ("output calculated: %.10f sec" % duration)

        # print image
        _, pb_name = os.path.split(pb_path)
        pb_name, _ = os.path.splitext(pb_name)
        name, ext = os.path.splitext(img_path)
        out_path = name + "_" + pb_name + ext
        skimage.io.imsave(out_path, output[0])
        print ("img saved:", out_path)


if __name__ == "__main__":
    with tf.device("/gpu:0"):
    # with tf.device("/cpu:0"):
        limit_bigimages = 950; # maximum width/height for using tz6-s-stitch-gen
        
        files = os.listdir("./analysis/")
        num_of_files = 0
        for file in files:
            if file.endswith((".jpg", ".JPG", ".jpeg", ".JPEG")):
                num_of_files += 1
        
        print ()
        print ("############")
        print ("Found {} JPEG files".format(num_of_files) )
        print ("############")
        print ()

        counter = 0
        success = 0
        for file in files:
            if file.endswith((".jpg", ".JPG", ".jpeg", ".JPEG")):
                counter += 1
                print ()
                print ("############")
                print ("File {} of {}".format(counter, num_of_files))
                print ("Converting {}".format(file))
                print ("############")
                my_image = "./analysis/" + file
                
                
                with Image.open(my_image) as image: 
                    width, height = image.size
                   
                try:
                    if width < limit_bigimages and height < limit_bigimages:
                        print ("############")
                        print ("Small image (< %s px width/height)" % limit_bigimages)
                        print ("############")
                        # for small image/ icon/ thumbnail, use non-deblur version has better result
                        render(pb_path='./results/tz6-s-stitch/tz6-s-stitch-gen.npy', img_path = my_image)
                    else:
                        print ("############")
                        print ("Big image (> %s px width/height)" % limit_bigimages)
                        print ("############")
                        # instead, slice the image into 4 smaller images and then join together to form a big one
                        # less memory is used (<1GB) but there will be defects on the boundary of the tiles
                        render_sliced(pb_path='./results/tz6-s-stitch-sblur-lowtv/tz6-s-stitch-sblur-lowtv-gen.npy', img_path = my_image, side_num=4)
                    success += 1
                except Exception as e:
                    print("!!!!!!!!!!!!!!!!!!!!")
                    print("Error with image {}:".format(file))
                    print("" + str(e))
                    print("!!!!!!!!!!!!!!!!!!!!")
                    
                # example for large image / photos from camera, deblur version looks better
                # warning: this example will consume lots of memory (around 9.xxGB)
                # render(pb_path='./results/tz6-s-stitch-sblur-lowtv/tz6-s-stitch-sblur-lowtv-gen.npy', img_path = my_image)
                
        print ()
        print ("############")
        print ("FINISHED - {} of {} successful".format(success, num_of_files))
        print ("############")
        input ("Press Enter to continue...")
