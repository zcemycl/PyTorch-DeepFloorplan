import tensorflow as tf
import torch
import torch.nn.functional as F
import os
import gc
import csv
import tqdm

def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto,feature)

def decode_data(features,size=512):
    image = tf.io.decode_raw(features['image'],tf.uint8)
    boundary = tf.io.decode_raw(features['boundary'],tf.uint8)
    room = tf.io.decode_raw(features['room'],tf.uint8)
    door = tf.io.decode_raw(features['door'],tf.uint8)

    #image = tf.cast(image,dtype=tf.float32)
    #image = tf.reshape(image,[size,size,3])
    #boundary = tf.reshape(boundary,[size,size])
    #room = tf.reshape(room,[size,size])
    #door = tf.reshape(door,[size,size])
    
    #breakpoint()

    #image = tf.divide(image,tf.constant(255.0))
    #label_boundary = tf.one_hot(boundary,3,axis=-1)
    #label_room = tf.one_hot(room,9,axis=-1)
    #breakpoint()
    # capacity = batch_size*128 ?

    return list(image.numpy()),list(boundary.numpy()), \
            list(room.numpy()),list(door.numpy())

if __name__ == "__main__":

    feature = {'image':tf.io.FixedLenFeature([],tf.string),
              'boundary':tf.io.FixedLenFeature([],tf.string),
              'room':tf.io.FixedLenFeature([],tf.string),
              'door':tf.io.FixedLenFeature([],tf.string)}

    filename = "dataset/r3d.tfrecords"
    print(os.path.isfile(filename))
    filenames = [filename]
    raw_dataset = tf.data.TFRecordDataset(filenames)

    parsed_dataset = raw_dataset.map(_parse_function)
    
    # 179 is the maximum
    with open('r3d.csv','w',newline='') as csvfile:
        fieldnames = ['image','boundary','room','door']
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
        writer.writeheader()
        for idx,features in enumerate(tqdm.tqdm(parsed_dataset.take(179))):
            image,boundary,room,door = decode_data(features)
            writer.writerow(
            {'image':image,'boundary':boundary,'room':room,
              'door':door})
    gc.collect()
