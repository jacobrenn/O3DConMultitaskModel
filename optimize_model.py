import tensorflow as tf
from beyondml import tflow
import pickle
import click

@click.command()
@click.argument('fn', type = click.Path(exists = True, file_okay = True, dir_okay = False))
def main(fn):
    model = tf.keras.models.load_model(fn, custom_objects = tflow.utils.get_custom_objects())

    input1 = tf.keras.layers.Input(model.input_shape[0][1:])
    input2 = tf.keras.layers.Input(model.input_shape[1][1:])
    image_x = tflow.layers.SparseMultiConv2D.from_layer(model.layers[2])([input1, input2])
    image_x = tflow.layers.SparseMultiConv2D.from_layer(model.layers[3])(image_x)
    image_x = tflow.layers.MultiMaxPool2D()(image_x)
    image_x = tflow.layers.SparseMultiConv2D.from_layer(model.layers[5])(image_x)
    image_x = tflow.layers.SparseMultiConv2D.from_layer(model.layers[6])(image_x)
    image_x = tflow.layers.MultiMaxPool2D()(image_x)
    image_x = tflow.layers.SparseMultiConv2D.from_layer(model.layers[11])(image_x)
    image_x = tflow.layers.SparseMultiConv2D.from_layer(model.layers[13])(image_x)
    image_x = tflow.layers.MultiMaxPool2D()(image_x)
    utkface_selector = tflow.layers.SelectorLayer(0)(image_x)
    cifar_selector = tflow.layers.SelectorLayer(1)(image_x)
    utkface_flatten = tf.keras.layers.Flatten()(utkface_selector)
    cifar_flatten = tf.keras.layers.Flatten()(cifar_selector)
    utkface_reshape = tflow.layers.SparseDense.from_layer(model.layers[23])(utkface_flatten)
    cifar_reshape = tflow.layers.SparseDense.from_layer(model.layers[24])(cifar_flatten)

    input3 = tf.keras.layers.Input(model.input_shape[2][1:])
    input4 = tf.keras.layers.Input(model.input_shape[3][1:])
    text_x = tflow.utils.build_token_position_embedding_block(
        128,
        30000,
        512
    )([input3, input4])
    text_x = tflow.utils.build_transformer_block((128, 512), 512, 8, 512)(text_x)
    text_x = tf.keras.layers.GlobalAveragePooling1D()(text_x)
    text_x = tf.keras.layers.Dropout(0.1)(text_x)
    text_x = tflow.layers.SparseDense.from_layer(model.layers[19])(text_x)
    text_x = tf.keras.layers.Dropout(0.1)(text_x)
    text_reshape = tflow.layers.SparseDense.from_layer(model.layers[25])(text_x)

    x = tflow.layers.SparseMultiDense.from_layer(model.layers[26])([utkface_reshape, cifar_reshape, text_reshape])
    x = tflow.layers.SparseMultiDense.from_layer(model.layers[27])(x)
    x = tflow.layers.SparseMultiDense.from_layer(model.layers[28])(x)

    utkface_sel = tflow.layers.SelectorLayer(0)(x)
    cifar_sel = tflow.layers.SelectorLayer(1)(x)
    text_sel = tflow.layers.SelectorLayer(2)(x)

    utkface_out = tflow.layers.SparseDense.from_layer(model.layers[-3])(utkface_sel)
    cifar_out = tflow.layers.SparseDense.from_layer(model.layers[-2])(cifar_sel)
    text_out = tflow.layers.SparseDense.from_layer(model.layers[-1])(text_sel)

    new_model = tf.keras.models.Model(
        [
            input1,
            input2,
            input3,
            input4
        ],
        [
            utkface_out,
            cifar_out,
            text_out
        ]
    )
    new_model.layers[10].set_weights(model.layers[10].get_weights())
    new_model.layers[12].set_weights(model.layers[12].get_weights())

    new_model.summary()
    with open('optimized_model.pkl', 'wb') as f:
        pickle.dump(new_model, f)

if __name__ == '__main__':
    main()
