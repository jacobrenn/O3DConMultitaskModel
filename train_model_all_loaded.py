import tensorflow as tf
import numpy as np
import click
from beyondml import tflow
import os
import datasets
from tqdm import tqdm

from sklearn.metrics import classification_report

DEFAULT_BATCH_SIZE = 256
DEFAULT_IMAGE_SIZE = (128, 128, 3)
DEFAULT_SCALING = 1./255
DEFAULT_TEXT_LENGTH = 128
DEFAULT_VOCAB_SIZE = 30000
DEFAULT_EMBED_DIM = 512
DEFAULT_NUM_HEADS = 8
DEFAULT_FF_DIM = 512
DEFAULT_DROPOUT = 0.1

def load_data(
    utkface_dir,
    cifar10_images,
    cifar10_labels,
    text_sequences,
    text_positions,
    text_labels,
    image_size = DEFAULT_IMAGE_SIZE,
    scaling = DEFAULT_SCALING,
):

    files = os.listdir(utkface_dir)
    np.random.shuffle(files)

    cifar10_images = tf.image.resize(cifar10_images, (image_size[0], image_size[1]))

    text_indexes = np.arange(text_labels.shape[0])
    np.random.shuffle(text_indexes)
    text_sequences = text_sequences[text_indexes]
    text_positions = text_positions[text_indexes]
    text_labels = text_labels[text_indexes]

    cutoffs = list(range(10, 100, 10))

    utkface_images = []
    utkface_labels = []
    for file in tqdm(files):
        img = tf.keras.preprocessing.image.load_img(
                os.path.join(utkface_dir, file),
                target_size = (image_size[0], image_size[1])
            )
        utkface_img = np.array(img)*scaling
        age = int(file.split('_')[0])
        age_label = sum([age > cutoff for cutoff in cutoffs])
        utkface_images.append(utkface_img)
        utkface_labels.append(age_label)

    utkface_images = np.asarray(utkface_images)
    utkface_labels = np.asarray(utkface_labels)

    data_size = min([utkface_images.shape[0], cifar10_images.shape[0], text_sequences.shape[0]])

    utkface_images = utkface_images[:data_size]
    utkface_labels = utkface_labels[:data_size]

    cifar10_images = cifar10_images[:data_size]
    cifar10_labels = cifar10_labels[:data_size]

    text_sequences = text_sequences[:data_size]
    text_positions = text_positions[:data_size]
    text_labels = text_labels[:data_size]

    return (utkface_images, cifar10_images, text_sequences, text_positions), (utkface_labels, cifar10_labels, text_labels)

def data_generator(
        utkface_dir,
        cifar10_images,
        cifar10_labels,
        text_sequences,
        text_indices,
        text_labels,
        batch_size = DEFAULT_BATCH_SIZE,
        image_size = DEFAULT_IMAGE_SIZE,
        scaling = DEFAULT_SCALING,
        text_length = DEFAULT_TEXT_LENGTH
):
    files = os.listdir(utkface_dir)
    np.random.shuffle(files)

    cifar10_images = tf.image.resize(cifar10_images, (image_size[0], image_size[1]))
    cifar10_images = cifar10_images*scaling

    text_indexes = np.arange(text_labels.shape[0])
    np.random.shuffle(text_indexes)
    text_indices = text_indices[text_indexes]
    text_labels = text_labels[text_indexes]

    cutoffs = list(range(10, 100, 10))

    utkface_idx = 0
    cifar10_idx = 0
    sequence_idx = 0

    while True:
        utkface_batch = []
        cifar10_batch = []
        sequences_batch = []
        indices_batch = []

        ages = []
        cifar10_batch_labels = []
        sequence_labels = []

        for _ in range(batch_size):
            if utkface_idx >= len(files):
                np.random.shuffle(files)
                utkface_idx = 0
            img = tf.keras.preprocessing.image.load_img(
                os.path.join(utkface_dir, files[utkface_idx]),
                target_size = (image_size[0], image_size[1])
            )
            utkface_img = np.array(img)*scaling
            age = int(files[utkface_idx].split('_')[0])
            age_label = sum([age > cutoff for cutoff in cutoffs])
            utkface_batch.append(utkface_img)
            ages.append(age_label)

            if cifar10_idx >= cifar10_images.shape[0]:
                cifar10_idx = 0
            cifar10_batch.append(cifar10_images[cifar10_idx])
            cifar10_batch_labels.append(cifar10_labels[cifar10_idx])

            if sequence_idx >= text_sequences.shape[0]:
                sequence_idx = 0
            sequences_batch.append(text_sequences[sequence_idx])
            indices_batch.append(text_indices[sequence_idx])
            sequence_labels.append(text_labels[sequence_idx])

            utkface_idx += 1
            cifar10_idx += 1

        yield ([np.asarray(utkface_batch), np.asarray(cifar10_batch), np.asarray(sequences_batch), np.asarray(indices_batch)], [np.asarray(ages), np.asarray(cifar10_batch_labels), np.zeros(batch_size)])

def build_model(
        text_length = DEFAULT_TEXT_LENGTH,
        vocab_size = DEFAULT_VOCAB_SIZE,
        embed_dim = DEFAULT_EMBED_DIM,
        num_heads = DEFAULT_NUM_HEADS,
        ff_dim = DEFAULT_FF_DIM,
        dropout = DEFAULT_DROPOUT
):
    utkface_input = tf.keras.layers.Input(DEFAULT_IMAGE_SIZE)
    cifar10_input = tf.keras.layers.Input(DEFAULT_IMAGE_SIZE)
    conv_x = tflow.layers.MultiMaskedConv2D(16, activation = 'relu')([utkface_input, cifar10_input])
    conv_x = tflow.layers.MultiMaskedConv2D(16, activation = 'relu')(conv_x)
    conv_x = tflow.layers.MultiMaxPool2D()(conv_x)
    conv_x = tflow.layers.MultiMaskedConv2D(32, activation = 'relu')(conv_x)
    conv_x = tflow.layers.MultiMaskedConv2D(32, activation = 'relu')(conv_x)
    conv_x = tflow.layers.MultiMaxPool2D()(conv_x)
    conv_x = tflow.layers.MultiMaskedConv2D(64, activation = 'relu')(conv_x)
    conv_x = tflow.layers.MultiMaskedConv2D(64, activation = 'relu')(conv_x)
    conv_x = tflow.layers.MultiMaxPool2D()(conv_x)
    utkface_sel = tflow.layers.SelectorLayer(0)(conv_x)
    cifar10_sel = tflow.layers.SelectorLayer(1)(conv_x)
    utkface_flatten = tf.keras.layers.Flatten()(utkface_sel)
    cifar10_flatten = tf.keras.layers.Flatten()(cifar10_sel)
    utkface_reshape = tflow.layers.MaskedDense(128, activation = 'relu')(utkface_flatten)
    cifar10_reshape = tflow.layers.MaskedDense(128, activation = 'relu')(cifar10_flatten)

    token_input = tf.keras.layers.Input(text_length)
    pos_input = tf.keras.layers.Input(text_length)
    tok_pos_embed_block = tflow.utils.build_token_position_embedding_block(
        text_length,
        vocab_size,
        embed_dim
    )([token_input, pos_input])
    transformer_block = tflow.utils.build_transformer_block((text_length, embed_dim), embed_dim, num_heads, ff_dim)(tok_pos_embed_block)
    text_x = tf.keras.layers.GlobalAveragePooling1D()(transformer_block)
    text_x = tf.keras.layers.Dropout(dropout)(text_x)
    text_x = tf.keras.layers.Dense(ff_dim, activation = 'relu')(text_x)
    text_x = tf.keras.layers.Dropout(dropout)(text_x)
    text_reshape = tflow.layers.MaskedDense(128, activation = 'relu')(text_x)

    x = tflow.layers.MultiMaskedDense(128, activation = 'relu')(
        [
            utkface_reshape,
            cifar10_reshape,
            text_reshape
        ]
    )
    x = tflow.layers.MultiMaskedDense(128, activation = 'relu')(x)
    x = tflow.layers.MultiMaskedDense(128, activation = 'relu')(x)
    utkface_sel = tflow.layers.SelectorLayer(0)(x)
    cifar10_sel = tflow.layers.SelectorLayer(1)(x)
    topic_sel = tflow.layers.SelectorLayer(2)(x)

    utkface_output = tflow.layers.MaskedDense(10, activation = 'softmax')(utkface_sel)
    cifar_output = tflow.layers.MaskedDense(10, activation = 'softmax')(cifar10_sel)
    topic_output = tflow.layers.MaskedDense(4, activation = 'softmax')(topic_sel)

    model = tf.keras.models.Model(
        [
            utkface_input,
            cifar10_input,
            token_input,
            pos_input
        ],
        [
            utkface_output,
            cifar_output,
            topic_output
        ]
    )
    model = tflow.utils.add_layer_masks(model)
    return model

@click.command()
@click.argument('train-dir', type = click.Path(exists = True, dir_okay = True, file_okay = False))
@click.argument('val-dir', type = click.Path(exists = True, dir_okay = True, file_okay = False))
@click.option('--batch-size', '-b', type = int, default = DEFAULT_BATCH_SIZE)
@click.option('--text-length', '-t', type = int, default = DEFAULT_TEXT_LENGTH)
@click.option('--vocab-size', '-v', type = int, default = DEFAULT_VOCAB_SIZE)
@click.option('--embed-dim', '-e', type = int, default = DEFAULT_EMBED_DIM)
@click.option('--num-heads', '-h', type = int, default = DEFAULT_NUM_HEADS)
@click.option('--ff-dim', '-f', type = int, default = DEFAULT_FF_DIM)
@click.option('--limit', '-l', type = int, default = None)
def main(
        train_dir,
        val_dir,
        batch_size,
        text_length,
        vocab_size,
        embed_dim,
        num_heads,
        ff_dim,
        limit
):
    (cifar10_x_train, cifar10_y_train), (cifar10_x_test, cifar10_y_test) = tf.keras.datasets.cifar10.load_data()
    cifar10_x_train = cifar10_x_train/255
    cifar10_x_test = cifar10_x_test/255
    ag_news_data = datasets.load_dataset('ag_news')
    train_text = ag_news_data['train']['text']
    train_labels = np.asarray(ag_news_data['train']['label'])
    test_text = ag_news_data['test']['text']
    test_labels = np.asarray(ag_news_data['test']['label'])

    tokenizer = tf.keras.preprocessing.text.Tokenizer(DEFAULT_VOCAB_SIZE)
    tokenizer.fit_on_texts(train_text)
    train_sequences = tokenizer.texts_to_sequences(train_text)
    train_sequences = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, text_length)
    train_positions = np.asarray([np.arange(train_sequences.shape[1])] * train_sequences.shape[0])
    
    test_sequences = tokenizer.texts_to_sequences(test_text)
    test_sequences = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, text_length)
    test_positions = np.asarray([np.arange(test_sequences.shape[1])] * test_sequences.shape[0])

    (train_utkface_images, train_cifar_images, train_text_sequences, train_text_positions), (train_utkface_labels, train_cifar_labels, train_text_labels) = load_data(train_dir, cifar10_x_train, cifar10_y_train, train_sequences, train_positions, train_labels)
    (val_utkface_images, val_cifar_images, val_text_sequences, val_text_positions), (val_utkface_labels, val_cifar_labels, val_text_labels) = load_data(val_dir, cifar10_x_test, cifar10_y_test, test_sequences, test_positions, test_labels)

    print(train_utkface_images.shape)
    print(train_cifar_images.shape)
    print(train_text_sequences.shape)
    print(train_text_positions.shape)

    print(train_utkface_images[0])
    print('\n')
    print(train_cifar_images[0])
    print('\n')
    print(train_text_sequences[0])
    print('\n')
    print(train_text_positions[0])
    print('\n')
    print(train_utkface_labels[:10])
    print('\n')
    print(train_cifar_labels[:10])
    print('\n')
    print(train_text_labels[:10])

    model = build_model(
        text_length = text_length,
        vocab_size = vocab_size,
        embed_dim = embed_dim,
        num_heads = num_heads,
        ff_dim = ff_dim,
        dropout = DEFAULT_DROPOUT
    )
    model.compile(
        loss = 'sparse_categorical_crossentropy',
        metrics = 'accuracy',
        optimizer = 'adam'
    )

    callback = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_loss',
        min_delta = 0.01,
        patience = 5,
        restore_best_weights = True
    )

    # Fit the model on image data for one epoch
    model.fit(
        [train_utkface_images, train_cifar_images, train_text_sequences, train_text_positions],
        [train_utkface_labels, train_cifar_labels, train_text_labels],
        epochs = 4,
        batch_size = 256,
        callbacks = [callback]
    )

    model = tflow.utils.mask_model(
        model,
        70,
        method = 'magnitude',
        exclusive = False
    )
    model.compile(
        loss = 'sparse_categorical_crossentropy',
        metrics = 'accuracy',
        optimizer = 'adam'
    )

    model.fit(
        [train_utkface_images, train_cifar_images, train_text_sequences, train_text_positions],
        [train_utkface_labels, train_cifar_labels, train_text_labels],
        epochs = 100,
        batch_size = 256,
        validation_split = 0.2,
        callbacks = [callback]
    )
    model.save('o3dcon_model.h5')

    model = tflow.utils.remove_layer_masks(model)
    model.save('o3dcon_model_pruned.h5')

    preds = model.predict([val_utkface_images, val_cifar_images, val_text_sequences, val_text_positions])
    utkface_preds = preds[0].argmax(axis = 1)
    cifar_preds = preds[1].argmax(axis = 1)
    text_preds = preds[2].argmax(axis = 1)

    print('UTKFace:')
    print(classification_report(val_utkface_labels, utkface_preds))
    print('\n\n')
    print('CIFAR10:')
    print(classification_report(val_cifar_labels, cifar_preds))
    print('\n\n')
    print('AGNews:')
    print(classification_report(val_text_labels, text_preds))

if __name__ == '__main__':
    main()

    
    
