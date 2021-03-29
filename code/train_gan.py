import argparse

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

RANDOM_SEED = 3
SHUFFLE_BUFFER_SIZE = 10000


def add_boolean_argument(parser: argparse.ArgumentParser, name: str, default=True):
    param_name = name.replace('-', '_')
    parser.add_argument('--' + name, dest=param_name, action='store_true')
    parser.add_argument('--no-' + name, dest=param_name, action='store_false')
    defaults = {param_name: default}
    parser.set_defaults(**defaults)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('-d', '--data-dir', type=str, default=None)
    parser.add_argument('-e', '--epochs', type=int, default=1)
    parser.add_argument('-t', '--train-size', type=int, default=None)
    parser.add_argument('-v', '--validation-size', type=int, default=1)
    parser.add_argument('-c', '--test-size', type=int, default=1)
    add_boolean_argument(parser, 'mixed-precision')
    return parser.parse_args()


class ModelFactory:

    def __init__(self, latent_dim: int):
        self.latent_dim = latent_dim

    def create_discriminator(self):
        init = tf.keras.initializers.RandomNormal(mean=0, stddev=0.02)
        channels = [64, 64, 128, 128, 128, 256]
        kernel_widths = [3, 3, 3, 3, 3, 3]
        strides = [2, 2, 2, 2, 1, 1]

        I = tf.keras.Input(shape=(64, 64, 3))
        X = tf.keras.layers.GaussianNoise(0.01)(I)
        for channel, kernel, stride in zip(channels, kernel_widths, strides):
            X = tf.keras.layers.Conv2D(channel, kernel, strides=stride,
                                       padding='same', kernel_initializer=init)(X)
            X = tf.keras.layers.GaussianDropout(0.005)(X)
            X = tf.keras.layers.LayerNormalization()(X)
            X = tf.keras.layers.LeakyReLU(0.02)(X)
        X = tf.keras.layers.Flatten()(X)
        X = tf.keras.layers.Dense(1, kernel_initializer=init)(X)

        return tf.keras.Model(I, X)

    def create_generator(self):
        starting_shape = [4, 4, 64]
        init = tf.keras.initializers.RandomNormal(mean=0, stddev=0.02)
        hidden_activation = 'relu'
        channels = [256, 128, 128, 64, 64, 3]
        upsampling = [2, 2, 2, 2, 1, 1]
        kernel_widths = [4, 4, 4, 4, 4, 4]
        strides = [1, 1, 1, 1, 1, 1]
        output_activation = 'tanh'

        input_layer = tf.keras.Input((self.latent_dim,))
        X = tf.keras.layers.Dense(np.prod(starting_shape),
                                  kernel_initializer=init)(input_layer)
        X = tf.keras.layers.LayerNormalization()(X)
        if hidden_activation == 'leaky_relu':
            X = tf.keras.layers.LeakyReLU(0.02)(X)
        else:
            X = tf.keras.layers.Activation(hidden_activation)(X)
        X = tf.keras.layers.Reshape(starting_shape)(X)

        Y = tf.keras.layers.Dense(64)(input_layer)
        Y = tf.keras.layers.LayerNormalization()(Y)
        if hidden_activation == 'leaky_relu':
            Y = tf.keras.layers.LeakyReLU(0.02)(Y)
        else:
            Y = tf.keras.layers.Activation(hidden_activation)(Y)
        Y = tf.keras.layers.Reshape((1, 1, 64))(Y)
        Y = tf.keras.layers.UpSampling2D(np.array(starting_shape[:2]))(Y)

        for i in range(len(channels) - 1):
            X = tf.keras.layers.Concatenate()([X, Y])
            X = tf.keras.layers.UpSampling2D(upsampling[i])(X)
            Y = tf.keras.layers.UpSampling2D(upsampling[i])(Y)
            X = tf.keras.layers.Conv2D(channels[i], kernel_widths[i], strides=strides[i],
                                       padding='same', kernel_initializer=init)(X)
            X = tf.keras.layers.LayerNormalization()(X)
            if hidden_activation == 'leaky_relu':
                X = tf.keras.layers.LeakyReLU(0.02)(X)
            else:
                X = tf.keras.layers.Activation(hidden_activation)(X)
        else:
            X = tf.keras.layers.Concatenate()([X, Y])
            X = tf.keras.layers.Conv2D(channels[-1], kernel_widths[-1], strides=strides[-1],
                                       padding='same', kernel_initializer=init)(X)
            output_layer = tf.keras.layers.Activation(output_activation)(X)

        model = tf.keras.Model(input_layer, output_layer)
        return model

    def create_encoder(self):
        image_shape = (64, 64, 3)
        channels = [64, 64, 128, 128, 128, 256]
        kernel_widths = [4, 4, 4, 4, 4, 4]
        strides = [2, 2, 2, 2, 1, 1]
        hidden_activation = "relu"
        output_activation = "linear"
        add_noise = True
        init = tf.keras.initializers.RandomNormal(mean=0, stddev=0.02)

        input_layer = tf.keras.Input(image_shape)
        X = input_layer

        if add_noise:
            X = tf.keras.layers.GaussianNoise(0.01)(X)
        for channel, kernel, stride in zip(channels, kernel_widths, strides):
            X = tf.keras.layers.Conv2D(channel, kernel, strides=stride,
                                       padding='same', kernel_initializer=init)(X)
            if add_noise:
                X = tf.keras.layers.GaussianDropout(0.005)(X)
            X = tf.keras.layers.LayerNormalization()(X)
            if hidden_activation == 'leaky_relu':
                X = tf.keras.layers.LeakyReLU(0.02)(X)
            else:
                X = tf.keras.layers.Activation(hidden_activation)(X)
        X = tf.keras.layers.Flatten()(X)
        X = tf.keras.layers.Dense(self.latent_dim, kernel_initializer=init)(X)
        output_layer = tf.keras.layers.Activation(output_activation)(X)
        model = tf.keras.Model(input_layer, output_layer)
        return model

    def create_discriminator_latent(self):
        layers = 16
        width = 16
        hidden_activation = "leaky_relu"
        add_noise = True

        input_layer = tf.keras.Input((self.latent_dim,))
        F = input_layer
        if add_noise:
            F = tf.keras.layers.GaussianNoise(0.01)(F)
        for i in range(layers):
            X = tf.keras.layers.Dense(width)(F)
            if add_noise:
                X = tf.keras.layers.GaussianDropout(0.005)(X)
            X = tf.keras.layers.LayerNormalization()(X)
            if hidden_activation == 'leaky_relu':
                X = tf.keras.layers.LeakyReLU(0.02)(X)
            else:
                X = tf.keras.layers.Activation(hidden_activation)(X)
            F = tf.keras.layers.Concatenate()([F, X])
        X = tf.keras.layers.Dense(128)(F)
        if hidden_activation == 'leaky_relu':
            X = tf.keras.layers.LeakyReLU(0.02)(X)
        else:
            X = tf.keras.layers.Activation(hidden_activation)(X)
        X = tf.keras.layers.Dense(1)(X)
        output_layer = tf.keras.layers.Activation('sigmoid')(X)
        model = tf.keras.Model(input_layer, output_layer)
        return model


class GAN(tf.keras.Model):
    def __init__(self, discriminator: tf.keras.Model, generator: tf.keras.Model, latent_dim: int):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile_gan(self,
                    d_optimizer: tf.keras.optimizers.Optimizer,
                    g_optimizer: tf.keras.optimizers.Optimizer,
                    loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat([
            tf.ones((batch_size, 1)) * 0.95,
            tf.ones((batch_size, 1)) * 0.05
        ], axis=0)
        # Add random noise to the labels - important trick!
        # labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(2 * batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((2 * batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return {"d_loss": d_loss, "g_loss": g_loss}


class AEGAN(tf.keras.Model):
    def __init__(self,
                 discriminator: tf.keras.Model,
                 generator: tf.keras.Model,
                 discriminator_latent: tf.keras.Model,
                 encoder: tf.keras.Model,
                 latent_dim: int):
        super(AEGAN, self).__init__()
        self.discriminator_latent = discriminator_latent
        self.encoder = encoder
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile_gan(self,
                    d_optimizer: tf.keras.optimizers.Optimizer,
                    g_optimizer: tf.keras.optimizers.Optimizer,
                    loss_fn):
        super(AEGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        batch_size = tf.shape(real_images)[0]

        # Sample random points in the latent space
        real_noise = tf.random.normal(shape=(batch_size, self.latent_dim))
        # Decode them to fake images
        fake_images = self.generator(real_noise)
        # Encode real images, we call it "fake noise"
        fake_noise = self.encoder(real_images)

        # Combine them with real images/noise
        combined_images = tf.concat([fake_images, real_images], axis=0)
        combined_noise = tf.concat([fake_noise, real_noise], axis=0)

        # Assemble labels discriminating real from fake images/noise
        image_labels = tf.concat([
            tf.ones((batch_size, 1)) * 0.95,
            tf.ones((batch_size, 1)) * 0.05
        ], axis=0)
        noise_labels = tf.concat([
            tf.ones((batch_size, 1)) * 0.95,
            tf.ones((batch_size, 1)) * 0.05
        ], axis=0)

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(image_labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        real_noise = tf.random.normal(shape=(2 * batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((2 * batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(real_noise))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return {"d_loss": d_loss, "g_loss": g_loss}


def setup_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(gpus)
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


class LearningRateLoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        g_lr = self.model.g_optimizer._decayed_lr('float32').numpy()
        d_lr = self.model.d_optimizer._decayed_lr('float32').numpy()
        print(f"g_lr: {g_lr}, d_lr: {d_lr}")


class GenerateFakesCallback(tf.keras.callbacks.Callback):

    def __init__(self, latent_dim):
        super(GenerateFakesCallback, self).__init__()
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        for i in range(9):
            random_latent_vectors = tf.random.normal(shape=[1, self.latent_dim])
            fake = self.model.generator(random_latent_vectors, training=False)
            fake = (fake + 1) / 2 * 255
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(fake.numpy()[0].astype("uint8"))
            plt.axis("off")
        plt.savefig(f'fake_{epoch}.png')
        plt.close()


@tf.function
def norm_image_f16(img: tf.Tensor) -> tf.Tensor:
    return tf.cast(img, dtype=tf.float16) / 255 * 2 - 1


@tf.function
def norm_image_f32(img: tf.Tensor) -> tf.Tensor:
    return tf.cast(img, dtype=tf.float32) / 255 * 2 - 1


def train_gan():
    discriminator = model_factory.create_discriminator()
    generator = model_factory.create_generator()
    gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
    gan.compile_gan(
        d_optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0005,
            # learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            #     initial_learning_rate=0.005,
            #     decay_steps=673,
            #     decay_rate=0.96,
            #     staircase=True),
            clipnorm=1,
            beta_1=0.5),
        g_optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0002,
            # learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            #     initial_learning_rate=0.0006,
            #     decay_steps=300,
            #     decay_rate=0.94,
            #     staircase=True),
            clipnorm=1,
            beta_1=0.5),
        loss_fn=tf.keras.losses.BinaryCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.SUM),
    )
    # To limit the execution time, we only train on 100 batches. You can train on
    # the entire dataset. You will need about 20 epochs to get nice results.
    gan.fit(train_ds,
            epochs=args.epochs,
            callbacks=[
                LearningRateLoggingCallback(),
                GenerateFakesCallback(latent_dim=latent_dim)
            ])


def train_aegan():
    discriminator = model_factory.create_discriminator()
    generator = model_factory.create_generator()
    encoder = model_factory.create_encoder()
    discriminator_latent = model_factory.create_discriminator_latent()
    discriminator_latent.compile(
        optimizer=tf.keras.optimizers.Adam(
            clipnorm=1,
            lr=0.0005,
            beta_1=0.5),
        loss=tf.keras.losses.BinaryCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.SUM))
    gan = AEGAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
    gan.compile_gan(
        d_optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0005,
            # learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            #     initial_learning_rate=0.005,
            #     decay_steps=673,
            #     decay_rate=0.96,
            #     staircase=True),
            clipnorm=1,
            beta_1=0.5),
        g_optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0002,
            # learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            #     initial_learning_rate=0.0006,
            #     decay_steps=300,
            #     decay_rate=0.94,
            #     staircase=True),
            clipnorm=1,
            beta_1=0.5),
        loss_fn=tf.keras.losses.BinaryCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.SUM),
    )
    # To limit the execution time, we only train on 100 batches. You can train on
    # the entire dataset. You will need about 20 epochs to get nice results.
    gan.fit(train_ds,
            epochs=args.epochs,
            callbacks=[
                LearningRateLoggingCallback(),
                GenerateFakesCallback(latent_dim=latent_dim)
            ])


if __name__ == '__main__':
    args = parse_args()
    setup_gpus()
    if args.mixed_precision:
        norm_image_func = norm_image_f16
    else:
        norm_image_func = norm_image_f32
    all_ds = tf.keras.preprocessing.image_dataset_from_directory(args.data_dir,
                                                                 batch_size=args.batch_size,
                                                                 shuffle=True,
                                                                 seed=RANDOM_SEED,
                                                                 label_mode=None,
                                                                 image_size=(64, 64)).map(norm_image_func).unbatch()
    test_ds = all_ds.take(args.test_size)
    validation_ds = all_ds.skip(args.test_size).take(args.validation_size).batch(args.batch_size).cache()
    train_ds = all_ds.skip(args.test_size + args.validation_size)
    if args.train_size is not None:
        train_ds = train_ds.take(args.train_size)
    train_ds = train_ds.batch(args.batch_size, drop_remainder=True).cache()

    if args.mixed_precision:
        print('Enabling mixed precision')
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
    else:
        print('Mixed precision is disabled')

    latent_dim = 128
    model_factory = ModelFactory(latent_dim)

    strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
    with strategy.scope():
        train_gan()

        # for images in test_ds.take(1):
        #     for i in range(9):
        #         ax = plt.subplot(3, 3, i + 1)
        #         plt.imshow(images[i].numpy().astype("uint8"))
        #         plt.axis("off")
        #         plt.savefig('test_ds.png')
