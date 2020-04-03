import tensorflow_datasets as tfds
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def make_model(img_shape, num_classes, model=None):
    """
    Создаем модель. По умолчанию - InceptionResNetV2.
    Если передан конструктор model, используется он (должен поддерживать
    интерфейс keras.applications).

    Из созданного экземпляра модели (в котором нет выходного слоя, т.е. это
    feature extractor) создается модель, которая будет обучаться.

    (Добавляется GlobalAveragePooling2D и два полносвязных слоя с дропаутом).
    """
    if model is None:
        model = keras.applications.InceptionResNetV2(include_top=False, weights='imagenet')
    else:
        model = model(include_top=False, weights='imagenet')
    for layer in model.layers:
        layer.trainable = False
    input = keras.layers.Input(img_shape)
    custom_model = model(input)
    custom_model = keras.layers.GlobalAveragePooling2D()(custom_model)
    custom_model = keras.layers.Dense(64, activation='relu')(custom_model)
    custom_model = keras.layers.Dropout(0.5)(custom_model)
    predictions = keras.layers.Dense(num_classes, activation='softmax')(custom_model)
    return keras.Model(input, predictions)


def prepare_data(data, image_shape):
    """
    Подготовка данных к формату tf.data.Dataset.

    data - tf.data.Dataset - обрабатывается: при помощи map применяются
    tf.image.resize до размера image_shape, а затем rescale.
    """

    def resize(x, y):
        return tf.image.resize(x, image_shape), y

    def rescale(x, y):
        return tf.cast(x, dtype=tf.float32) / 255., y  # cast из int во float

    return (data
            .map(resize)
            .map(rescale)
            # .cache()  # - лучше не кешировать в память в колабе.
            .shuffle(1024)  # - рандомизированный порядок загрузки изображений
            )


def make_feature_extractor(model, crop_layers):
    """
    Делаем feature_extractor из модели model, отбрасывая crop_layers слоев.
    """
    inp = model.input
    output = model.layers[-(crop_layers + 1)].output
    return keras.Model(inp, output)


def extract(image, model):
    """
    Извлекает признаки из одного изображения image при помощи model.
    Нормализует и возвращает извлеченный вектор с признаками.
    """
    image = np.expand_dims(image, axis=0)
    features = model.predict(image)
    features = features / norm(features)
    return features


def plot_neighbors(images, indices, dist, rows, cols):
    """
    # TODO: переделать так, чтобы размер сетки считался автоматически.
    Строит график с несколькими изображениями.
    images - массив изображений.
    indices - индексы ближайших соседей в images.
    dist - список расстояний  # TODO: должен быть той же размерности, что и индексы
    rows - количество строк на plt.subplots.
    cols - количество столбцов на plt.subplots.
    """
    images = images[indices]
    i = 0
    fig, axis = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 12))
    for row in axis:
        for ax in row:
            ax.imshow(images[i])
            ax.set_title(f"Distance: {dist[i]}")
            i += 1
    plt.show()


# Словарь с классами imagenette
CLASSES = dict(
    enumerate(
        [
            'tench', 'English springer',
            'cassette player', 'chain saw',
            'church', 'French horn',
            'garbage truck', 'gas pump',
            'golf ball', 'parachute'
        ]
    )
)


class SimilaritySearch:
    """
    Класс для поиска похожих изображений при помощи KNN
    """

    def __init__(self, model):
        self.model = model
        # Остальные атрибуты определяются во время обучения.
        self.neig = None
        self.images = None
        self.trained = False
        self._normalized_vectors = None

    def fit(self, images):
        """
        Передаем модели массив изображений images.
        Модель
        1) Создает из них векторы
        2) Нормализует векторы
        3) Обучает sklearn-класс для поиска ближайших соседей.
        4) Выставляет флаг, что обучена
        """
        self.images = images
        feature_vectors = self.model.predict(images)
        self._normalized_vectors = feature_vectors / np.expand_dims(norm(feature_vectors, axis=1), axis=-1)
        self.neig = (NearestNeighbors(n_neighbors=4,
                                      algorithm='brute',
                                      metric='euclidean').
                     fit(self.normalized_vectors))
        self.trained = True

    def find(self, vector):
        """
        Поиск изображений из массива, на котором обучалась модель,
        векторные представления которых наиболее близки к vector.
        Работает, только если модель обучена.
        """
        if not self.trained:
            raise RuntimeError("Model was not trained yet.")
        dist, indices = self.neig.kneighbors([vector])
        plot_neighbors(self.images, indices[0], dist[0], 2, 2)

    def __call__(self, vector):
        """
        Вызов метода find через ()
        """
        return self.find(vector)

    @property
    def normalized_vectors(self):
        return self._normalized_vectors


def make_tsne_plot_from_vectors(vectors, labels=None, figsize=(6, 4.5)):
    """
    Эта функция строит t-SNE и рисует plt.scatterplot.
    Опцилнально plt.scatterplot аннотирован метками labels.
    """
    tsne_results = TSNE(n_components=2,
                        verbose=1,
                        metric='euclidean').fit_transform(vectors)

    cmap = plt.cm.get_cmap('coolwarm')
    plt.figure(figsize=figsize, dpi=80)
    scatter = plt.scatter(tsne_results[:, 0],
                          tsne_results[:, 1],
                          c=labels,
                          cmap=cmap)
    if labels is not None:
        plt.colorbar(scatter)
    plt.show()


def make_pca_then_tsne_plot(vectors, labels=None):
    """
    На основе веторов vectors сначала делаем PCA, затем tsne.
    labels - метки, используемые для аннотации plt.scatterplot.
    """
    # Следующая строчка нужна, чтобы PCA можно было выполнить, когда размерность
    # целевого пространства была задана больше, чем количество векторов.
    # Мы сокращаем ее до количества векторов.
    pca_dimension = min(100, len(vectors))
    pca = PCA(pca_dimension)
    pca.fit(vectors)
    pca_vectors = pca.transform(vectors)
    make_tsne_plot_from_vectors(pca_vectors, labels)


def plot_pair(im_l, im_r, idx):
    """
    Печатает idx-й элемент из "первого" и "второго" массивов на
    одном плоте как пару изображений.
    """
    fig, axis = plt.subplots(nrows=1, ncols=2)
    axis[0].imshow(im_l[idx])
    axis[1].imshow(im_r[idx])
    plt.show()


def _plot_misclassified(ax, image, true_label, predicted_label):
    """
    Рисует изображение в переданном объекте ax.
    Добавляет подписи с настоящим и спрогнозированным классом.
    """
    ax.imshow(image)
    ax.set_title(f"TRUE LABEL: {CLASSES[true_label]},\nPREDICTED: {CLASSES[predicted_label]}")


def get_subplots_grid(n_items, max_n_cols):
    """
    Рассчитывает необходимое количество строк и столбцов,
    исходя из максимального количества столбцов и количества элементов.
    """
    if n_items % max_n_cols == 0:
        n_rows = n_items // max_n_cols
    else:
        n_rows = n_items // max_n_cols + 1
    n_cols = min(n_items, max_n_cols)
    return n_rows, n_cols


def plot_misclassified(misclassified, labels, predictions, max_n_cols=5):
    """
    Получает список ошибочно классифицированных изображений missclassified,
    список _всех_ меток labels,
    список _всех_ лейблов predicitons.
    Эти при помощи _plot_misclassified строит subplots шириной max_n_cols.
    """
    w, h = 4, 3  # ширина и высота области для одного subplot
    n_misc = len(misclassified)
    if n_misc == 0:
        print("There's nothing to plot!")
    elif n_misc == 1:
        n_rows, n_cols = 1, 1
        fig, axis = plt.subplots(nrows=n_rows, ncols=n_cols,
                                 figsize=(w * n_cols, h * n_rows))
        axis.axis('off')
        _plot_misclassified(axis,
                            misclassified[0],
                            labels[labels != predictions][0],
                            predictions[labels != predictions][0])
    else:
        n_rows, n_cols = get_subplots_grid(n_misc, max_n_cols)
        fig, axis = plt.subplots(nrows=n_rows, ncols=n_cols,
                                 figsize=(w * n_cols, h * n_rows))
        for i, image in enumerate(misclassified):
            ax = axis[i] if n_rows == 1 else axis[i // n_cols, i % n_cols]
            ax.axis('off')  # отключает подписи осей
            _plot_misclassified(ax,
                                misclassified[i],
                                labels[labels != predictions][i],
                                predictions[labels != predictions][i])
        plt.show()


def plot_feature_extraction(feature_extractor,
                            images,
                            labels,
                            test_dataset,
                            n_images_to_plot=0):
    """
    Функция для исследования скрытого пространства, которое извлек
    feature_extractor.
    feature extractor - нейросеть, возвращает вектора из n-мерного пространства
    images - набор изображений, который будет передан в нейросеть
    labels - набор меток (нужен для построения t-SNE)
    n_images_to_plot - количество изображений, для которого будут показаны
    три ближайших соседа
    """
    # Обучение модели для поиска ближайших соседей
    search = SimilaritySearch(feature_extractor)
    search.fit(images)
    # Получение нормализованного векторного представления происходит внутри
    # SimilaritySearch таким образом:
    # feature_vectors = feature_extractor.predict(images)
    # feature_vectors / np.expand_dims(norm(feature_vectors, axis=1), axis=-1)
    normalized_vectors = search.normalized_vectors

    for image_num in range(n_images_to_plot):
        search(normalized_vectors[image_num])

    fs_shape = normalized_vectors.shape[-1]
    if fs_shape > 100:  # Если размерность "большая", сначала сделаем PCA
        make_pca_then_tsne_plot(normalized_vectors, labels)
    else:  # Иначе сразу делаем t-SNE
        make_tsne_plot_from_vectors(normalized_vectors, labels)
    return feature_extractor
