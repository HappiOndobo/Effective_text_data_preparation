# Импорт необходимых библиотек
import re
import string
import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np


# Определяем функции для стандартизации и разбиения текста
def custom_standardization_fn(string_tensor):
    # Перевод текста в нижний регистр и удаление знаков пунктуации
    lowercase_string = tf.strings.lower(string_tensor)
    return tf.strings.regex_replace(
        lowercase_string, f"[{re.escape(string.punctuation)}]", "")


def custom_split_fn(string_tensor):
    # Разделение текста на отдельные слова
    return tf.strings.split(string_tensor)


# Создаем слой TextVectorization
text_vectorization = layers.TextVectorization(
    output_mode="int",
    standardize=custom_standardization_fn,
    split=custom_split_fn,
)

# Данные для обучения
dataset = [
    "I write, erase, rewrite",
    "Erase again, and then",
    "A poppy blooms.",
]

# Адаптация TextVectorization к данным
text_vectorization.adapt(dataset)

# Печать словаря
print("Словарь, созданный TextVectorization:")
print(text_vectorization.get_vocabulary())

# Тестовое предложение для кодирования
test_sentence = "I write, rewrite, and still rewrite again"
encoded_sentence = text_vectorization(test_sentence)
print("\nЗакодированное предложение:")
print(encoded_sentence.numpy())

# Визуализация закодированного предложения
plt.figure(figsize=(10, 2))
plt.bar(range(len(encoded_sentence.numpy())), encoded_sentence.numpy())
plt.title("Закодированное предложение")
plt.xlabel("Позиция слова в предложении")
plt.ylabel("Индекс слова")
plt.show()

# Создание обратного словаря
vocabulary = text_vectorization.get_vocabulary()
inverse_vocab = dict(enumerate(vocabulary))

# Декодирование предложения
decoded_sentence = " ".join(inverse_vocab[int(i)] for i in encoded_sentence)
print("\nДекодированное предложение:")
print(decoded_sentence)

# Визуализация первых слов словаря
plt.figure(figsize=(10, 5))
sns.barplot(x=list(range(len(vocabulary[:10]))), y=vocabulary[:10])
plt.title("Первые слова словаря")
plt.xlabel("Индекс")
plt.ylabel("Слово")
plt.xticks(rotation=45)
plt.show()

# Симуляция векторов слов (пример для визуализации)
word_vectors = np.random.rand(len(vocabulary), 10)  # Замените на реальные embeddings, если есть

# PCA для отображения векторов слов в 2D
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(word_vectors)

# Визуализация векторов слов
plt.figure(figsize=(12, 8))
for i, word in enumerate(vocabulary[:30]):  # Ограничение до 30 слов для читаемости
    plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1])
    plt.text(reduced_embeddings[i, 0] + 0.01, reduced_embeddings[i, 1], word)
plt.title("Визуализация embeddings (PCA)")
plt.show()

# Пример визуализации обучения модели
# (Замените history на данные из вашего обучения)
history = {
    'accuracy': [0.6, 0.7, 0.8, 0.85, 0.9],
    'val_accuracy': [0.55, 0.65, 0.75, 0.8, 0.88],
    'loss': [1.2, 0.8, 0.6, 0.5, 0.4],
    'val_loss': [1.3, 0.9, 0.7, 0.6, 0.5],
}

# Визуализация точности
plt.figure(figsize=(12, 6))
plt.plot(history['accuracy'], label='Точность на обучении')
plt.plot(history['val_accuracy'], label='Точность на валидации')
plt.title('График обучения - Точность')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()
plt.show()

# Визуализация потерь
plt.figure(figsize=(12, 6))
plt.plot(history['loss'], label='Потери на обучении')
plt.plot(history['val_loss'], label='Потери на валидации')
plt.title('График обучения - Потери')
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.legend()
plt.show()