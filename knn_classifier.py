import math
import statistics


class KnnClassifier:

    k = None
    x_train = None
    y_train = None

    def __init__(self, k=5):

        self.k = k
        print("k=", k)

    def fit(self, x, y):

        self.x_train = x
        self.y_train = y

        return self

    def predict(self, x):

        y_predict = []

        for i in range(len(x)):
            vector1 = x[i]
            distance_index_list = []

            for j in range(len(self.x_train)):
                vector2 = self.x_train[j]

                distance = self.get_cosine_similarity(vector1, vector2)
                distance_index_list.append((distance, j))

            ordered_distance_index_list = sorted(distance_index_list)
            top_k_distance_index_list = ordered_distance_index_list[:self.k]

            labels = []
            for tuuple in top_k_distance_index_list:
                index_of_training_instance = tuuple[1]
                label_of_training_instance = self.y_train[index_of_training_instance]
                labels.append(label_of_training_instance)

            label = statistics.mode(labels)
            y_predict.append(label)

        return y_predict

    def get_euclidean_distance(self, point1, point2):

        sum = 0
        for i in range(len(point1)):
            val = point1[i]
            difference = val - point2[i]
            sum += math.pow(difference, 2)

        return math.sqrt(sum)

    def get_manhattan_distance(self, point1, point2):

        sumx = 0
        sumy = 0
        for i in range(len(point1)):
            for j in range(len(point2)):
                sumx += (abs(point1[i] - point1[j]))
                sumy += (abs(point2[i] - point2[j]))

        return sumx + sumy

    def get_cosine_similarity(self, point1, point2):

        suma = 0
        sumb = 0
        sumc = 0

        for i in range(len(point1)):
            a = point1[i]
            b = point2[i]

            suma += a * a
            sumb += b * b
            sumc += a * b

        return sumc / math.sqrt(suma * sumb)
