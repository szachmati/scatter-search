from scipy.spatial import distance


class DistanceMatrixGenerator:
    """ Klasa odopowiedzialna za utworzenie macierzy odległości pomiędzy miastami"""

    def euclideanDistance(self, p1, p2) -> float:
        """ Oblicza długość euklidesową między dwoma miastami o współrzędnych (x,y)
            :parameter p1 - [x, y]
            :parameter p2 - [x, y]
        """
        return distance.euclidean(p1, p2)

    def createDistanceMatrix(self, coordinates) -> list:
        """ Funkcja tworząca macierz odległości pomiędzy miastami
            np. [0, 100, 150]
                [50, 0, 100]
                [12, 20, 0]
            gdzie "0" - oznacza aktualne miasto, pozostałe wartości - koszt odwiedzenia danego miasta
        """
        distances = []
        for i in range(0, len(coordinates)):
            temp = []
            for j in range(0, len(coordinates)):
                temp.append(self.euclideanDistance(coordinates[j], coordinates[i]))
            distances.append(temp)
        return distances

