import tsplib95
from scipy.spatial import distance


class DistanceMatrixGenerator:
    """ Klasa odopowiedzialna za załadowanie problemu i utworzenie macierzy odległości pomiędzy miastami"""

    def __init__(self, fileName):
        self.fileName = fileName
        self.problem = tsplib95.load(fileName)

    def euclideanDistance(self, p1, p2) -> float:
        """ Oblicza długość euklidesową między dwoma miastami o współrzędnych (x,y)
            :parameter p1 - [x, y]
            :parameter p2 - [x, y]
        """
        return distance.euclidean(p1, p2)

    def createDistanceMatrix(self) -> list:
        """ Funkcja tworząca macierz odległości pomiędzy miastami
            np. [0, 100, 150]
                [50, 0, 100]
                [12, 20, 0]
            gdzie "0" - oznacza aktualne miasto, pozostałe wartości - koszt odwiedzenia danego miasta
        """
        distances = []
        coordinates = list(self.problem.node_coords.values())
        for i in range(0, len(coordinates)):
            temp = []
            for j in range(0, len(coordinates)):
                temp.append(self.euclideanDistance(coordinates[j], coordinates[i]))
            distances.append(temp)
        return distances

