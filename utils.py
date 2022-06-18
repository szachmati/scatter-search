import numpy as np


def calculatePathCost(distanceMatrix: np.array, path: list) -> float:
    """ Funkcja obliczająca koszt podanej trasy
        :parameter distanceMatrix - macierz kosztów podróży pomiedzy miastami
        :parameter path - aktualna ścieżka
        :return distance - całkowity koszt ścieżki
    """
    totalPathDistance = 0
    for i in range(0, len(path) - 1):
        j = i + 1
        totalPathDistance += distanceMatrix[path[i] - 1, path[j] - 1]
    return totalPathDistance
