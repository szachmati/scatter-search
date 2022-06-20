import random
import numpy as np


class DiversificationGenerator:
    """ Generator dywersyfikujący tworzący nowe rozwiązania początkowe """

    def generate(self, seed: list, startElement: int = 5, n: int = 10) -> list:
        """ Funkcja generująca n rozwiązań w postaci permutacji wygenerowanego wcześniej ziarna
            :parameter seed - ziarno
            :parameter startElement - element ziarna od którego zaczynamy tworzyć permutację np. 5
            :parameter n - liczba rozwiązań do wygenerowania
            Rozwiązanie utworzone zostało na podstawie publikacji:
            https://www.researchgate.net/publication/221024271_A_Template_for_Scatter_Search_and_Path_Relinking
            https://www.researchgate.net/publication/313096950_Diversification_Methods_for_Zero-One_Optimization
        """
        print("*** Diversification Generator :: start ***")
        g = startElement  # np. P(5,5) , P(5,4), P(5,3), P(5, 2), P(5, 1)
        results = []
        maximum = max(seed)
        while len(results) < n:
            permutations = []
            iterator = g
            while iterator > 0:
                subpermutations = [iterator]
                element = iterator
                while True:
                    element += g
                    if element > maximum:
                        break
                    subpermutations.append(element)
                # W celu lepszego zobrazowania działania można odkomentować linię 205
                # print(f"subpermutation: P({g},{iterator}) - {subpermutations}")
                permutations.append(subpermutations)
                iterator -= 1
            g += 1
            results.append(np.concatenate(permutations))
        print(f"*** Diversification Generator :: end - utworzono {len(results)} rozwiązań ***")
        return results

    def generate2(self, dataset: np.array) -> list:
        """" Funkcja generująca ziarno - losową ścieżkę z (1, n) miast
            :parameter dataset - zbiór danych
            :return randomPath
         """
        populationRange = list(range(1, dataset.shape[0] + 1))
        return random.sample(populationRange, dataset.shape[0])