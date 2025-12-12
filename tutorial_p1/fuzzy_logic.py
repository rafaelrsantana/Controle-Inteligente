import numpy as np


class FuzzyTakagiSugeno:
    """
    Implementação Takagi-Sugeno de Ordem Zero.
    Saídas das regras são constantes (singletons).
    Defuzzificação por Média Ponderada.
    Rápido e eficiente para controle em tempo real e otimização.
    """

    def __init__(self, params):
        # params = [max_erro, max_der, max_saida]
        self.max_erro = params[0]
        self.max_der = params[1]
        self.max_saida = params[2]

    def _triangulo(self, x, a, b, c):
        return max(min((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)), 0)

    def calcular(self, erro, d_erro):
        # Fuzzification
        mu_erro_N = self._triangulo(erro, -2 * self.max_erro, -self.max_erro, 0)
        mu_erro_Z = self._triangulo(erro, -self.max_erro, 0, self.max_erro)
        mu_erro_P = self._triangulo(erro, 0, self.max_erro, 2 * self.max_erro)

        mu_der_N = self._triangulo(d_erro, -2 * self.max_der, -self.max_der, 0)
        mu_der_Z = self._triangulo(d_erro, -self.max_der, 0, self.max_der)
        mu_der_P = self._triangulo(d_erro, 0, self.max_der, 2 * self.max_der)

        # Inference & Defuzzification (Weighted Average)
        numerador = 0.0
        denominador = 0.0

        # Regras (Pesos x Saída Constante)
        # R1: N, N -> Negativo Forte
        w = min(mu_erro_N, mu_der_N)
        numerador += w * (-self.max_saida)
        denominador += w

        # R2: N, Z -> Negativo Médio
        w = min(mu_erro_N, mu_der_Z)
        numerador += w * (-self.max_saida * 0.5)
        denominador += w

        # R3: N, P -> Zero
        w = min(mu_erro_N, mu_der_P)
        numerador += w * (0)
        denominador += w

        # R4: Z, N -> Negativo Fraco
        w = min(mu_erro_Z, mu_der_N)
        numerador += w * (-self.max_saida * 0.2)
        denominador += w

        # R5: Z, Z -> Zero
        w = min(mu_erro_Z, mu_der_Z)
        numerador += w * (0)
        denominador += w

        # R6: Z, P -> Positivo Fraco
        w = min(mu_erro_Z, mu_der_P)
        numerador += w * (self.max_saida * 0.2)
        denominador += w

        # R7: P, N -> Zero
        w = min(mu_erro_P, mu_der_N)
        numerador += w * (0)
        denominador += w

        # R8: P, Z -> Positivo Médio
        w = min(mu_erro_P, mu_der_Z)
        numerador += w * (self.max_saida * 0.5)
        denominador += w

        # R9: P, P -> Positivo Forte
        w = min(mu_erro_P, mu_der_P)
        numerador += w * (self.max_saida)
        denominador += w

        if denominador == 0:
            return 0.0

        return numerador / denominador


class FuzzyMamdani:
    """
    Implementação Mamdani Clássica.
    Saídas das regras são conjuntos fuzzy (triângulos).
    Agregação pelo Máximo.
    Defuzzificação por Centroide de Área (discretizado).
    Mais computacionalmente intensivo.
    """

    def __init__(self, params):
        self.max_erro = params[0]
        self.max_der = params[1]
        self.max_saida = params[2]

        # Discretização do universo de saída para cálculo do centroide
        # 100 pontos entre -max e +max
        self.y_universe = np.linspace(-self.max_saida, self.max_saida, 100)

    def _triangulo(self, x, a, b, c):
        # Vetorizado para suportar arrays numpy (usado na defuzzificação)
        return np.maximum(
            np.minimum((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)), 0
        )

    def calcular(self, erro, d_erro):
        # 1. Fuzzification (Entradas)
        mu_erro_N = self._triangulo(erro, -2 * self.max_erro, -self.max_erro, 0)
        mu_erro_Z = self._triangulo(erro, -self.max_erro, 0, self.max_erro)
        mu_erro_P = self._triangulo(erro, 0, self.max_erro, 2 * self.max_erro)

        mu_der_N = self._triangulo(d_erro, -2 * self.max_der, -self.max_der, 0)
        mu_der_Z = self._triangulo(d_erro, -self.max_der, 0, self.max_der)
        mu_der_P = self._triangulo(d_erro, 0, self.max_der, 2 * self.max_der)

        # 2. Inference (Corte dos conjuntos de saída)
        # Definindo os conjuntos de saída (Triângulos) no universo Y
        # N: Negativo, Z: Zero, P: Positivo (Simplificado para 3 conjuntos na saída para performance, ou mapear os 5 níveis)
        # Vamos mapear os níveis usados no T-S para conjuntos triangulares centrados nesses valores

        # Pesos das regras (Firing Strength)
        weights = []
        # R1: N, N -> Negativo Forte (-1.0)
        weights.append((min(mu_erro_N, mu_der_N), -self.max_saida))
        # R2: N, Z -> Negativo Médio (-0.5)
        weights.append((min(mu_erro_N, mu_der_Z), -self.max_saida * 0.5))
        # R3: N, P -> Zero (0)
        weights.append((min(mu_erro_N, mu_der_P), 0))
        # R4: Z, N -> Negativo Fraco (-0.2)
        weights.append((min(mu_erro_Z, mu_der_N), -self.max_saida * 0.2))
        # R5: Z, Z -> Zero (0)
        weights.append((min(mu_erro_Z, mu_der_Z), 0))
        # R6: Z, P -> Positivo Fraco (0.2)
        weights.append((min(mu_erro_Z, mu_der_P), self.max_saida * 0.2))
        # R7: P, N -> Zero (0)
        weights.append((min(mu_erro_P, mu_der_N), 0))
        # R8: P, Z -> Positivo Médio (0.5)
        weights.append((min(mu_erro_P, mu_der_Z), self.max_saida * 0.5))
        # R9: P, P -> Positivo Forte (1.0)
        weights.append((min(mu_erro_P, mu_der_P), self.max_saida))

        # 3. Aggregation & Defuzzification (Centroid)
        # Acumulador da função de pertinência agregada (Max)
        mu_agg = np.zeros_like(self.y_universe)

        # Largura da base dos triângulos de saída (ajustável, aqui usamos 1/2 do range total para sobreposição)
        width = self.max_saida * 0.5

        for w, center in weights:
            if w > 0:
                # Cria o triângulo de saída centrado em 'center'
                mu_out = self._triangulo(
                    self.y_universe, center - width, center, center + width
                )
                # Corta o triângulo na altura w (Método Mamdani: Min)
                mu_out_clipped = np.minimum(w, mu_out)
                # Agrega com o acumulador (Método Mamdani: Max)
                mu_agg = np.maximum(mu_agg, mu_out_clipped)

        # Cálculo do Centroide: sum(y * mu(y)) / sum(mu(y))
        soma_area = np.sum(mu_agg)

        if soma_area == 0:
            return 0.0

        soma_momento = np.sum(self.y_universe * mu_agg)
        return soma_momento / soma_area
