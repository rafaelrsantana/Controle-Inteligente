import numpy as np

class FuzzyPD:
    
    def __init__(self):

        self.max_erro = 1.0
        self.max_der = 10.0
        self.max_saida = 15.0

    def _triangulo(self, x, a , b, c):

        return max(min((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)), 0)
    
    def calcular(self, erro, d_erro):
        # Para o ERRO
        mu_erro_N = self._triangulo(erro, -2*self.max_erro, -self.max_erro, 0)

        mu_erro_Z = self._triangulo(erro, -self.max_erro, 0, self.max_erro)

        mu_erro_P = self._triangulo(erro, 0, self.max_erro, 2*self.max_erro)

        # Para a DERIVADA
        mu_der_N = self._triangulo(d_erro, -2*self.max_der, -self.max_der, 0)
        mu_der_Z = self._triangulo(d_erro, -self.max_der, 0, self.max_der)
        mu_der_P = self._triangulo(d_erro, 0, self.max_der, 2*self.max_der)

        regras = []

        w1 = min(mu_erro_N, mu_der_N)
        regras.append((w1, -self.max_saida))

        w2 = min(mu_erro_N, mu_der_Z)
        regras.append((w2, -self.max_saida*0.5))

        w3 = min(mu_erro_N, mu_der_P)
        regras.append((w3, 0))

        w4 = min(mu_erro_Z, mu_der_N)
        regras.append((w4, -self.max_saida*0.2))

        w5 = min(mu_erro_Z, mu_der_Z)
        regras.append((w5, 0))

        w6 = min(mu_erro_Z, mu_der_P)
        regras.append((w6, self.max_saida*0.2))

        w7 = min(mu_erro_P, mu_der_N)
        regras.append((w7, 0))

        w8 = min(mu_erro_P, mu_der_Z)
        regras.append((w8, self.max_saida*0.5))

        w9 = min(mu_erro_P, mu_der_P)
        regras.append((w9, self.max_saida))

        numerador = 0.0
        denominador = 0.0

        for peso, valor_saida in regras:
            numerador += peso * valor_saida
            denominador += peso
        
        if denominador == 0:
            return 0.0
        
        u = numerador / denominador
        return u
