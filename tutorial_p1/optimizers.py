import numpy as np
import random


def direct_search(cost_func, w0, tol=1e-4, max_iter=10):
    w = np.array(w0, dtype=float)
    delta = np.ones(len(w)) * 0.5
    for k in range(max_iter):
        h = np.zeros(len(w))
        J_w = cost_func(w)
        for i in range(len(w)):
            w_test = w.copy()
            w_test[i] += delta[i]
            if cost_func(w_test) < J_w:
                h[i] = delta[i]
            else:
                w_test[i] -= 2 * delta[i]
                if cost_func(w_test) < J_w:
                    h[i] = -delta[i]
                else:
                    delta[i] /= 2.0
        if np.max(np.abs(delta)) < tol:
            break

        best_l, best_J = 0, J_w
        for l in [0.5, 1.0, 2.0]:
            if cost_func(w + l * h) < best_J:
                best_l = l
                best_J = cost_func(w + l * h)
        w = w + best_l * h
    return w, cost_func(w)


def flexible_polyhedron(cost_func, w0, tol=1e-4, max_iter=10):
    alpha, gamma, beta, delta = 1.0, 2.0, 0.5, 0.5
    simplex = [np.array(w0, dtype=float)]
    for i in range(len(w0)):
        pt = np.array(w0, dtype=float)
        pt[i] += 0.5
        simplex.append(pt)

    for k in range(max_iter):
        simplex.sort(key=lambda x: cost_func(x))
        w_L, w_H = simplex[0], simplex[-1]
        J_L, J_H = cost_func(w_L), cost_func(w_H)
        c = np.mean(simplex[:-1], axis=0)

        if np.linalg.norm(w_H - w_L) < tol:
            break

        w_R = c + alpha * (c - w_H)
        J_R = cost_func(w_R)

        if J_L <= J_R < J_H:
            simplex[-1] = w_R
        elif J_R < J_L:
            w_E = c + gamma * (w_R - c)
            simplex[-1] = w_E if cost_func(w_E) < J_R else w_R
        else:
            w_C = c + beta * (w_H - c)
            if cost_func(w_C) < J_H:
                simplex[-1] = w_C
            else:
                for i in range(1, len(simplex)):
                    simplex[i] = w_L + delta * (simplex[i] - w_L)
    return simplex[0], cost_func(simplex[0])


def genetic_algorithm(cost_func, pop_size=10, max_gen=5, tol=1e-4, bounds=None):
    if bounds is None:
        bounds = [(0, 50), (0, 50), (0, 5)]

    population = [[random.uniform(b[0], b[1]) for b in bounds] for _ in range(pop_size)]

    for gen in range(max_gen):
        scored = [(ind, cost_func(ind)) for ind in population]
        scored.sort(key=lambda x: x[1])
        if scored[0][1] < tol:
            break

        new_pop = [scored[0][0]]  # Elitism
        n_params = len(bounds)
        while len(new_pop) < pop_size:
            p1 = scored[int((random.random() ** 2) * pop_size) % len(scored)][0]
            p2 = scored[int((random.random() ** 2) * pop_size) % len(scored)][0]

            if n_params > 1:
                cut = random.randint(1, n_params - 1)
                child = np.concatenate((p1[:cut], p2[cut:]))
            else:
                child = p1 if random.random() < 0.5 else p2

            if random.random() < 0.2:  # Mutation
                idx = random.randint(0, n_params - 1)
                child[idx] = max(0, child[idx] + random.gauss(0, 1.0))
            new_pop.append(child)
        population = new_pop

    scored = [(ind, cost_func(ind)) for ind in population]
    scored.sort(key=lambda x: x[1])
    return scored[0][0], scored[0][1]
