import math
import random


class Solver:
    def demo(self, a, b, c):
        d = b ** 2 - 4 * a * c
        if d > 0:
            disc = math.sqrt(d)
            root1 = (-b + disc) / (2 * a)
            root2 = (-b - disc) / (2 * a)
            return root1, root2
        elif d == 0:
            return -b / (2 * a)
        else:
            return "This equation has no roots"


if __name__ == '__main__':
    solver = Solver()

    while True:
        a = (random.random() - 0.5) * 10
        b = (random.random() - 0.5) * 10
        c = (random.random() - 0.5) * 10
        result = solver.demo(a, b, c)
        print(result)