import numpy as np
from sklearn.decomposition import TruncatedSVD
from numpy import transpose as trnsp
from numpy import matmul as mul


class SpectrumPursuit:
    def __init__(self, data):
        self.data = data
        self.residual = data
        self.S = []
        (N, M) = np.shape(self.data)
        self.N = N
        self.M = M
        self.k = 1

    def most_important_direction(self):             # Direction of the first singular vector of the residual
        svd = TruncatedSVD(n_components=1)
        u = svd.fit(trnsp(self.residual))
        return u.components_[0]

    def select_one_data(self, exc='no'):            # Matching the best direction to the data
        u = self.most_important_direction()
        coefficients = mul(trnsp(self.residual), u)
        for c in range(self.M):
            coefficients[c] /= np.linalg.norm(self.residual[:,c])
        selected = np.argmax(coefficients)
        if exc != 'no':
            self.S.remove(exc)
        self.S.append(selected)
        return selected

    def projection(self, indices):                  # Projection of a matrix on its own columns indexed by indices
        data = self.data
        SUB = data[:, indices]
        projection_matrix = np.eye(self.N) - mul(mul(SUB, np.linalg.inv(mul(trnsp(SUB), SUB))), trnsp(SUB))
        return mul(projection_matrix, data)

    def so_far_projection(self, exc='no'):          # Add a new selected sample recursively
        data = self.data
        if exc != 'no':
            SUB = data[:, [x for x in self.S if x != exc]]
        else:
            SUB = data[:, self.S]
        projection_matrix = np.eye(self.N) - mul(mul(SUB, np.linalg.inv(mul(trnsp(SUB), SUB))), trnsp(SUB))
        self.residual = mul(projection_matrix, data)

    def IPM(self, k):                               # IPM Algorithm (CVPR 2019)
        self.k = k
        self.residual = self.data
        for i in range(self.k):
            self.select_one_data()
            self.so_far_projection()
        return self.S

    def SP(self, k, iters=4, init='IPM'):            # SP Algorithm (CVPR 2020)
        self.residual = self.data
        self.k = k
        if init == 'IPM':
            self.S = []
            self.IPM(k)
        elif type(init) == list and len(init) == k:
            self.S = init
        else:
            self.S = list(np.random.choice(self.M, self.k, replace=False))
        for i in range(iters):
            for j in range(self.k):
                self.so_far_projection(self.S[0])
                self.select_one_data(self.S[0])
        self.so_far_projection()
        return self.S


if __name__ == '__main__':
    N = 64
    M = 5000
    R = 10
    np.random.seed(2)
    A = np.random.rand(N, M)
    A = mul(np.random.rand(N, R), np.random.rand(R, M)) + A

    K = 5
    selector = SpectrumPursuit(A)
    S1 = selector.IPM(K)
    err1 = np.linalg.norm(selector.residual, ord='fro')
    S2 = selector.SP(K)
    err2 = np.linalg.norm(selector.residual, ord='fro')
    err3 = np.linalg.norm(selector.projection(np.random.choice(M, K, replace=False)), ord='fro')

    print(S1)
    print(S2)
    print([err1, err2, err3])
