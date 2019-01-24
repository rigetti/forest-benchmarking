import pytest
import numpy.random
numpy.random.seed(1)  # seed random number generation for all calls to rand_ops

import forest_benchmarking.random_operators as rand_ops
import numpy as np
from sympy.combinatorics import Permutation
from numpy import linalg as la
import forest_benchmarking.distance_measures as dm

D2_SWAP = np.array([[1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1]])
D3_SWAP = np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 1., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 1., 0., 0.],
                    [0., 1., 0., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 1., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 1., 0.],
                    [0., 0., 1., 0., 0., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 1.]])


# AKA P21 which maps H_1 \otimes H_2 to H_2 \otimes H_1 for D=2 and D=3 respectively.

# =================================================================================================
# Test:  Permute tensor factors
# =================================================================================================
def test_permute_tensor_factor_SWAP():
    # test the Dimension two and three SWAP operators
    D2 = 2
    D3 = 3
    perm_swap = [1, 0]
    # Here we test against hard coded SWAP operators
    assert dm.hilbert_schmidt_ip(rand_ops.permute_tensor_factors(D2, perm_swap), D2_SWAP / 4) == 1.0
    assert dm.hilbert_schmidt_ip(rand_ops.permute_tensor_factors(D3, perm_swap), D3_SWAP / 9) == 1.0


def test_permute_tensor_factor_three_qubits():
    # test permutations on three qubits
    D2 = 2
    perm = [1, 2, 0]
    basis = list(range(0, D2))
    states = []
    for jdx in basis:
        emptyvec = np.zeros((D2, 1))
        emptyvec[jdx] = 1
        states.append(emptyvec)
    Pop = rand_ops.permute_tensor_factors(D2, perm)
    fn_ans1 = np.matmul(Pop, np.kron(np.kron(states[0], states[0]), states[1]))
    by_hand_ans1 = np.kron(np.kron(states[0], states[1]), states[0])
    assert np.dot(fn_ans1.T, by_hand_ans1) == 1.0

    fn_ans2 = np.matmul(Pop, np.kron(np.kron(states[0], states[1]), states[0]))
    by_hand_ans2 = np.kron(np.kron(states[1], states[0]), states[0])
    assert np.dot(fn_ans2.T, by_hand_ans2) == 1.0

    fn_ans3 = np.matmul(Pop, np.kron(np.kron(states[1], states[0]), states[0]))
    by_hand_ans3 = np.kron(np.kron(states[0], states[0]), states[1])
    assert np.dot(fn_ans3.T, by_hand_ans3) == 1.0


def test_permute_tensor_factor_four_qubit_permutation_operator():
    # Test the generation of the permutation operator from the symbolic list
    # given to us by SymPy and if that is the same thing from Andrew Scott's papers.

    # Generate basis vectors of D2 dim Hilbert space
    D2 = 2
    basis = list(range(0, D2))
    states = []
    for jdx in basis:
        emptyvec = np.zeros((D2, 1))
        emptyvec[jdx] = 1
        states.append(emptyvec)

    # ------------------------------------
    # Permutation operator convention 1
    # ------------------------------------
    p_13_24 = Permutation(0, 2)(1, 3).list()
    Pop = rand_ops.permute_tensor_factors(D2, p_13_24)
    # See page 2 of https://arxiv.org/pdf/0809.3813.pdf
    # P (|O1> ⊗ |O2> ⊗ |O3> ⊗ |O4>) \mapsto  |O3> ⊗ |O4> ⊗ |O1> ⊗ |O2>

    # Check if |1> ⊗ |0> ⊗ |0> ⊗ |0> -->  |0> ⊗ |0> ⊗ |1> ⊗ |0>
    fn_ans1 = np.matmul(Pop, np.kron(np.kron(np.kron(states[1], states[0]), states[0]), states[0]))
    by_hand_ans1 = np.kron(np.kron(np.kron(states[0], states[0]), states[1]), states[0])
    assert np.dot(fn_ans1.T, by_hand_ans1) == 1.0

    # Check if |0> ⊗ |0> ⊗ |1> ⊗ |0> -->  |1> ⊗ |0> ⊗ |0> ⊗ |0>
    fn_ans2 = np.matmul(Pop, np.kron(np.kron(np.kron(states[0], states[0]), states[1]), states[0]))
    by_hand_ans2 = np.kron(np.kron(np.kron(states[1], states[0]), states[0]), states[0])
    assert np.dot(fn_ans2.T, by_hand_ans2) == 1.0

    # Check if |0> ⊗ |1> ⊗ |0> ⊗ |0> -->  |0> ⊗ |0> ⊗ |0> ⊗ |1>
    fn_ans3 = np.matmul(Pop, np.kron(np.kron(np.kron(states[0], states[1]), states[0]), states[0]))
    by_hand_ans3 = np.kron(np.kron(np.kron(states[0], states[0]), states[0]), states[1])
    assert np.dot(fn_ans3.T, by_hand_ans3) == 1.0

    # Check if |0> ⊗ |0> ⊗ |0> ⊗ |1> -->  |0> ⊗ |1> ⊗ |0> ⊗ |0>
    fn_ans4 = np.matmul(Pop, np.kron(np.kron(np.kron(states[0], states[0]), states[0]), states[1]))
    by_hand_ans4 = np.kron(np.kron(np.kron(states[0], states[1]), states[0]), states[1])
    assert np.dot(fn_ans3.T, by_hand_ans3) == 1.0

    # ------------------------------------
    # Permutation operator convention 2
    # ------------------------------------
    p_4321 = Permutation([3, 2, 1, 0]).list()
    Pop = rand_ops.permute_tensor_factors(D2, p_4321)
    # See Eqn 5.17 of https://arxiv.org/pdf/0711.1017.pdf
    # P (|O1> ⊗ |O2> ⊗ |O3> ⊗ |O4>) \mapsto  |O4> ⊗ |O3> ⊗ |O2> ⊗ |O1>

    # Check if |1> ⊗ |0> ⊗ |0> ⊗ |0> -->  |0> ⊗ |0> ⊗ |0> ⊗ |1>
    fn_ans1 = np.matmul(Pop, np.kron(np.kron(np.kron(states[1], states[0]), states[0]), states[0]))
    by_hand_ans1 = np.kron(np.kron(np.kron(states[0], states[0]), states[0]), states[1])
    assert np.dot(fn_ans1.T, by_hand_ans1) == 1.0

    # Check if |0> ⊗ |1> ⊗ |1> ⊗ |0> -->  |0> ⊗ |1> ⊗ |1> ⊗ |0>
    fn_ans2 = np.matmul(Pop, np.kron(np.kron(np.kron(states[0], states[1]), states[1]), states[0]))
    by_hand_ans2 = np.kron(np.kron(np.kron(states[0], states[1]), states[1]), states[0])
    assert np.dot(fn_ans2.T, by_hand_ans2) == 1.0

    # Check if |1> ⊗ |0> ⊗ |1> ⊗ |0> -->  |0> ⊗ |1> ⊗ |0> ⊗ |1>
    fn_ans3 = np.matmul(Pop, np.kron(np.kron(np.kron(states[1], states[0]), states[1]), states[0]))
    by_hand_ans3 = np.kron(np.kron(np.kron(states[0], states[1]), states[0]), states[1])
    assert np.dot(fn_ans3.T, by_hand_ans3) == 1.0

    # -------------------------------------
    # Permutuation Operator multiplication
    # -------------------------------------
    # See text below Eqn. 5.21 in https://arxiv.org/pdf/0711.1017.pdf
    # P_2341 P_3412 P_4123 = P_2341 P_2341 = P_3412
    p_2341 = Permutation([1, 2, 3, 0]).list()
    P_2341 = rand_ops.permute_tensor_factors(D2, p_2341)
    p_3412 = Permutation([2, 3, 0, 1]).list()
    P_3412 = rand_ops.permute_tensor_factors(D2, p_3412)
    assert np.array_equal(np.matmul(P_2341, P_2341), P_3412)

    # P_2341 P_3412 P_4123 = P_3412
    p_4123 = Permutation([3, 0, 1, 2]).list()
    P_4123 = rand_ops.permute_tensor_factors(D2, p_4123)
    assert np.array_equal(np.matmul(np.matmul(P_2341, P_3412), P_4123), P_3412)


# =================================================================================================
# Test:  Random Unitaries
# =================================================================================================
def test_random_unitaries_are_unitary():
    N_avg = 30
    D2 = 2
    D3 = 3
    avg_trace2 = 0
    avg_det2 = 0
    avg_trace3 = 0
    avg_det3 = 0
    for idx in range(0, N_avg):
        U2 = rand_ops.haar_rand_unitary(D2)
        U3 = rand_ops.haar_rand_unitary(D3)
        avg_trace2 += np.trace(np.matmul(U2, np.conjugate(U2.T))) / N_avg
        avg_det2 += np.abs(la.det(U2)) / N_avg
        avg_trace3 += np.trace(np.matmul(U3, np.conjugate(U3.T))) / N_avg
        avg_det3 += np.abs(la.det(U3)) / N_avg

    assert np.allclose(avg_trace2, 2.0)
    assert np.allclose(avg_det2, 1.0)
    assert np.allclose(avg_trace3, 3.0)
    assert np.allclose(avg_det3, 1.0)


def test_random_unitaries_first_moment():
    # the first moment should be proportional to P_21/D
    N_avg = 50000
    D2 = 2
    D3 = 3
    perm_swap = [1, 0]
    # Permutation operators
    # SWAP aka P_21 which maps H_1 \otimes H_2 to H_2 \otimes H_1
    D2_SWAP = rand_ops.permute_tensor_factors(D2, perm_swap)
    D3_SWAP = rand_ops.permute_tensor_factors(D3, perm_swap)
    X = rand_ops.haar_rand_unitary(D2)
    Id2 = np.matmul(X, np.conjugate(X.T))
    Y = rand_ops.haar_rand_unitary(D3)
    Id3 = np.matmul(Y, np.conjugate(Y.T))
    D2_avg = np.zeros_like(np.kron(Id2, Id2))
    D3_avg = np.zeros_like(np.kron(Id3, Id3))

    for n in range(0, N_avg):
        U2 = rand_ops.haar_rand_unitary(D2)
        U2d = np.conjugate(U2.T)
        D2_avg += np.kron(U2, U2d) / N_avg
        U3 = rand_ops.haar_rand_unitary(D3)
        U3d = np.conjugate(U3.T)
        D3_avg += np.kron(U3, U3d) / N_avg

    # Compute the Frobenius norm of the different between the estimated operator and the answer
    assert np.real(la.norm((D2_avg - D2_SWAP / D2), 'fro')) <= 0.01
    assert np.real(la.norm((D2_avg - D2_SWAP / D2), 'fro')) >= 0.00
    assert np.real(la.norm((D3_avg - D3_SWAP / D3), 'fro')) <= 0.02
    assert np.real(la.norm((D3_avg - D3_SWAP / D3), 'fro')) >= 0.00
    # ^^ this test the first un-numbered equation on page 2 of https://arxiv.org/pdf/0809.3813.pdf
    #   for dimensions 2 and 3


def test_random_unitaries_second_moment():
    # the second moment should be proportional to
    #
    #   P_13_24 + P_14_23             P_1423 + P_1324
    #  ------------------     -     -------------------
    #       ( d^2 -1)                   d ( d^2 -1)
    #

    N_avg = 80000
    D2 = 2
    X = rand_ops.haar_rand_unitary(D2)
    Id2 = np.matmul(X, np.conjugate(X.T))

    # lists representing the permutations
    p_3412 = Permutation([2, 3, 0, 1]).list()
    p_4321 = Permutation([3, 2, 1, 0]).list()
    p_4312 = Permutation([3, 2, 0, 1]).list()
    p_3421 = Permutation([2, 3, 1, 0]).list()
    # Permutation operators
    P_3412 = rand_ops.permute_tensor_factors(D2, p_3412)
    P_4321 = rand_ops.permute_tensor_factors(D2, p_4321)
    P_4312 = rand_ops.permute_tensor_factors(D2, p_4312)
    P_3421 = rand_ops.permute_tensor_factors(D2, p_3421)
    # ^^ convention in https://arxiv.org/pdf/0711.1017.pdf

    ## For archaeological reasons I will leave this code for those who come next..
    ## lists representing the permutations
    # p_14_23 = Permutation(0, 3)(1, 2).list()
    # p_13_24 = Permutation(0, 2)(1, 3).list()
    # p_1423 = Permutation([0, 3, 1, 2]).list()
    # p_1324 = Permutation([0, 2, 1, 3]).list()
    ## Permutation operators
    # P_14_23 = rand_ops.permute_tensor_factors(D2, p_14_23)
    # P_13_24 = rand_ops.permute_tensor_factors(D2, p_13_24)
    # P_1423 = rand_ops.permute_tensor_factors(D2, p_1423)
    # P_1324 = rand_ops.permute_tensor_factors(D2, p_1324)
    ## ^^ convention in https://arxiv.org/pdf/0809.3813.pdf

    # initalize array
    D2_var = np.zeros_like(np.kron(np.kron(np.kron(Id2, Id2), Id2), Id2))

    for n in range(0, N_avg):
        U2 = rand_ops.haar_rand_unitary(D2)
        U2d = np.conjugate(U2.T)
        D2_var += np.kron(np.kron(np.kron(U2, U2), U2d), U2d) / N_avg

    alpha = 1 / (D2 ** 2 - 1)  # 0.3333
    beta = 1 / (D2 * (D2 ** 2 - 1))  # 0.1666

    theanswer1 = alpha * (P_3412 + P_4321) - beta * (P_4312 + P_3421)
    # ^^ Equation 5.17 in https://arxiv.org/pdf/0711.1017.pdf

    ## For archaeological reasons I will leave this code for those who come next..
    # theanswer2 = alpha * (P_13_24 + P_14_23)  - beta * (P_1423 + P_1324)
    ## ^^ Equation at the bottom of page 2 in https://arxiv.org/pdf/0809.3813.pdf

    # round and remove tiny imaginary parts
    truth = np.around(theanswer1, 2)
    estimate = np.around(np.real(D2_var), 2)

    # are the estimated operator and the answer close?
    print(truth)
    print(estimate)
    assert np.allclose(truth, estimate)
    # ^^ this test equation 5.17 in https://arxiv.org/pdf/0711.1017.pdf


# =================================================================================================
# Test:  haar_rand_state(D):
# =================================================================================================
def test_unit_length():
    N_avg = 20
    D = 2
    state_norm_list = []
    for idx in range(0, N_avg):
        state = rand_ops.haar_rand_state(D)
        state_norm_list += [np.matmul(np.matrix.getH(state), state)]
    state_norm = np.asarray(state_norm_list)
    avg_norm = np.mean(state_norm)

    assert np.max(np.absolute(np.imag(state_norm))) < 1e-10
    assert np.min(np.real(state_norm)) >= -1e-10
    assert np.real(avg_norm) <= (1.0 + 1e-10)
    assert np.real(avg_norm) >= (1.0 - 1e-10)


# =================================================================================================
# Test:  Ginibre state matrix
# =================================================================================================
def test_is_positive_operator():
    N_avg = 10
    K = 2

    D = 2
    eigenvallist = []
    for idx in range(0, N_avg):
        eigenval = la.eig(rand_ops.ginibre_state_matrix(D, K))[0]
        eigenvallist += [eigenval]
    eigenvalues = np.asarray(eigenvallist)
    eigenvalues = eigenvalues.reshape(1, D * N_avg)

    assert np.max(np.absolute(np.imag(eigenvalues))) < 1e-10
    assert np.min(np.real(eigenvalues)) >= -1e-10

    D = 3
    eigenvallist = []
    for idx in range(0, N_avg):
        eigenval = la.eig(rand_ops.ginibre_state_matrix(D, K))[0]
        eigenvallist += [eigenval]
    eigenvalues = np.asarray(eigenvallist)
    eigenvalues = eigenvalues.reshape(1, D * N_avg)

    assert np.max(np.absolute(np.imag(eigenvalues))) < 1e-10
    assert np.min(np.real(eigenvalues)) >= -1e-10


def test_is_trace_one():
    N_avg = 100
    K = 2
    D = 2
    avg_trace = 0
    for idx in range(0, N_avg):
        avg_trace += np.trace(rand_ops.ginibre_state_matrix(D, K)) / N_avg

    assert avg_trace <= 1 + 1e-10
    assert avg_trace >= 1 - 1e-10


def test_has_correct_second_moment():
    # Numerically calculate Eq. 3.20 from
    # Zyczkowski and Sommers, J. Phys. A: Math. Gen. 34 7111, (2001)
    #
    #  <Tr[rho^2])_{D,K} = ( D + K ) / ( D * K + 1 )
    #
    #  D is dimension of Hilbert space and K is rank of state matrix

    N_avg = 5000

    K = 2

    D = 2
    avg_purity = 0
    for idx in range(0, N_avg):
        rho = rand_ops.ginibre_state_matrix(D, K)
        avg_purity += np.trace(np.matmul(rho, rho)) / N_avg

    ans = (D + K) / (D * K + 1)
    assert np.absolute(avg_purity - ans) < 1e-2

    D = 3
    avg_purity = 0
    for idx in range(0, N_avg):
        rho = rand_ops.ginibre_state_matrix(D, K)
        avg_purity += np.trace(np.matmul(rho, rho)) / N_avg

    ans = (D + K) / (D * K + 1)
    assert np.absolute(avg_purity - ans) < 1e-2


# =================================================================================================
# Test: State matrix from Bures measure
# =================================================================================================
def test_is_positive_operator():
    N_avg = 30
    D = 2
    eigenvallist = []
    for idx in range(0, N_avg):
        eigenval = la.eig(rand_ops.bures_measure_state_matrix(D))[0]
        eigenvallist += [eigenval]
    eigenvalues = np.asarray(eigenvallist)
    eigenvalues = eigenvalues.reshape(1, D * N_avg)

    assert np.max(np.absolute(np.imag(eigenvalues))) < 1e-10
    assert np.min(np.real(eigenvalues)) >= -1e-10


def test_has_correct_second_moment():
    # Numerically calculate Eq. 3.1 from
    # Sommers and Zyczkowski, J. Phys. A: Math. Gen. 37 8457, (2004)
    #
    #  <Tr[rho^2])_{D} = ( 5*D^2 + 1 ) / [ (2D) *(D^2 + 2 ) ]
    #
    #  D is dimension of Hilbert space

    N_avg = 5000

    D = 2
    avg_purity = 0
    for idx in range(0, N_avg):
        rho = rand_ops.bures_measure_state_matrix(D)
        avg_purity += np.trace(np.matmul(rho, rho)) / N_avg

    ans = (5 * D ** 2 + 1) / ((2 * D) * (D ** 2 + 2))
    assert np.absolute(avg_purity - ans) < 1e-1

    D = 3
    avg_purity = 0
    for idx in range(0, N_avg):
        rho = rand_ops.bures_measure_state_matrix(D)
        avg_purity += np.trace(np.matmul(rho, rho)) / N_avg

    ans = (5 * D ** 2 + 1) / ((2 * D) * (D ** 2 + 2))
    assert np.absolute(avg_purity - ans) < 1e-1


# =================================================================================================
# Test: random CPTP map from BCSZ distribution
# =================================================================================================
def test_BCSZ_dist_is_complete_positive():
    # A quantum channel is completely positive, iff the Choi matrix is non-negative.
    D = 2
    K = 2
    N_avg = 10

    eigenvallist = []
    for idx in range(0, N_avg):
        choi = rand_ops.rand_map_with_BCSZ_dist(D, K)
        eigenval = la.eig(choi)[0]
        eigenvallist += [eigenval]
    eigenvalues = np.asarray(eigenvallist)
    eigenvalues = eigenvalues.reshape(1, D * D * N_avg)
    assert np.max(np.absolute(np.imag(eigenvalues))) < 1e-10
    assert np.min(np.real(eigenvalues)) >= -1e-10


def test_BCSZ_dist_is_trace_preserving():
    D = 2
    K = 2
    choi = rand_ops.rand_map_with_BCSZ_dist(D, K)
    choi_tensor = choi.reshape([D, D, D, D])
    choi_red = np.trace(choi_tensor, axis1=0, axis2=2)
    assert np.isclose(choi_red, np.eye(D)).all()
