import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT, LinearAmplitudeFunction
from qiskit.algorithms import IterativeAmplitudeEstimation
from qiskit_aer import AerSimulator
from qiskit_finance.circuit.library import NormalDistribution
import pennylane as qml

# ============================================================
# 1. CLASSICAL PRE-PROCESSING (O(1) per tick)
# ============================================================

class RingBuffer:
    def __init__(self, W):
        self.W = W
        self.buf = np.zeros(W)
        self.idx = 0
        self.count = 0

    def push(self, x):
        old = self.buf[self.idx]
        self.buf[self.idx] = x
        self.idx = (self.idx + 1) % self.W
        self.count = min(self.count + 1, self.W)
        return old, x

class RunningMoments:
    def __init__(self, W):
        self.W = W
        self.mean = 0.0
        self.var = 0.0
        self.sum = 0.0
        self.sum2 = 0.0
        self.n = 0
        self.ema = 0.0

    def update(self, x_new, x_old):
        if self.n < self.W:
            self.n += 1
            self.sum += x_new
            self.sum2 += x_new ** 2
            self.mean = self.sum / self.n
            self.var = (self.sum2 / self.n) - self.mean ** 2
        else:
            self.sum += x_new - x_old
            self.sum2 += x_new ** 2 - x_old ** 2
            self.mean = self.sum / self.W
            self.var = (self.sum2 / self.W) - self.mean ** 2

    def ema_update(self, x, lam):
        self.ema = lam * x + (1 - lam) * self.ema
        return self.ema

# ============================================================
# 2. QUANTUM STAGE 1: QAE for moment estimation
# ============================================================

def build_qae_circuit(n_qubits, f_distribution, f_payoff, n_ancilla=1):
    q = QuantumRegister(n_qubits, 'q')
    a = QuantumRegister(n_ancilla, 'a')
    c = ClassicalRegister(n_ancilla, 'c')
    qc = QuantumCircuit(q, a, c)

    # 1. State preparation
    qc.append(f_distribution, q[:])

    # 2. Payoff encoding (LinearAmplitudeFunction)
    qc.append(f_payoff, q[:] + a[:])

    return qc

def run_qae_estimation(qc, shots=8192):
    # Menggunakan AerSimulator lokal agar bisa langsung diuji
    backend = AerSimulator()
    ae = IterativeAmplitudeEstimation(
        epsilon_target=1e-3,
        alpha=0.05,
        quantum_instance=backend
    )
    result = ae.estimate(qc)
    return result.estimation

# ============================================================
# 3. QUANTUM STAGE 2: VQE for Sweet Spot optimization
# ============================================================

def sweet_spot_hamiltonian(alpha, beta, gamma, mu, sigma, ema):
    # Placeholder Ising Hamiltonian (di produksi, ini adalah SparsePauliOp)
    # H = alpha(S - mu)^2 + ...
    # Untuk testing, kita return dummy Hamiltonian
    coeffs = [alpha, beta, gamma]
    observables = [qml.PauliZ(0), qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)]
    return qml.Hamiltonian(coeffs, observables)

def vqe_ansatz(params, wires):
    n_layers = len(params) // len(wires)
    for l in range(n_layers):
        for i, w in enumerate(wires):
            qml.RY(params[l * len(wires) + i], wires=w)
        for i, w in enumerate(wires):
            qml.RZ(params[l * len(wires) + i], wires=w)
        for i in range(len(wires) - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])

def run_vqe(H, n_qubits, max_iter=200):
    dev = qml.device('default.qubit', wires=n_qubits)
    opt = qml.AdamOptimizer(stepsize=0.01)

    @qml.qnode(dev)
    def cost_fn(params):
        vqe_ansatz(params, range(n_qubits))
        return qml.expval(H)

    params = np.random.uniform(0, 2 * np.pi, size=4 * n_qubits)
    for it in range(max_iter):
        params, energy = opt.step_and_cost(cost_fn, params)

    @qml.qnode(dev)
    def sample_circuit(params):
        vqe_ansatz(params, range(n_qubits))
        return qml.sample()

    samples = sample_circuit(params)
    if len(samples) == 0:
        return 0.0
    # Mengambil bitstring yang paling sering muncul
    best_bitstring = max(set(map(tuple, samples)), key=list(map(tuple, samples)).count)
    S_star = sum(int(b) * 2 ** k for k, b in enumerate(reversed(best_bitstring)))
    return float(S_star)

# ============================================================
# 4. MAIN HYBRID LOOP
# ============================================================

class QuantumSweetSpotEngine:
    def __init__(self, W=256, K=64, alpha=0.4, beta=0.3, gamma=0.3):
        self.W = W
        self.K = K
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.ring = RingBuffer(W)
        self.moments = RunningMoments(W)
        self.tick_count = 0
        self.S_star = 0.0
        self.last_x = 0.0

    def on_tick(self, x, volume=1.0):
        x_old, x_new = self.ring.push(x)
        self.moments.update(x_new, x_old)
        self.moments.ema_update(x, lam=0.05)

        self.tick_count += 1
        self.last_x = x

        # Classical Sweet Spot (Dengan sanity check pembagian nol)
        denominator = self.alpha + (self.gamma * self.moments.var)
        if denominator != 0:
            S_classical = (
                self.alpha * self.moments.mean +
                self.gamma * self.moments.var * self.moments.ema
            ) / denominator
        else:
            S_classical = self.moments.mean

        # Quantum refresh setiap K tick
        if self.tick_count % self.K == 0:
            self.S_star = self._quantum_refresh()
        else:
            self.S_star = S_classical

        return self.S_star

    def _quantum_refresh(self):
        n_qubits = 6  # Dibuat statis untuk testing, di produksi: int(np.log2(self.W))
        # QAE Proxy
        dist = NormalDistribution(2**n_qubits, mu=self.moments.mean, sigma=np.sqrt(max(self.moments.var, 0.0001)))
        payoff = LinearAmplitudeFunction(2**n_qubits, 0, 1) # Placeholder
        qc = build_qae_circuit(n_qubits, dist, payoff)
        
        mu_hat = run_qae_estimation(qc)
        sigma_hat = np.sqrt(max(self.moments.var, 0.0001)) # Proxy variance

        # VQE
        H = sweet_spot_hamiltonian(
            self.alpha, self.beta, self.gamma,
            mu_hat, sigma_hat, self.moments.ema
        )
        S_star = run_vqe(H, n_qubits=n_qubits)
        return S_star

    def handle_data_loss(self, last_valid_x, max_gap=10):
        """Edge case: missing ticks. Interpolasi dan masukkan ke buffer."""
        for i in range(1, max_gap + 1):
            x_interp = last_valid_x + i * (self.moments.mean - last_valid_x) / max_gap
            self.on_tick(x_interp, volume=0.0)
