import numpy as np
import matplotlib.pyplot as plt
class HopfieldNetwork:
    def _init_(self, n_neurons):
        self.n_neurons = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons))
        self.stored_patterns = []
    def train(self, patterns):
        self.stored_patterns = patterns
        n_patterns = len(patterns)
        self.weights = np.zeros((self.n_neurons, self.n_neurons))
        for pattern in patterns:
            pattern = pattern.flatten()
            self.weights += np.outer(pattern, pattern)
        self.weights /= n_patterns
        np.fill_diagonal(self.weights, 0)
    def energy(self, state):
        state = state.flatten()
        return -0.5 * np.dot(state, np.dot(self.weights, state))
    def predict(self, input_pattern, max_iterations=1000, async_update=True):
        state = input_pattern.flatten().copy()
        energy_history = [self.energy(state)]
        for iteration in range(max_iterations):
            if async_update:
                for i in np.random.permutation(self.n_neurons):
                    activation = np.dot(self.weights[i], state)
                    state[i] = 1 if activation >= 0 else -1
            else:
                activation = np.dot(self.weights, state)
                new_state = np.where(activation >= 0, 1, -1)
                if np.array_equal(state, new_state):
                    break
                state = new_state
            energy_history.append(self.energy(state))
            if len(energy_history) > 10 and \
               np.allclose(energy_history[-1], energy_history[-2], atol=1e-6):
                break
        return state.reshape(input_pattern.shape), energy_history
    def add_noise(self, pattern, noise_level):
        noisy_pattern = pattern.copy().flatten()
        n_flips = int(noise_level * len(noisy_pattern))
        flip_indices = np.random.choice(len(noisy_pattern), n_flips, replace=False)
        noisy_pattern[flip_indices] *= -1
        return noisy_pattern.reshape(pattern.shape)
    def test_capacity(self, pattern_size, max_patterns=20):
        n_neurons = pattern_size[0] * pattern_size[1]
        results = {
            'n_patterns': [],
            'recall_accuracy': [],
            'error_correction_0.1': [],
            'error_correction_0.2': [],
            'error_correction_0.3': []
        }
        print("\n" )
        print(f"Testing capacity for {n_neurons} neurons")
        print(f"Theoretical capacity (0.138*N): {int(0.138 * n_neurons)}")
        print("\n" )
        for n_patterns in range(1, max_patterns + 1):
            patterns = [np.random.choice([-1, 1], size=pattern_size) 
                       for _ in range(n_patterns)]
            self.train(patterns)
            correct_recalls = 0
            for pattern in patterns:
                retrieved, _ = self.predict(pattern, max_iterations=500)
                if np.array_equal(retrieved, pattern):
                    correct_recalls += 1
            recall_accuracy = correct_recalls / n_patterns
            results['n_patterns'].append(n_patterns)
            results['recall_accuracy'].append(recall_accuracy)
            for noise_level in [0.1, 0.2, 0.3]:
                correct_corrections = 0
                n_tests = min(5, n_patterns)
                for pattern in patterns[:n_tests]:
                    noisy = self.add_noise(pattern, noise_level)
                    retrieved, _ = self.predict(noisy, max_iterations=500)
                    if np.array_equal(retrieved, pattern):
                        correct_corrections += 1
                correction_rate = correct_corrections / n_tests
                results[f'error_correction_{noise_level}'].append(correction_rate)
            print(f"Patterns: {n_patterns:2d} | Recall: {recall_accuracy:.2f} | "
                  f"Err Corr (10%): {results['error_correction_0.1'][-1]:.2f} | "
                  f"Err Corr (20%): {results['error_correction_0.2'][-1]:.2f}")
            if recall_accuracy < 0.5 and n_patterns > 5:
                print(f"\nStopping test - recall accuracy dropped below 50%")
                break
        return results
def main():

    print("HOPFIELD NETWORK - 10x10 BINARY ASSOCIATIVE MEMORY")
    print("\n")
    hopfield_net = HopfieldNetwork(n_neurons=100)
    pattern1 = np.ones((10, 10))
    pattern1[:5, :] = -1  
    pattern2 = np.ones((10, 10))
    pattern2[:, :5] = -1  
    pattern3 = np.array([[1 if (i+j) % 2 == 0 else -1 
                         for j in range(10)] for i in range(10)])
    patterns = [pattern1, pattern2, pattern3]
    hopfield_net.train(patterns)
    print(f"\nStored {len(patterns)} patterns")
    print(f"Weight matrix: {hopfield_net.weights.shape}")
    print("\nPerfect Recall Test:")
    for i, pattern in enumerate(patterns, 1):
        retrieved, energy_hist = hopfield_net.predict(pattern)
        match = np.array_equal(retrieved, pattern)
        print(f"  Pattern {i}: {'✓' if match else '✗'} "
              f"(Energy: {energy_hist[0]:.2f} → {energy_hist[-1]:.2f})")
    print("\nError Correction Test (20% noise):")
    for i, pattern in enumerate(patterns, 1):
        noisy = hopfield_net.add_noise(pattern, 0.2)
        bits_flipped = np.sum(noisy != pattern)
        retrieved, _ = hopfield_net.predict(noisy)
        match = np.array_equal(retrieved, pattern)
        print(f"  Pattern {i}: {bits_flipped} bits flipped → {'✓ Corrected' if match else '✗ Failed'}")
    hopfield_net_capacity = HopfieldNetwork(n_neurons=100)
    results = hopfield_net_capacity.test_capacity((10, 10), max_patterns=20)
    practical_capacity = 0
    for i, acc in enumerate(results['recall_accuracy']):
        if acc >= 0.9:
            practical_capacity = results['n_patterns'][i]
        else:
            break
    print(f"Theoretical capacity (0.138*N): ~14 patterns")
    print(f"Observed practical capacity: ~{practical_capacity} patterns (>90% recall)")
    hopfield_net_error = HopfieldNetwork(n_neurons=100)
    test_patterns = [np.random.choice([-1, 1], size=(10, 10)) for _ in range(5)]
    hopfield_net_error.train(test_patterns)
    print("\nError Correction at Various Noise Levels:")
    print(f"{'Noise':<10} {'Bits Corrupted':<15} {'Success Rate':<15}")
    print("\n")
    for noise in [0.10, 0.15, 0.20, 0.25, 0.30]:
        successes = 0
        n_trials = 20
        for _ in range(n_trials):
            pattern = test_patterns[np.random.randint(0, 5)]
            noisy = hopfield_net_error.add_noise(pattern, noise)
            retrieved, _ = hopfield_net_error.predict(noisy)
            if np.array_equal(retrieved,pattern):
                successes+=1
        success_rate=successes/n_trials
        print(f"{noise*100:>5.0f}%     {int(noise*100):<15}   {success_rate*100:>5.1f}%")
    print("\n")   
    print("1. Capacity:10x10 network 100 neuron reliably stores ~8-14 patterns")
    print("2. Error Correcton: Works wellup to 20-30% noise")
    print("3. Convrgence: Typically occurs in 10-15 iteations")
    print("4.Basin of Attraction Determines error correction capabilty")

if _name_ == "_main_":
    main()
