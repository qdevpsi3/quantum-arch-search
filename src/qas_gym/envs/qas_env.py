import sys
from contextlib import closing
from io import StringIO
from typing import Dict, List, Optional, Union

import cirq
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


class QuantumArchSearchEnv(gym.Env):
    metadata = {'render.modes': ['ansi', 'human']}

    def __init__(
        self,
        target: np.ndarray,
        qubits: List[cirq.LineQubit],
        state_observables: List[cirq.GateOperation],
        action_gates: List[cirq.GateOperation],
        fidelity_threshold: float,
        reward_penalty: float,
        max_timesteps: int,
        error_observables: Optional[float] = None,
        error_gates: Optional[float] = None,
    ):
        super(QuantumArchSearchEnv, self).__init__()

        # set parameters
        self.target = target
        self.qubits = qubits
        self.state_observables = state_observables
        self.action_gates = action_gates
        self.fidelity_threshold = fidelity_threshold
        self.reward_penalty = reward_penalty
        self.max_timesteps = max_timesteps
        self.error_observables = error_observables
        self.error_gates = error_gates

        # set environment
        self.target_density = target * np.conj(target).T
        self.simulator = cirq.Simulator()

        # set spaces
        self.observation_space = spaces.Box(low=-1.,
                                            high=1.,
                                            shape=(len(state_observables), ))
        self.action_space = spaces.Discrete(n=len(action_gates))
        self.seed()

    def __str__(self):
        desc = 'QuantumArchSearch-v0('
        desc += '{}={}, '.format('Qubits', len(self.qubits))
        desc += '{}={}, '.format('Target', self.target)
        desc += '{}=[{}], '.format(
            'Gates', ', '.join(gate.__str__() for gate in self.action_gates))
        desc += '{}=[{}])'.format(
            'Observables',
            ', '.join(gate.__str__() for gate in self.state_observables))
        return desc

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.circuit_gates = []
        return self._get_obs()

    def _get_cirq(self, maybe_add_noise=False):
        circuit = cirq.Circuit(cirq.I(qubit) for qubit in self.qubits)
        for gate in self.circuit_gates:
            circuit.append(gate)
            if maybe_add_noise and (self.error_gates is not None):
                noise_gate = cirq.depolarize(
                    self.error_gates).on_each(*gate._qubits)
                circuit.append(noise_gate)
        if maybe_add_noise and (self.error_observables is not None):
            noise_observable = cirq.bit_flip(
                self.error_observables).on_each(*self.qubits)
            circuit.append(noise_observable)
        return circuit

    def _get_obs(self):
        circuit = self._get_cirq(maybe_add_noise=True)
        obs = self.simulator.simulate_expectation_values(
            circuit, observables=self.state_observables)
        return np.array(obs).real

    def _get_fidelity(self):
        circuit = self._get_cirq(maybe_add_noise=True)
        pred = self.simulator.simulate(circuit).final_state_vector
        inner = np.inner(np.conj(pred), self.target)
        fidelity = np.conj(inner) * inner
        return fidelity.real

    def step(self, action):

        # update circuit
        action_gate = self.action_gates[action]
        self.circuit_gates.append(action_gate)

        # compute observation
        observation = self._get_obs()

        # compute fidelity
        fidelity = self._get_fidelity()

        # compute reward
        if fidelity > self.fidelity_threshold:
            reward = fidelity - self.reward_penalty
        else:
            reward = -self.reward_penalty

        # check if terminal
        terminal = (reward > 0.) or (len(self.circuit_gates) >=
                                     self.max_timesteps)

        # return info
        info = {'fidelity': fidelity, 'circuit': self._get_cirq()}

        return observation, reward, terminal, info

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write('\n' + self._get_cirq(False).__str__() + '\n')

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
