from gym.envs.registration import register

register(id='QuantumArchSearch-v0',
         entry_point='qas_gym.envs:QuantumArchSearch',
         nondeterministic=True)

register(id='BasicTwoQubit-v0',
         entry_point='qas_gym.envs:BasicTwoQubitEnv',
         nondeterministic=True)

register(id='BasicThreeQubit-v0',
         entry_point='qas_gym.envs:BasicThreeQubitEnv',
         nondeterministic=True)

register(id='BasicNQubit-v0',
         entry_point='qas_gym.envs:BasicNQubitEnv',
         nondeterministic=True)

register(id='NoisyTwoQubit-v0',
         entry_point='qas_gym.envs:NoisyTwoQubitEnv',
         nondeterministic=True)

register(id='NoisyThreeQubit-v0',
         entry_point='qas_gym.envs:NoisyThreeQubitEnv',
         nondeterministic=True)

register(id='NoisyNQubit-v0',
         entry_point='qas_gym.envs:NoisyNQubitEnv',
         nondeterministic=True)
