@ECHO OFF

:arbitrary_state_prepare

REM (Grover method) Creating superpositions that correspond to efficiently integrable probability
wget -nc https://arxiv.org/pdf/quant-ph/0208112.pdf

REM (Grover method impl.) Systematic Preparation of Arbitrary Probability Distribution with a Quantum Computer
REM https://medium.com/qiskit/systematic-preparation-of-arbitrary-probability-distribution-with-a-quantum-computer-165dfd8fbd7d

REM (Grover method impl.) Preparing a quantum state from a classical probability distribution
REM https://quantumcomputing.stackexchange.com/questions/12104/preparing-a-quantum-state-from-a-classical-probability-distribution

REM (Mottonen method) Transformation of quantum states using uniformly controlled rotations
wget -nc https://arxiv.org/pdf/quant-ph/0407010.pdf

REM (QGAN/VQA method, RY-cyclic(CZ)) Quantum Generative Adversarial Networks for Learning and Loading Random Distributions
wget -nc https://arxiv.org/pdf/1904.00043.pdf

REM Optimal tuning of quantum generative adversarial networks for multivariate distribution loading
REM https://doi.org/10.3390/quantum4010006

REM (RY-adjacent(CNOT)) Approximate amplitude encoding in shallow parameterized quantum circuits and its application to financial market indicator
wget -nc https://arxiv.org/pdf/2103.13211.pdf

REM (mctrl-V circuit) Quantum state preparation protocol for encoding classical data into the amplitudes of a quantum information processing register's wave function
wget -nc https://arxiv.org/pdf/2107.14127.pdf

REM 量子态制备及其在量子机器学习中的前景
wget -nc https://wulixb.iphy.ac.cn/pdf-content/10.7498/aps.70.20210958.pdf

REM (Low-rank approx, nice!!) Low-rank quantum state preparation
wget -nc https://arxiv.org/pdf/2111.03132.pdf

REM (产生单峰分布的线路) Loading Probability Distributions in a Quantum circuit
wget -nc https://arxiv.org/pdf/2208.13372.pdf

REM (RX/RY/RZ-adjacent(CNOT)) Approximate complex amplitude encoding algorithm and its application to data classification problems
wget -nc https://arxiv.org/pdf/2211.13039.pdf

REM Preparing random state for quantum financing with quantum walks
wget -nc https://arxiv.org/pdf/2302.12500.pdf

REM (Benchmark EEAs) Generating probability distributions using variational quantum circuits
wget -nc https://arxiv.org/pdf/2307.09147.pdf

REM (RY-RZ-linear(CNOT)) Warm-Starting the VQE with Approximate Complex Amplitude Encoding
wget -nc https://arxiv.org/pdf/2402.17378.pdf

REM (产生对称分布的线路) Quantum State Preparation for Probability Distributions with Mirror Symmetry Using Matrix Product States
wget -nc https://arxiv.org/pdf/2403.16729.pdf


:expressive_ansatz

REM Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms
wget -nc https://arxiv.org/pdf/1905.10876.pdf

REM An Expressive Ansatz for Low-Depth Quantum Approximate Optimisation
wget -nc https://arxiv.org/pdf/2302.04479.pdf


:QCNN

REM A Quantum Convolutional Neural Network for Image Classification
wget -nc https://arxiv.org/pdf/2107.03630v1.pdf

REM Quantum convolutional neural network for classical data classification
wget -nc https://arxiv.org/pdf/2108.00661.pdf

REM Quantum convolutional neural networks formulti-channel supervised learning
wget -nc https://cqdmqd.yale.edu/sites/default/files/2024-03/QuantumConvolutionalNeuralNetworksForMultiChannelSupervisedLearning_0.pdf


:QDecisionTree

REM Quantum decision tree classifier
REM https://www.researchgate.net/publication/260526004_Quantum_decision_tree_classifier

REM On Quantum Decision Trees
wget -nc https://arxiv.org/pdf/1703.03693.pdf

REM The Quantum Version Of Classification Decision Tree Constructing Algorithm C5.0
wget -nc https://arxiv.org/pdf/1907.06840.pdf

REM Representation of binary classification trees with binary features by quantum circuits
REM https://quantum-journal.org/papers/q-2022-03-30-676/
REM https://github.com/RaoulHeese/qtree


:QSynth

REM Synthesis of Quantum Logic Circuit
wget -nc https://arxiv.org/pdf/quant-ph/0406176.pdf
