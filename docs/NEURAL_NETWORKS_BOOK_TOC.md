# Neural Networks: Book Table of Contents

## Contents
**Preface** 6

### 1 Introduction 8
1.1 Structure of Neural Networks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9
1.2 Computation in Neural Networks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
1.2.1 Performance vs. Learning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
1.2.2 Errors and generalization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
1.2.3 Example: The Three Object Detector . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
1.2.4 Neural vs. Symbolic Computation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16

### 2 Applications of Neural Networks 19
2.1 Engineering vs. Scientific applications . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
2.2 Engineering uses of neural networks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
2.3 Computational neuroscience . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
2.4 Connectionism . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
2.5 Computational Cognitive Neuroscience . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
2.6 From science to engineering and from engineering to science . . . . . . . . . . . . . . . . . . . 24

### 3 History of Neural Networks 26
3.1 Pre-history . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
3.2 Birth of Neural Networks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
3.3 The Cognitive Revolution . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
3.4 The Age of the Perceptron . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
3.5 The “Dark Ages” . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
3.6 First Resurgence: Backprop and The PDP Group . . . . . . . . . . . . . . . . . . . . . . . . . 34
3.7 Second Decline and Second Resurgence: Convolutional Networks . . . . . . . . . . . . . . . . 35
3.8 The Age of Generative AI . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36

### 4 Basic Neuroscience 37
4.1 Neurons and synapses . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37
4.1.1 Neurons . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37
4.1.2 Synapses and neural dynamics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38
4.1.3 Neuromodulators . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41
4.2 The Brain and its Neural Networks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41
4.2.1 Cortex . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 42
4.2.2 The Occipital Lobe . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 43
4.2.3 The Parietal and Temporal Lobes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45
4.2.4 The Frontal Lobe . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 46
4.2.5 Other Neural Networks in the Brain . . . . . . . . . . . . . . . . . . . . . . . . . . . . 47

### 5 Activation Functions 50
5.1 Weighted Inputs and Activation Functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 50
5.2 Threshold Activation Functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 52
5.3 Linear Activation Functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 53
5.4 Sigmoid Activation Functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 54
5.5 Non-local activation functions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 55
5.6 Exercises . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 57

### 6 Linear Algebra and Neural Networks 60
6.1 Vectors and Vector Spaces . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 60
6.2 Vectors and Vector Spaces in Neural Networks . . . . . . . . . . . . . . . . . . . . . . . . . . 62
6.3 Dimensionality Reduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 64
6.4 The Dot Product . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 65
6.5 Norm and Distance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 67
6.6 Other Vector Comparison Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 68
6.7 Matrices and Weight Matrices . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 72
6.8 Graphical Conventions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 75
6.8.1 Source-target graphical conventions . . . . . . . . . . . . . . . . . . . . . . . . . . . . 75
6.8.2 Target-source graphical conventions . . . . . . . . . . . . . . . . . . . . . . . . . . . . 76
6.9 Matrix Multiplication (Part 1) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 77
6.10 Matrix Multiplication (Applications) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 79
6.11 Matrix Multiplication (Part 2) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 80
6.12 Flow Diagrams . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 81
6.13 Tensors . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 82
6.14 Appendix: Vector Operations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 85
6.15 Appendix: Elementwise (Hadamard) Product . . . . . . . . . . . . . . . . . . . . . . . . . . . 87
6.16 Appendix: Block Matrix Representations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 88
6.17 Exercises . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 88

### 7 Data Science and Learning Basics 90
7.1 Data Science Workflow . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 90
7.2 Datasets . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 91
7.3 Data Wrangling (or Preprocessing) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 92
7.4 Datasets for Neural Networks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 95
7.5 Generalization and Testing Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 96
7.6 Supervised vs. Unsupervised Learning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 98
7.7 Other types of model and learning algorithm . . . . . . . . . . . . . . . . . . . . . . . . . . . 99

### 8 Word Embeddings 101
8.1 Background in Computational Linguistics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 101
8.2 Document embeddings . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 103
8.3 Word embeddings . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
8.3.1 Co-occurrence Based Word Embeddings . . . . . . . . . . . . . . . . . . . . . . . . . . 105
8.3.2 Co-occurrence Matrices . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 106
8.3.3 Neural Network Based Embeddings . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 107
8.3.4 Geometric Properties of Word Embeddings . . . . . . . . . . . . . . . . . . . . . . . . 107
8.3.5 Evaluation of Word Embeddings . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 108
8.4 Workflow: Creating Word Embeddings . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 109
8.4.1 Sentence segmentation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 109
8.4.2 Word tokenization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 109
8.4.3 Normalization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 110
8.4.4 Create the word embeddings . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 110
8.4.5 Using a word embedding to make a document embedding . . . . . . . . . . . . . . . . 111

### 9 Unsupervised Learning 112
9.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 112
9.2 Hebbian Learning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 112
9.3 Hebbian Pattern Association for Feed-Forward Networks . . . . . . . . . . . . . . . . . . . . . 114
9.4 Oja’s Rule and Dimensionality Reduction Networks . . . . . . . . . . . . . . . . . . . . . . . . 116
9.5 Competitive learning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 117
9.5.1 Simple Competitive Networks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 117
9.5.2 Self Organizing Maps . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 120

### 10 Dynamical Systems Theory 123
10.1 Dynamical Systems Theory . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 124
10.2 Parameters and State Variables . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 128
10.3 Classification of orbits . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 128
10.3.1 The Shapes of Orbits . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 129
10.3.2 Attractors and Repellers . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 131
10.3.3 Combining these classifications . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 132

### 11 Unsupervised Learning in Recurrent Networks 133
11.1 Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 133
11.2 Hebbian Pattern Association for Recurrent Networks . . . . . . . . . . . . . . . . . . . . . . . 133
11.3 Some features of recurrent auto-associators . . . . . . . . . . . . . . . . . . . . . . . . . . . . 134
11.4 Hopfield Networks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 136

### 12 Supervised Learning 138
12.1 Labeled datasets . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 138
12.2 Supervised Learning: A First Intuitive Pass . . . . . . . . . . . . . . . . . . . . . . . . . . . . 139
12.3 Classification and Regression . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 140
12.4 Visualizing Classification as Partitioning an Input Region into Decision Regions . . . . . . . . 142
12.5 Visualizing Regression as Fitting a Surface to a Cloud of Points . . . . . . . . . . . . . . . . . 143
12.6 Error . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 144
12.7 Error Surfaces and Gradient Descent . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 146
12.8 Expansion of these methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 148
12.9 SSE Exercises . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 149

### 13 Least Mean Squares and Backprop 150
13.1 Least Mean Squares Rule . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 150
13.2 LMS Example . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 151
13.3 Linearly Separable and Inseparable Problems . . . . . . . . . . . . . . . . . . . . . . . . . . . 153
13.4 Backprop . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 155
13.5 XOR and Internal Representations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 156
13.6 LMS Exercises . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 158

### 14 Convolutional Neural Networks 159
14.1 Convolutional Layers . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 159
14.2 Applying a Filter to a Volume . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 161
14.3 Filter Banks (Representational Width) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 163
14.4 Multiple Convolutional Layers (Representational Depth) . . . . . . . . . . . . . . . . . . . . . 164
14.4.1 Pooling . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 165
14.4.2 Flattening and Dense Layers . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 166
14.5 Applications of Convolutional Networks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 166
14.6 Exercises . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 167

### 15 Internal Representations in Neural Networks 170
15.1 Internal Representations in Neural Networks . . . . . . . . . . . . . . . . . . . . . . . . . . . 170
15.2 Net Talk . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 172
15.3 Elman’s Prediction Networks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 173
15.4 Deep Vision Networks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 174
15.5 Other Examples . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 175

### 16 Supervised Recurrent Networks 176
16.1 Types of Supervised Recurrent Networks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 177
16.2 Simple Recurrent Networks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 178
16.3 Backpropagation Through Time . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 180
16.4 Recurrent Networks and Language Generation . . . . . . . . . . . . . . . . . . . . . . . . . . 181
16.5 Limitations of Supervised Recurrent Networks . . . . . . . . . . . . . . . . . . . . . . . . . . . 182

### 17 Transformer Architectures and LLMs 183
17.1 In-Context Processing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 184
17.2 Learning to Speak Internetese . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 184
17.3 Training Using Next-Word Prediction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 185
17.4 How Conversations are Generated Using Next-Token Predictions . . . . . . . . . . . . . . . . 187
17.4.1 The Recursive Trick . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 187
17.4.2 Softmax Outputs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 188
17.4.3 Parameters and Hyperparameters . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 189
17.5 The Transformer Architecture . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 190
17.5.1 Transformer Blocks and the Residual Stream . . . . . . . . . . . . . . . . . . . . . . . 190
17.5.2 Representational Depth and Width Revisited . . . . . . . . . . . . . . . . . . . . . . . 192
17.6 LLMs and the Cognitive Sciences . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 194
17.6.1 Stochastic Parrot or Genuine Intelligence? . . . . . . . . . . . . . . . . . . . . . . . . . 195
17.6.2 LLMs and Behavioral Sciences . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 196
17.6.3 LLMs and Neuroscience . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 199
17.6.4 LLMs and Philosophy . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 200
17.7 LLMs in Practice . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 202
17.7.1 Pre-Training, Post-Training, and Fine-Tuning . . . . . . . . . . . . . . . . . . . . . . . 202
17.7.2 Prompt Engineering . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 203
17.7.3 Agentic Systems . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 203

### 18 Mechanistic Interpretability 205
18.1 Historical Context . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 206
18.2 Working Hypotheses of Mechanistic Interpretability . . . . . . . . . . . . . . . . . . . . . . . 207
18.3 The Toolbox of Mechanistic Interpretability . . . . . . . . . . . . . . . . . . . . . . . . . . . . 208
18.3.1 Linear Probes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 208
18.3.2 Sparse Autoencoders . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 209
18.3.3 Activation Addition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 210
18.3.4 Ablations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 210
18.4 Major Results in Mechanistic Interpretability . . . . . . . . . . . . . . . . . . . . . . . . . . . 211
18.4.1 Toy Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 211
18.4.2 Induction Heads . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 211

### 19 Spiking Models: Neurons & Synapses 213
19.1 Level of abstraction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 214
19.2 Background: The Action Potential . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 216
19.3 Integrate and Fire Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 216
19.3.1 The Heaviside step function . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 217
19.3.2 Linear Integrate and Fire . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 217
19.4 Synapses with Spiking Neurons . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 217
19.4.1 Spike Responses . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 217
19.5 Long-term plasticity . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 218
19.5.1 Spike-Timing Dependent Plasticity (STDP) . . . . . . . . . . . . . . . . . . . . . . . . 218
19.5.2 STDP . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 219
