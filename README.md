# Neural Network Optimization for Edge Devices

> Four techniques for shrinking deep networks to fit constrained hardware — pruning, quantization, knowledge distillation, neural architecture search, and binarized networks — each worked through in its own notebook.

Edge deployment is a budget problem: a model that runs comfortably on a GPU has
to fit in a few megabytes and a fraction of the compute. This repo works through
the main families of answers, one notebook each, with the accompanying report in
[`Neural_Network_Optimization_For_Edge_Device.pdf`](Neural_Network_Optimization_For_Edge_Device.pdf).

## The notebooks

| Notebook | Technique | Stack |
|---|---|---|
| [`Pruning.ipynb`](Pruning.ipynb) | **Pruning + quantization** — remove low-magnitude weights, then compress what remains to lower precision | PyTorch, CIFAR-10, ONNX Runtime dynamic quantization |
| [`knowledge_dislltiled_model.ipynb`](knowledge_dislltiled_model.ipynb) | **Knowledge distillation** from a ResNet18 teacher into a smaller student | PyTorch, torchvision |
| [`NAS1_PR.ipynb`](NAS1_PR.ipynb) | **Neural architecture search** — search over layer configurations | Keras, MNIST, TensorBoard |
| [`NAS2_PR.ipynb`](NAS2_PR.ipynb) | **NAS with a REINFORCE controller** — policy-gradient architecture sampling | PyTorch, `Categorical`, TensorBoard |
| [`Binarized_neural_network.ipynb`](Binarized_neural_network.ipynb) | **Binarized networks** — weights and activations constrained to ±1 | NumPy |

## The techniques

**Pruning** removes weights that contribute least, usually by magnitude,
producing a sparse network. It shrinks the parameter count but needs sparse
kernels or structured pruning to actually speed anything up.

**Quantization** reduces numerical precision — float32 to int8 — cutting model
size roughly 4× and letting integer hardware do the work. The notebook uses
ONNX Runtime's dynamic quantization, which quantizes weights ahead of time and
activations on the fly, so it needs no calibration dataset.

**Knowledge distillation** trains a small student to match a large teacher's
soft output distribution rather than just the hard labels. The teacher's
relative confidence across wrong classes carries information that one-hot labels
throw away, which is why a distilled student usually beats the same
architecture trained from scratch.

**Neural architecture search** automates the design itself. `NAS2_PR.ipynb`
frames it as reinforcement learning: a controller samples an architecture,
the architecture gets trained and scored, and the resulting reward updates the
controller by policy gradient.

**Binarized neural networks** take quantization to its limit — ±1 weights and
activations turn multiply-accumulate into XNOR and popcount, which is
dramatically cheaper on hardware without an FPU. The accuracy cost is
correspondingly steep.

## Running them

Each notebook is self-contained. Open in Jupyter or Colab and run top to bottom.

```bash
pip install torch torchvision tensorflow keras onnxruntime numpy matplotlib opencv-python
```

Note that `Pruning.ipynb` uses `google.colab.patches.cv2_imshow`, so it expects
Colab — swap in `matplotlib` or `cv2.imshow` to run it locally.

## Status

These are working notebooks from a study of the field, not a packaged library.
`Binarized_neural_network.ipynb` is a single cell and the least developed of the
five. The notebooks carry little markdown narration — the PDF report is where
the written analysis lives.
