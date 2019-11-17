# NLP-Systems
Various NLP models and corresponding codes

Models Currently Implemented:
- Transformer (arXiv:1706.03762)
- Universal Transformer (arXiv:1807.03819)
- Evolved Transformer (arXiv:1901.11117)
- Recurrent-Augmented Transformer (arXiv:1904.03092)
- Transformer-XL (arXiv:1901.02860)
- RNN
- Seq2Seq (based on the variant described in arXiv:1804.00946)
- Neural GPU (arXiv:1511.08228)

Current Codes Implemented:
- Neural_GPU.py (runs the Neural GPU model on a variety of algorithmic tasks)
- run_WT2.py (analyzes the WT2 dataset or PTB dataset using all Transformer models and RNN.py) (/Dataset/WT2/preprocess_WT2.py must be run before hand)

Requirements:
- numpy
- sentencepiece (https://github.com/google/sentencepiece)
- tensorflow (ver >= 1.14. Not tested on ver 2.0)
