# NLP-Systems
Various Deep Learning NLP models and corresponding codes

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
- run_algorithmic_tasks.py 
  * runs the Neural GPU model on a variety of algorithmic tasks. The tasks available are seen in Dataset/tasks.py
- run_small_lang.py 
  * analyzes the WT2 dataset or PTB dataset, using the any Transformer-model or RNN model.  
  * datasets and preprocessing code (Dataset/WT2/preprocess_WT2.py and Dataset/PTB/preprocess_PTB.py) must be run beforehand.

Requirements:
- numpy (https://github.com/numpy/numpy)
- sentencepiece (https://github.com/google/sentencepiece)
- tensorflow (https://github.com/tensorflow/tensorflow) (ver >= 1.14. Not tested on ver 2.0)
