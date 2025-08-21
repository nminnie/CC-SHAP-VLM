# Measuring the degree of multimodality in VLMs across languages

This code builds upon the MM-SHAP implementation for the paper "Do Vision &amp; Language Decoders use Images and Text equally? How Self-consistent are their Explanations?"

In addition, we incorporate the Perceptual Score method from the paper "Perceptual Score: What Data Modalities Does Your Model Perceive?"

The implementation is extended to the multilingual setting, employing two recent VLMs (LLaVa-OneVision and Pangea), as well as 5 question types and 8 languages in the xGQA benchmark. No further processing is applied to the images and texts in this dataset.

## Installation and running
1. `conda create -n <env-name> python=3.12.1`
2. `pip install -r requirements.txt`
3. Specify the model name either as a local path or Huggingface path
4. Download the xGQA images and questions from their respective repositories and change the paths in `config.py` accordingly. Data repositories:
  * GQA images: https://cs.stanford.edu/people/dorarad/gqa/download.html 
  * xGQA multilingual questions: https://github.com/adapter-hub/xGQA
5. Run `run-xgqa.py` as follows with the arguments: `python run-xgqa.py xgqa [model_name] [num_samples] [save_json] [data_root] [lang] [question_type]`.
For example, `python run-xgqa.py xgqa llava_onevision 20 1 data/ zh query`

### Supported models
1. Pangea
1. LLaVA-OneVision
1. BakLLaVA
1. LLaVA-NeXT-Vicuna
1. LLaVA-NeXT-Mistral

## Credits
The Shapley value implementation in the `shap` folder is a modified version of https://github.com/slundberg/shap .

## Cite the original methods and dataset
```bibtex
@inproceedings{parcalabescu2025do,
    title={Do Vision \& Language Decoders use Images and Text equally? How Self-consistent are their Explanations?},
    author={Letitia Parcalabescu and Anette Frank},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=lCasyP21Bf}
}
```

```bibtex
@article{gat2021perceptual,
  title={Perceptual score: What data modalities does your model perceive?},
  author={Gat, Itai and Schwartz, Idan and Schwing, Alex},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={21630--21643},
  year={2021}
}
```

```bibtex
@article{pfeiffer2021xgqa,
  title={xGQA: Cross-lingual visual question answering},
  author={Pfeiffer, Jonas and Geigle, Gregor and Kamath, Aishwarya and Steitz, Jan-Martin O and Roth, Stefan and Vuli{\'c}, Ivan and Gurevych, Iryna},
  journal={arXiv preprint arXiv:2109.06082},
  year={2021}
}
```
