# Adversarial Illusions in Multi-Modal Embeddings

Paper link:
[https://arxiv.org/abs/2308.11804](https://arxiv.org/abs/2308.11804)

This is preliminary set of notebooks, should be easy to edit and
adapt. I ran everything using 4 12GB GPUs, so would just load each
notebook on a separate card.

**Configs**:
- Install
  [ImageBind](https://github.com/facebookresearch/ImageBind#usage),
  I run notebooks (except `generate_text.ipynb``) directly from the ImageBind repo.
- Install
  [PandaGPT](https://github.com/yxuansu/PandaGPT#2-running-pandagpt-demo-back-to-top),
  no need to run the demo, just get the model weights. I used tensors to save the
  modified audios and images. To enable generation from PyTorch
  tensors replace
  [openllama.py](https://github.com/yxuansu/PandaGPT/blob/main/code/model/openllama.py)
  with contents of `pandagpt_openllama.py` and you will be able
  to load directly from the tensor when your path contains "hack"
  (certainly open to better design suggestions).


Please feel free to email: [eugene@cs.cornell.edu](mailto:eugene@cs.cornell.edu) or raise an issue.
