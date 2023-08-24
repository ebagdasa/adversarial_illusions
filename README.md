<h1 align="center"> <i>Ceci n'est pas une pomme*:</i>   <br>
Adversarial Illusions in Multi-Modal Embeddings </h1>

<p align="center"> <i>Eugene Bagdasaryan and Vitaly Shmatikov</i></p>

Multi-modal encoders map images, sounds, texts, videos, etc. into a
single embedding space, aligning representations across modalities
(e.g., associate an image of a dog with a barking sound). We show that
multi-modal embeddings can be vulnerable to an attack we call
"adversarial illusions." Given an input in any modality, an adversary
can perturb it so as to make its embedding close to that of an
arbitrary, adversary-chosen input in another modality. Illusions thus
enable the adversary to align any image with any text, any text with
any sound, etc. 

Adversarial illusions exploit proximity in the embedding space and are
thus agnostic to downstream tasks. Using ImageBind embeddings, we
demonstrate how adversarially aligned inputs, generated without
knowledge of specific downstream tasks, mislead image generation, text
generation, and zero-shot classification.

Paper link:
[https://arxiv.org/abs/2308.11804](https://arxiv.org/abs/2308.11804)

<img src="illusion.png" alt="drawing" width="600"/>

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
- For zero-shot classification of audios into ImageNet, you'd need
  ImageNet validation dataset to compare embeddings.
- Create a folder for assets, i.e. images, audios, etc. ImageBind has
  `.assets` that you can take some examples from, but you can also use
  your own.

Please feel free to email: [eugene@cs.cornell.edu](mailto:eugene@cs.cornell.edu) or raise an issue.


