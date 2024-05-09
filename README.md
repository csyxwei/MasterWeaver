# MasterWeaver
MasterWeaver: Taming Editability and Identity for Personalized Text-to-Image Generation


[![arXiv](https://img.shields.io/badge/arXiv-2405.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2404.xxxxx)
[![Project](https://img.shields.io/badge/Project-Website-orange)](https://masterweaver.github.io/)
![visitors](https://visitor-badge.laobi.icu/badge?page_id=csyxwei/MasterWeaver)

![method](assets/teaser.jpg)

With one single reference image, our MasterWeaver can generate photo-realistic personalized images with diverse clothing, accessories, facial attributes and actions in various contexts.

## Method

--- 

![method](assets/method.jpg)

**(a) Training pipeline of our MasterWeaver.**  To improve the editability while maintaining identity fidelity, we propose an editing direction loss for training. Additionally, we construct a face-augmented dataset to facilitate disentangled identity learning, further improving editability. **(b) Framework of our MasterWeaver.** It adopts an encoder to extract identity features and employ it with text to steer personalized image generation through cross attention.
  
![method](assets/editing_loss.jpg)

By inputting paired text prompts that denote an editing operation, e.g., (a photo of a woman, a photo of a smiling woman), we identify the editing direction in the feature space of diffusion model. Then we align the editing direction of our MasterWeaver with that of original T2I model to improve the text controllability without affecting the identity.


## The paper and code will be released soon.