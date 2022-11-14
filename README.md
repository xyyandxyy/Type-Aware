# Introduction

This is a modification of [Learning Type-Aware Embeddings for Fashion Compatibility](https://arxiv.org/abs/1803.09196) based on the [code](https://github.com/mvasil/fashion-compatibility) provided by the author. 

Their work is illuminating and the code is detailed, which is very helpful when someone wants to be familiar with or learn Fashion Compatibility tasks.

However, we realized that the old code is inconvenient for us to run on PyTorch (1.7.0 or later) and CUDA (10.1 or later) we use now because it relies on PyTorch 0.1.12.

Therefore we made some modifications to make the code more compatible with our environment settings.

If you want to learn the details of Type-Aware or train it from scratch by yourself, hope this could be helpful to you! :)

***

这份代码是我们在学习Type-Aware时, 为了能让代码在我们目前版本的环境, PyTorch (1.7.0 or later) and CUDA (10.1 or later), 上更方便地运行, 对原作者的代码修改的.

如果你想对Type-Aware这篇工作的细节进行进一步的学习, 或参考地修改它的代码, 希望我们自己修改的代码对你有帮助! 此外代码中包含了我们当时为了方便理解代码的一些中文注释, 我们留下了他们希望也能有帮助.

# Dependencies

* `python3.6+`
* `torch==1.7.0+`
* `torchvision==0.8.0+`
* `tqdm`

# Usage

1. Download the Polyvore Outfits dataset from [here](https://drive.google.com/file/d/13-J4fAPZahauaGycw3j_YvbAHO7tOTW5/view?usp=sharing).
2. Set the parameters as you like in `args.py`
3. run the `main.py` 

