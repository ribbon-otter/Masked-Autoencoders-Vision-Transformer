A recreation of Masked Autoencoders Are Scalable Vision Learners (Kaiming He, Xinlei Chen) [arvix link](https://arxiv.org/pdf/2111.06377) with two major differences.

1. tokens are gathered both densely and with dilated convolutions. This means each token knows about its surrounding context at a reduced resolution. This does mean that when tokens are removed during the training process, not all direct pixel data is removed. Additionally, because of the use of padding, the dilated convolutions potentially encode some positions data as well. 
2. The dataset is CelebA rather than the datasets that Kaiming He et all used. 

All errors are my own, rather than the paper authors.

The code originally was based on a this tutorial by [Building a Vision Transformer from scratch](https://www.geeksforgeeks.org/deep-learning/building-a-vision-transformer-from-scratch-in-pytorch/) but since evolved significantly, but some lines of code remain identical. 
Ignoring the tutorial code copyright questions, this respository is licensed under the Unlicense.

```
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <https://unlicense.org>
```
