# Person-Image-Generation
## Readme

### Pytorch implementation for reproducing PIG-MM

* the main backbone is based on HDGan[Pytorch implementation](https://github.com/ypxie/HDGan)

### Dependencies

* Python 3+
* Pytorch 0.3.+
* Anaconda 3.6+

### Data
* please viist [CUHK_PEDES]( http://xiaotong.me/static/projects/person-search-language/dataset.html) and download images of CUHK-PEDES (You may need to contact with the author to acqurie the dataset).
* The 'reid_raw.json' is from https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description. This file can be used to split the train/val/test.
* we use BERT embedding for the text embedding, you can also use char-CNN-RNN to get text embedding.[reedscot/icml2016](https://github.com/reedscot/icml2016)
### pretrain model
* [BaiduYun](https://pan.baidu.com/s/153f0fRoz0bHccZi29SHSVQ?pwd=f4kf)code: f4kf

### Train

* go to train/train_gan:

* some pathes to dataset  and model in the code should redirection to your own path 

```python
python train_worker.py
```
* To use multiple GPUs, simply set device='0,1,..' as a set of gpu ids.

### Test

* go to test:

```python
python test_worker.py
```

### Tips

* we use the first 10k training set due to limited computing resources, but we also do the experiment on the whole training set.
* 3206 is  class_num of 10k training set &&  11003 is class_num of whole training set
* the training on the whole train set (30k) is hard, model collapse may happen sometimes, we suggest you to finetune the parameter, e.g,  id_loss_rate.. or just re-train the model use the same parameter(I have tried, it is **useful**, emm..)


### Acknowledgements

HDGan[Pytorch implementation](https://github.com/ypxie/HDGan)

StakGAN [Pytorch implementation](https://github.com/hanzhanggit/StackGAN-v2)

AttanGan[Pytorch implementation](https://github.com/taoxugit/AttnGAN)

