## Readme 

* this code is based on the StakGAN [Pytorch implementation](https://github.com/hanzhanggit/StackGAN-v2)https://github.com/taoxugit/AttnGAN)

### Data

* please viist [CUHK_PEDES]( http://xiaotong.me/static/projects/person-search-language/dataset.html) and download images of CUHK-PEDES (You may need to contact with the author to acqurie the dataset).
* The 'reid_raw.json' is from https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description. This file can be used to split the train/val/test.
* we use BERT embedding for the text embedding, you can also use char-CNN-RNN to get text embedding.[reedscot/icml2016](https://github.com/reedscot/icml2016)

### Train

* some pathes to dataset  and model in the code should redirection to your own path 
* you can change some hyper-parameter in the cuhk_3stages.yml

``` python
   python train_worker.py
   
```
*  To use multiple GPUs, simply set device='0,1,..' as a set of gpu ids.

### Tips

the vanilla stackgan is hard to  generate the pedestrians images, so we use the spectual norm and [mode seeking](https://github.com/HelenMao/MSGAN) to alleviate the model collapse. 

### Acknowledgements

 [mode seeking](https://github.com/HelenMao/MSGAN) 

