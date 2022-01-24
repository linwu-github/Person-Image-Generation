## Read Me

* this code is based on the AttanGan[Pytorch implementation](https://github.com/taoxugit/AttnGAN)

### Data 
* please viist [CUHK_PEDES]( http://xiaotong.me/static/projects/person-search-language/dataset.html) and download images of CUHK-PEDES (You may need to contact with the author to acqurie the dataset).
* The 'reid_raw.json' is from https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description. This file can be used to split the train/val/test.
* we use BERT embedding for the text embedding, you can also use char-CNN-RNN to get text embedding.[reedscot/icml2016](https://github.com/reedscot/icml2016)
* you can train the DAMSM by yourself or use our pretrained model 
```python
python pretrain_DAMSM_CUHK.py


```

### Train

* some pathes to dataset  and model in the code should redirection to your own path 
* you can change some hyper-parameter in the cuhk_attn2.yml

``` python
   python train_worker.py
   
   ```
*  To use multiple GPUs, simply set device='0,1,..' as a set of gpu ids.


