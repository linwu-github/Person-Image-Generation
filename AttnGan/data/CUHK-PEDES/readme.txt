## Cite

@article{li2017person,
  title={Person search with natural language description},
  author={Li, Shuang and Xiao, Tong and Li, Hongsheng and Zhou, Bolei and Yue, Dayu and Wang, Xiaogang},
  journal={arXiv preprint arXiv:1702.05729},
  year={2017}
}


## Terms of Use

By downloading the dataset, you agree to the following terms:

1.  You will use the data only for non-commercial research and educational purposes.
2.  You will **NOT** distribute the dataset.
3.  The Chinese University of Hong Kong makes no representations or warranties regarding the data. All rights of the images reserved by the original owners.
4.  You accept full responsibility for your use of the data and shall defend and indemnify The Chinese University of Hong Kong, including their employees, officers and agents, against any and all claims arising from your use of the data, including but not limited to your use of any copies of copyrighted images that you may create from the data.


## Overall Information

The dataset contains 40,206 images of 13,003 persons from five existing person re-identification datasets, CUHK03, Market-1501, CUHK-SYSU (test_query & train_query), VIPER, and CUHK01. Since persons in Market-1501 and CUHK03 have many similar samples, to balance the number of persons from different domains, we randomly selected four images for each person in the two datasets. All the image were labeled by crowd workers from Amazon Mechanical Turk (AMT), where each image was annotated with two sentence descriptions and a total of 80,412 sentences were collected.

The dataset incorporates rich details about person appearances, actions, poses and interactions with other objects. The sentence descriptions are generally long (> 23 words in average), and has abundant vocabulary and little repetitive information. There are a total of 1,893,118 words and 9,408 unique words in our dataset. The longest sentence has 96 words and the average word length is 23.5.


## Introduction

'imgs':              Images collected from five existing person re-identification datasets.

'caption_all.json':  Annotations of 40,206 person images. The format of annotations is shown in the Example.
					 For each image, there are three terms of annotations:
					 "captions": two natural language descriptions 
					 "id": person ID of the image. There are 13,003 persons, so the "id" ranges from 1 to 13,003.
					 "file_path": The save path of the image


*Example
{
    "id": 1,
    "file_path": "test_query/p10376_s14337.jpg",
    "captions": [
      "She wears a purple long sleeved, ankle length dress. There is a pattern on the dress.",
      "This woman is heavy set. She is facing to the left of the camera. She is wearing an ankle length dress and is carrying something in her hand. She has her head down."
    ]
}