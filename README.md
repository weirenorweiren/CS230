Name Nationality Identification with 
==============================================================
LSTM + Character Level Embedding + Attention Mechanism
==============================================================
This is the project for Stanford CS230 at spring quarter of 2020. The project is developed based on Jinhyuk Lee et al's work of [Name Nationaltiy Classification with Recurrent Neural Networks]. Our model adds an additional quadgram embedding and the attention mechanism on LSTM to extract more features in order to improve the performance. The dataset is from https://github.com/d4em0n/nationality-classify which contains (name,nationality) pairs from 18 countries. Due to sudden time-cutting for the project, we don't have enough time to fine-tune the model. But the good news is that the two figures below give an encouraging experiment results. We can see that for an attention length of 5 and a batch size of 1024, the learning rate within 0.001 and 0.003 shows very exciting and promising accuracy for test set. 

<p float="left">
<img src="https://github.com/weirenorweiren/CS230/blob/master/Own%20dataset/Tuning/attn_length%3D5/accu1.jpg" alt="alt text" width="450">

<img src="https://github.com/weirenorweiren/CS230/blob/master/Own%20dataset/Tuning/attn_length%3D5/accu5.jpg" alt="alt text" width="450">
</p>

## Requirements
* Python 3
* Tensorflow 1.0.1 (GPU enabled)
* Gensim

## New (added) directories 
### (Please check https://github.com/63coldnoodle/ethnicity-tensorflow for origin directories)
* main_ad.py : Running our advanced model with adjustable hyperparameters
* model_ad.py : Model structure for the advanced model
* ops_ad.py : Tensorflow ops used in model_ad.py
* dataset_ad.py : Dataset reading and experiment workflow
* preprocess_own.py : Preprocessing 18-countries dataset for the advanced model
* preprocess_ad.py : Preprocessing crawled dataset in Jinhyuk Lee et al's work for advanced model
* preprocess_test.py : Preprocessing 18-countries dataset to fit in Jinhyuk Lee et al's model
* data/own/ : Preprocessed data from preprocess_own.py
* data/name_nationality/ : 18-countries dataset
* data/ad/ : Preprocessed data from preprocess_ad.py
* data/test/ : Preprocessed data from preprocess_test.py
* data/test with $/ : Preprocessed data from preprocess_test.py (cleaned verison as mentioned in Jinhyuk Lee et al's work)
* Own dataset/ : Experiment results of preprocessed data from preprocess_own.py and preprocess_test.py
* RNN dataset/ : Experiment results of preprocessed data from preprocess_ad.py

## Run the code
```bash
$ python main_ad.py
```
Or you can try the ipynb file in 'Own dataset' or 'RNN dataset' for convenience. 

## Contact 
- Wei Ren (weiren@stanford.edu)
- ZhenHuan Hong (jhong812@stanford.edu)
- Jiate Li (jiateli@stanford.edu)

## Acknowledgement 
We thank for Jinhyuk Lee's response to help the development of the project. We also thank the teaching team of CS230 for your support and patience. 
