# Sock Sort

## Description 

Sort socks into pairs using deep learning image classification and Kafka stream processing

![DeepLearning](docs/sockrecognize-optimize.gif)


## Image Preparation
Prepare a dataset of sock images in ImageRecord format.  Further references [here](https://gluon-cv.mxnet.io/build/examples_datasets/recordio.html) and [here](https://arthurcaillau.com/image-record-iter/)

First, we need to generate a `.lst` file, i.e. a list of these images containing label and filename information.


```
cd 10-image-preparation
```

Setup Python
```
python3 -m venv myenv
source myenv/bin/activate
pip install mxnet opencv-python
```


Remove any existing files
```
rm -f *.lst *.rec *.idx
```

Create dataset of sock images in ImageRecord format
```
python im2rec.py --list  --train-ratio 0.8   --recursive ./sock-images_rec sock-images/
```

Gives 
```
class-confluent 0
class-databricks 1
class-github 2
class-google 3
class-mongo 4
class-streamset 5
```

After the execution, you find files `sock-images_rec_train.lst` and `sock-images_rec_val.lst` generated. 

```
wc -l *.lst
544 sock-images_rec_train.lst
137 sock-images_rec_val.lst
681 total
```  

With this file, the next step is:

```
python im2rec.py   --resize 512   --center-crop   --num-thread 4 ./sock-images_rec ./sock-images/
```


It gives you four more files: (`sock-images_rec_train.idx`, `sock-images_rec_train.rec`, `sock-images_rec_val.idx`, `sock-images_rec_val.rec`). Now, you can use them to train!

```
 aws s3 cp . s3://deeplens-sagemaker-socksort --exclude "*" --include "*.idx"  --include "*.rec"  --include "*.lst" --recursive
```


# Model Training
Now we want to train an image classification model that can classify sock images.  We will use transfer learning mode using AWS Sagemaker.  We can launch a Sagemaker notebook for image classification algorithm in transfer learning mode to fine-tune a pre-trained model (trained on sock images data) to learn to classify a new dataset.  A more extensive explaination  [here](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/imageclassification_caltech/Image-classification-transfer-learning-highlevel.ipynb)

```
cd 20-model-training
```

# Deeplens Lambda Function
Now to the the Deeplens Greengrass Lambda Function.  That is, we need to build, publish and deploy the Sock Sort AWS DeepLens Inference Lambda Function

Steps to build `sock_deeplens_inference_function.zip`

```
cd 30-deeplens-greengrass-lambda
mkdir package_deeplens_inference_function
pip install --target ./package_deeplens_inference_function paho-mqtt
cd package_deeplens_inference_function
wget https://docs.aws.amazon.com/deeplens/latest/dg/samples/deeplens_inference_function_template.zip
unzip deeplens_inference_function_template.zip
rm deeplens_inference_function_template.zip
zip -r9 ${OLDPWD}/sock_deeplens_inference_function.zip .
cd ..
ls  sock_deeplens_inference_function.zip
```

## Publish Sock Sort AWS DeepLens Inference Lambda Function
See [here](https://docs.aws.amazon.com/deeplens/latest/dg/deeplens-inference-lambda-create.html) for a step by step guide 

