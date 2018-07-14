Pneumonia Diagnosis using XRays from Kaggle Data Sets
===============
<h3 id="Introduction"> Data Introduction </h3>

The data is from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)( The original data sets is from [here](https://data.mendeley.com/datasets/rscbjbr9sj/2) ). But here we just use kaggle's data to analysis.

There are two category in kaggle's data sets : Normal and PNEUMONIA (PEUMONIA can also split into virus and bacteria, but currently we just consider Normal & PNEUMONIA to study)
![image](https://github.com/fr407041/Pneumonia-Diagnosis-using-XRays/blob/master/image/2category.png)

<h3> Data explore </h3>
The kaggle's data have split data set into 3 folders : train、val and test.
<br>The train folder totally have 5216 jpg files (Normal:1341，PNEUMONIA:3875).
<br>The val folder totally have 16 jpg files (Normal:8，PNEUMONIA:8).
<br>The test folder totally have 624 jpg files (Normal:234，PNEUMONIA:390).

**Remark\! The train folder is an imbalance data sets for Normal & PNEUMONIA (about 1:3)**
<h3> Data Augmentation </h3>
Here we just use keras's function image_data_generator. Below is my generator R code

```
image_data_generator(
  rescale            = 1/255    ,
  rotation_range     = 5        ,
  width_shift_range  = 0.1      ,
  height_shift_range = 0.05     ,
  shear_range        = 0.1      ,
  zoom_range         = 0.15     ,
  horizontal_flip    = TRUE     ,
  vertical_flip      = FALSE    ,
  fill_mode          = "reflect"
)
```

<h3> Model Build </h3>
I use xception model with transfer learning.

```
conv_base      <- application_xception(
                                       weights     = "imagenet"    ,
                                       include_top = FALSE         ,
                                       input_shape = c(299, 299, 3)
                                      )
unfreeze_weights(conv_base, from = "block3_sepconv1_act")   

input_tensor   <- layer_input(shape = list(299, 299, 3), name = "input_tensor")
output_tensor  <- input_tensor %>%
                  conv_base %>% 
                  layer_global_average_pooling_2d() %>%
                  layer_dense(units = 1024, activation = "relu", name='fc1') %>% 
                  layer_dropout(rate = 0.3, name='dropout1') %>%
                  layer_dense(units = 512, activation = "relu", name='fc2') %>% 
                  layer_dropout(rate = 0.3, name='dropout2') %>%
                  layer_dense(units = 2, activation = "softmax", name='fc3')
model          <- keras_model(input_tensor, output_tensor)
```

<h3> Model Fit </h3>

```
model %>% compile(
  loss      = "binary_crossentropy"        ,
  optimizer = optimizer_rmsprop(lr = 1e-5) ,
  metrics   = c("accuracy")
)

training_step_size          <- ceiling(length(list.files(train_dir     , recursive = T)) / training_batch_size  )
validation_step_size        <- ceiling(length(list.files(validation_dir, recursive = T)) / validation_batch_size)
weight_adjustment           <- length(list.files(paste(train_dir, '/NORMAL/'   , sep = ""), recursive = T)) / 
                               length(list.files(paste(train_dir, '/PNEUMONIA/', sep = ""), recursive = T))
history <- model %>% fit_generator(
  train_generator                                      ,
  steps_per_epoch  = training_step_size                ,
  class_weight     = list("0"=1,"1"=weight_adjustment) ,
  epochs           = 30                                ,
  validation_data  = validation_generator              ,
  validation_steps = validation_step_size
)
```
Below is my training progress and it is stable for validation data sets from 7th epoch (Good! it is 100% accuracy for validation).

![Traing_Progress](https://github.com/fr407041/Pneumonia-Diagnosis-using-XRays/blob/master/image/training%20Progress.png)

<h3> Classified Result </h3>
Below is my test data set classified result. 
<br>Acquired precision(Positive predictive value/Precision/準確性) is 92.57%.
<br>Recall(True positive rateSensitivity/Sensitivity/靈敏性) is 95.90%.
<br>Specificity(True negative rate/特異性) is 87.18%.

Above result is enough to compared with [Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning](https://www.cell.com/cell/abstract/S0092-8674(18)30154-5) (Precision 92.8%, Recall 93.2%, Specificity 90.1%)

![Traing_Progress](https://github.com/fr407041/Pneumonia-Diagnosis-using-XRays/blob/master/image/test_classified_result.png)

<h3> Conclusion and Future Work </h3>
In this example, the valiadation is only 16 jpgs which is not enough to fine tune our model's hyperparameters. But it's a simple example to learning deep learning with R. I hope my code will help someone learn keras in R. 
<br> There are two things I want to try :

1. For PNEUMONIA, there are still two category which can classified. Trying to classified them and make summary.

2. Study [another data set from kaggle](https://www.kaggle.com/nih-chest-xrays/data) 

Finally We are all standing on the shoulders of giants.

<h3> Reference </h3>
1. https://www.kaggle.com/tentotheminus9/normal-vs-pneumonia-keras-in-r

2. https://www.kaggle.com/aakashnain/beating-everything-with-depthwise-convolution
