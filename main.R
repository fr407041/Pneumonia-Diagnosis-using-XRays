library(keras)
library(pROC)
library(caret)
library(e1071)

train_dir             <- 'train/'
validation_dir        <- 'val/'
test_dir              <- 'test/'
shape_size            <- 299
training_batch_size   <- 32
validation_batch_size <- 32

# step 1 define generator 
train_datagen <- image_data_generator(
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

validation_test_datagen <- image_data_generator(rescale = 1/255)  

train_generator <- flow_images_from_directory(
  train_dir                               ,                            
  train_datagen                           ,                       
  classes = c('NORMAL', 'PNEUMONIA')      ,
  target_size = c(shape_size, shape_size) ,            
  batch_size  = training_batch_size       ,
  class_mode  = "categorical"             ,
  shuffle     = T                         ,
  seed        = 5566
)

validation_generator <- flow_images_from_directory(
  validation_dir                          ,
  validation_test_datagen                 ,
  classes     = c('NORMAL', 'PNEUMONIA')  ,
  target_size = c(shape_size, shape_size) ,
  batch_size  = validation_batch_size     ,
  class_mode  = "categorical"             ,
  shuffle     = T                         ,
  seed        = 5566
)

test_generator <- flow_images_from_directory(
  test_dir                                ,
  validation_test_datagen                 ,
  classes     = c('NORMAL', 'PNEUMONIA')  ,
  target_size = c(shape_size, shape_size) ,
  batch_size  = 1                         ,
  class_mode  = "categorical"             ,
  shuffle     = FALSE
)

# step 2 define model
input_tensor   <- layer_input(shape = list(shape_size, shape_size, 3), name = "input_tensor")
conv_base      <- application_xception(
                                       weights     = "imagenet" ,
                                       include_top = FALSE      ,
                                       input_shape = c(shape_size, shape_size, 3)
                                      )
output_tensor  <- input_tensor %>%
                  conv_base %>% 
                  layer_global_average_pooling_2d() %>%
                  layer_dense(units = 1024, activation = "relu", name='fc1') %>% 
                  layer_dropout(rate = 0.3, name='dropout1') %>%
                  layer_dense(units = 512, activation = "relu", name='fc2') %>% 
                  layer_dropout(rate = 0.3, name='dropout2') %>%
                  layer_dense(units = 2, activation = "softmax", name='fc3')
model          <- keras_model(input_tensor, output_tensor)
unfreeze_weights(conv_base, from = "block4_sepconv1_act")


# step 3 model fit
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
