

#importing libraries
import warnings
warnings.filterwarnings('ignore')
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.constraints import min_max_norm
import keras.backend as K
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import LeakyReLU

#metric IoU
def calculate_iou(y_true, y_pred):

    
    results = []
    
    for i in range(0,y_true.shape[0]):
    
        # set the types so we are sure what type we are using
        y_true = y_true.astype(np.float32)
        y_pred = y_pred.astype(np.float32)


        # boxTrue
        x_boxTrue_Bleft = y_true[0,0]  # numpy index selection
        x_boxTrue_tright = y_true[0,1]
        y_boxTrue_Bleft = y_true[0,2]
        y_boxTrue_tright = y_true[0,3]
        boxTrue_width = x_boxTrue_tright - x_boxTrue_Bleft
        boxTrue_height = y_boxTrue_tright - y_boxTrue_Bleft
        area_boxTrue = (boxTrue_width * boxTrue_height)

        # boxPred
        x_boxPred_Bleft = y_pred[0,0]
        x_boxPred_tright = y_pred[0,1]
        y_boxPred_Bleft = y_pred[0,2]
        y_boxPred_tright = y_pred[0,3]
        boxPred_width = x_boxPred_tright - x_boxPred_Bleft
        boxPred_height = y_boxTrue_tright - y_boxPred_Bleft
        area_boxPred = (boxPred_width * boxPred_height)


        x_boxTrue_tleft = x_boxTrue_Bleft
        y_boxTrue_tleft = y_boxTrue_Bleft

        x_boxPred_tleft = x_boxPred_Bleft
        y_boxPred_tleft = y_boxPred_Bleft


        # calculate the bottom right coordinates for boxTrue and boxPred

        # boxTrue
        x_boxTrue_br = x_boxTrue_tleft + boxTrue_width
        y_boxTrue_br = y_boxTrue_tleft + boxTrue_height # Version 2 revision

        # boxPred
        x_boxPred_br = x_boxPred_tleft + boxPred_width
        y_boxPred_br = y_boxPred_tleft + boxPred_height # Version 2 revision


        # calculate the top left and bottom right coordinates for the intersection box, boxInt

        # boxInt - top left coords
        x_boxInt_tleft = np.max([x_boxTrue_tleft,x_boxPred_tleft])
        y_boxInt_tleft = np.max([y_boxTrue_tleft,y_boxPred_tleft]) # Version 2 revision

        # boxInt - bottom right coords
        x_boxInt_br = np.min([x_boxTrue_br,x_boxPred_br])
        y_boxInt_br = np.min([y_boxTrue_br,y_boxPred_br]) 

        # Calculate the area of boxInt, i.e. the area of the intersection 
        # between boxTrue and boxPred.
        # The np.max() function forces the intersection area to 0 if the boxes don't overlap.
        
        
        # Version 2 revision
        area_of_intersection = \
        np.max([0,(x_boxInt_br - x_boxInt_tleft)]) * np.max([0,(y_boxInt_br - y_boxInt_tleft)])

        iou = area_of_intersection / ((area_boxTrue + area_boxPred) - area_of_intersection)


        # This must match the type used in py_func
        iou = iou.astype(np.float32)
        
        # append the result to a list at the end of each loop
        results.append(iou)
    
    # return the mean IoU score for the batch
    return np.mean(results)

def IoU(y_true, y_pred):
    
    # Note: the type float32 is very important. It must be the same type as the output from
    # the python function above or you too may spend many late night hours 
    # trying to debug and almost give up.
    
    iou = tf.py_func(calculate_iou, [y_true, y_pred], tf.float32)

    return iou

def mean_squared_error(y_true, y_pred):
	return K.mean(K.square(y_pred - y_true))

#loading dataframe
data = pd.read_csv(r'training.csv')

#Training only on those images which have legible coordinates
data = data[data['x1']>=0]
data = data[data['x2']>=0]
data = data[data['y1']>=0]
data = data[data['y2']>=0]

data = data[data['x1']<=640]
data = data[data['x2']<=640]
data = data[data['y1']<=480]
data = data[data['y2']<=480]




#data = data.drop_duplicates(subset=['image_name','x1','x2' , 'y1' ,'y2'])
testx = pd.read_csv(r'test.csv')


from keras.preprocessing.image import ImageDataGenerator

#Training Set Generator
train_datagen = ImageDataGenerator(rescale =1./255 , validation_split = 0.20)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_dataframe( dataframe = data,directory = './images',x_col = "image_name" ,y_col =  ['x1' , 'x2' ,'y1' , 'y2'] ,target_size = (64 , 64),class_mode = 'other' , has_ext =True,batch_size=32  )

#CNN with 3 Convolution Layer
classifier = Sequential()
classifier.add(Convolution2D(128, 3 ,3 ,input_shape=(64,64,3) , activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Convolution2D(256 , 3 ,3 ,activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Convolution2D(512 , 3 ,3 ,activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Flattening
classifier.add(Flatten())


#hidden layer
classifier.add(Dense(output_dim = 1024 , activation = 'relu'))
classifier.add(Dense(output_dim = 4 )) 
classifier.compile(optimizer = 'adam', loss = mean_squared_error, metrics = [IoU])

#Validation Generator
validaton_generator = train_datagen.flow_from_dataframe(dataframe = data,directory = './images',x_col = "image_name" ,y_col =  ['x1' , 'x2' ,'y1' , 'y2'] ,target_size = (64 , 64),class_mode = 'other' , has_ext =True,batch_size=32 )

#Test_Set Generator
test_set = test_datagen.flow_from_dataframe(dataframe = testx,directory = './images' , x_col = "image_name",y_col =  None ,has_ext = True ,target_size = (64,64) ,batch_size = 32 ,class_mode = None)
classifier.fit_generator(generator = training_set,samples_per_epoch = 24000, nb_epoch = 1,validation_data = validaton_generator , nb_val_samples =32)

#predicting
solution = classifier.predict_generator(test_set ,752)


#convertig the predicted float values into integer
solution=solution.astype(int)

#Saving the predicted values in csv file
df = pd.DataFrame(solution)
df.to_csv("sol1.csv")
