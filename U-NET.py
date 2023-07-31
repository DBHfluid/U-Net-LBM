import tensorflow as tf # 
import os # 
import random # 
import numpy as np # 
 
from tqdm import tqdm  # 

from skimage.io import imread, imshow # 
from skimage.transform import resize # 
import matplotlib.pyplot as plt # 





IMG_WIDTH    = 64 # 
IMG_HEIGHT   = 32 # 
IMG_CHANNELS = 1 # gray scale

TRAIN_PATH = 'stage1_train/' #
TEST_PATH = 'stage1_test/' # 
train_ids = next(os.walk(TRAIN_PATH))[1] #  get train folder numbers
test_ids = next(os.walk(TEST_PATH))[1]   #  get test  folder numbers

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS), dtype=np.uint8) # 4D array for trainging datas with data types
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS), dtype=np.uint8) # 4D array for masked    datas with data types
img_     = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS), dtype=np.uint8)
mask_    = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS), dtype=np.uint8)

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)): # 
    path1 = TRAIN_PATH + id_ # 
    img_ = imread(path1 + '/images/' + id_ + '.png') 
    img_ = resize(img_, (IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS), mode='constant', preserve_range=True) # 
    X_train[n] = img_ # 
    
   
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)): # 
    path1 = TRAIN_PATH + id_ # 
    mask_ = imread(path1 + '/maskss/' + id_ + '.png')
    mask_ = resize(mask_, (IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS), mode='constant', preserve_range=True) # 
    Y_train[n] = mask_ # 
        
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS), dtype=np.uint8) # 4D array for trainging datas with data types
Y_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS), dtype=np.uint8) # 4D array for masked    datas with data types
img__     = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS), dtype=np.uint8)
mask__    = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS), dtype=np.uint8)

for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)): # 
    path1 = TEST_PATH + id_ # 
    img__ = imread(path1 + '/images/' + id_ + '.png') 
    img__ = resize(img__, (IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS), mode='constant', preserve_range=True) # 
    X_test[n] = img__ # 
    
   
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)): # 
    path1 = TEST_PATH + id_ # 
    mask__ = imread(path1 + '/maskss/' + id_ + '.png')
    mask__ = resize(mask__, (IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS), mode='constant', preserve_range=True) # 
    Y_test[n] = mask__ # 
            

       
image_x = random.randint(0, len(train_ids)) # 
imshow(X_train[image_x]) # 
plt.show() # 
imshow(Y_train[image_x]) # 
plt.show() # 




#Build the model
inputs = tf.keras.layers.Input(( IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS)) # 
#s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs) # 

#Contraction path
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Expansive path 
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
 
u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
 
u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1])
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
 
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='relu')(c9) # 
 
model = tf.keras.Model(inputs=[inputs], outputs=[outputs]) # 
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy']) # 
model.summary() # 
idx = random.randint(0, len(X_train)) # 

checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5', verbose=1, save_best_only=True) # 

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')] # 

results = model.fit(X_train, Y_train, validation_split=0.1,batch_size=50,  epochs=100, callbacks=callbacks)


preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9) ], verbose=1) # 
preds_val   = model.predict(X_train[ int(X_train.shape[0]*0.9):], verbose=1) # 
preds_test = model.predict(X_test, verbose=1) #

 
#preds_train_t = (preds_train > 0.1).astype(np.uint8) # 
#preds_val_t   = (preds_val > 0.1).astype(np.uint8) # 
 


# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train))
imshow(X_train[ix])
plt.show()
imshow(Y_train[ix])
plt.show()
imshow(preds_train[ix],cmap='gray')
plt.show()
aaa=X_train[ix]-preds_train[ix]
imshow(aaa,cmap='gray')


imshow(X_test[0])
plt.show()
imshow(Y_test[0])
plt.show()
imshow(preds_test[0],cmap='gray')
plt.show()
#aaa=X_test[1]-preds_test[1]
#imshow(aaa,cmap='gray')





        
        
    