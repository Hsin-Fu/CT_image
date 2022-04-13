# CT_image
# Load data and deal data

```py
all_images_list = glob(os.path.join('tiff_images','*.tif'))

check_contrast = re.compile(r'ID_([\d]+)_AGE_[\d]+_CONTRAST_([\d]+)_CT')
label = []
id_list = []
for image in all_images_list:
    id_list.append(check_contrast.findall(image)[0][0])
    label.append(check_contrast.findall(image)[0][1])
    
label_list = pd.DataFrame(label,id_list)
images = np.stack([jimread(i) for i in all_images_list],0)

```

# Split Data and reshape

```py
X_train, X_test, y_train, y_test = train_test_split(images, label_list, test_size=0.1, random_state=0)

n_train, depth, width, height = X_train.shape
n_test,_,_,_ = X_test.shape

input_shape = (width,height,depth)

input_train = X_train.reshape((n_train, width,height,depth))
input_train.shape
input_train.astype('float32')
input_train = input_train / np.max(input_train)

input_test = X_test.reshape(n_test, *input_shape)
input_test.astype('float32')
input_test = input_test / np.max(input_test)

output_train = keras.utils.np_utils.to_categorical(y_train, 2)
output_test = keras.utils.np_utils.to_categorical(y_test, 2)
```

# Model
```py
model = Sequential()

model.add(Conv2D(50, (5, 5), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(3, 3))) 

model.add(Conv2D(30, (4, 4), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2))) 

model.add(Flatten()) 
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])
          
model.fit(input_train, output_train,
            batch_size=20,
            epochs=20,
            verbose=1,
            validation_data=(input_test, output_test))
```
# Example
```py
tmp = random.randint(0, len(input_test)-1  )  
plt.imshow(X_test[tmp][0])

print('predict result')
if predicted_val[tmp] == 0:
    print('Contrast')
else:
    print('Not Contrast')

print('----------------------------')
print('acturally result')
if y_test[0][tmp] == 1:
    print('Contrast')
else:
    print('Not Contrast')
```
predict result :Not Contrast

acturally result :Not Contrast


![output](https://user-images.githubusercontent.com/103483905/163112705-1eda16c8-bdd3-41a8-8b05-ec8f8d6b4527.png)

predict result :Not Contrast

acturally result :Not Contrast

![output1](https://user-images.githubusercontent.com/103483905/163113119-ca6f82c0-83ba-4364-8b26-c468bfd76403.png)

