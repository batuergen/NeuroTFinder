# Audit of the Legacy Model

This is the audit of the notebook created in 2022, its wrongdoings and a list of things that needs to be fixed in the restructure and completion of the project. It also lists issues 

Issues and findings will be labelled with a priority legend according to the order that they need to be fixed.

## Priority Legend

**!!!**: critical; might invalidate the reported result
**!!**: high; wrong behaviour that needs to be fixed or a large and avoidable loss in quality
**!**: low; clarity or small inefficiencies.

## 1. Evaluation Validity

### 1.1 The test set is the validation set (!!!)

```python
history1 = cnn.fit(train, ..., validation_data=test, ...)
model1_rlr = ReduceLROnPlateau(monitor='val_loss', ...)
model1_mcp = ModelCheckpoint(..., monitor='val_binary_accuracy', save_best_only=True)
cnn.load_weights(path_weight)
```
Testing makes 3 decisions: when to cut the learning rate, which epoch to keep and the reported score. It picks the maximum of 64 noisy test scores, and then reports that maximum which is the selection on the test set. The number is biased optimistically by its construction.

- separate the validation split from the training pool. touch the test split only once.

### 1.2 Validation only sees a random subset of each epoch (!!)

`validation_steps = 1311 // 64 = 20`, meaning 1280 of the 1311 images. The test generator also uses Keras' default `shuffle = true`, meaning each epoch scores a different random 1280 images. 


## 2. Data Handling

### 2.1 Labels are remapped by mutating iterator internals (!!)

```python
train = generator_train.flow_from_directory('Training', ..., class_mode="binary")
train.classes[train.classes == 3] = 1
...
train.classes[train.classes == 2] = 0
```

### 2.2 Resizing destroys the geometry and keeps source cues (!!)

`target_size=(128, 128)` stretches every image so that it becomes a 128 by 128 square regardless of its aspect ratio and keeps the black borders, where the sizes of the black borders differ per source. These are shortcut cues.

## 2.3 No augmentation of the data (!!)

Every augmentation arguement is 0 or `False`. The network memorizes the training set.

### 2.4 Hard coded sample counts (!)

`steps_per_epoch = 5712 // 64`. Counts should come from the data itself.