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

### 1.3 Train / Test Leakage inside the dataset (!!!)

The Kaggle Brain Tumor MRI Dataset merges three sources: figshare, SARTAR and Br35H. There are reports of other people (Vangala et al.) where some test images are a near duplicate in the train dataset.

- Record exact Kaggle dataset version and per split, per class counts
- Recompute the exact duplicates using file hash and near duplicates using perceptual hashing within and across splits
- Check patient overlap by matching test images to the original figshare dataset which ships patient IDs.
- Write the results to `docs/LEAKAGE.md.`

### 1.4 Source and label confounding / shortcut learning (!!!)

Data points which are labelled 0 comes from Br35H almost entirely, and label 1 is figshare + SARTAJ. These sources differ in resolution, aspect ratio, borders, sequence and much more. 

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

### 2.3 No augmentation of the data (!!)

Every augmentation arguement is 0 or `False`. The network memorizes the training set.

### 2.4 Hard coded sample counts (!)

`steps_per_epoch = 5712 // 64`. Counts should come from the data itself.

### 2.5 Class imbalance

1686 of 5803 training images are no tumour images.

- Add class weights in the loss, report the balanced accuracy and sensitivity.

## 3. Model

### 3.1 Large portion of the parameters sit in one dense layer (!!!)

### 3.2 CNN trained from scrathc (!!)

- Use pretrained options for better results

## 4. Training (!!)

### 4.1 Early stopping on training loss (!!)

### 4.2 No seeding, one run, no uncertainty, no bootstrapping.

