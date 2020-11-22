# ISY5004
COMPARISON OF UNET, UNET++, RES-UNET, SEGNET AND SEG-UNET FOR SEGMENTATION OF CELLS’ NUCLEI IMAGE

## Requirement
- Python 3.7    
- Tensorflow-gpu 1.15.0  
- Keras 2.3.1
- pandas 1.0.5
- numpy 1.19.2
- matplotlib 3.3.0
- scikit-image 0.16.2
- scikit-learn 0.20.3

## Dataset
Please refer to [here](https://www.kaggle.com/c/data-science-bowl-2018/data)
unzip the dataset and put it under：
```
|-- ISY5004
    |-- h5
    |-- model_plot
    |-- utils
    |-- plot_loss.py
    |-- test.py
    |-- train.py
    |-- data-science-bowl-2018
    |   |-- stage1_test
    |   |-- stage1_train
```

## Experiment
**Train models:**

```
python train.py
```

**Test models:**

```
python test.py
```
**Plot losses:**

```
python plot_loss.py
```
## Results 
**COMPARISON**
![LCTFP](https://github.com/se7ven012//ISY5004/tree/main/model_plot/SegmentationResults.png)

