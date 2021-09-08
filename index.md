# What’s that instrument?

By: Eric Wang, Bobby Bhandari, and Vibhas Nair

## Identifying Instruments

Consider the following audio clip: [Guess the instrument!](/008__[cel][nod][cla]0058__1.wav)

What musical instrument are you listening to? 

To those of you who recognized the instrument, awesome! But to those of you who thought to yourself, “ah, that’s a violin,” this deeply offends the member of our group who is a cellist; he’ll have you know that that was an audio clip of the cello. Not a “big violin;” a cello.
Don’t you wish you had some sort of tool to distinguish between the different sounds of different instruments? Because to those of you who thought the cello was a violin, the cellist hopes you have a tool; this project is one such tool.

## Overview

1. Convert audio files into spectrogram images.
2. Create and train models.
3. Create visualizations of the models’ performances.
4. Analyze and compare results.

## Dataset and Data Preprocessing

The dataset used in this project was [IRMAS](https://zenodo.org/record/1290750#.YTfo455KhTa). The dataset contains 6,705 2-second audio clips of 11 instruments: cello, clarinet, flute, acoustic guitar, electric guitar, organ, piano, saxophone, trumpet, violin, and voice. 
The audio clips are of these instruments playing some short segment of music; all these clips have the instruments playing different sections in music with different pitches and volumes.

For each of these clips, we
- generated a waveform, which was not particularly useful due to the instruments being of varying volumes,
- generated a spectrogram, which looks at the frequencies of each audio file, giving a better understanding of what the “shape” of the audio was, and
- applied a log transform to the spectrogram, spreading out the frequencies differently, which could make the patterns in lower frequencies easier to detect.

For the example audio, here are the aforementioned visualizations of the audio:

![Example of data preprocessing](/Screen Shot 2021-09-07 at 6.40.08 PM.png)

In this way, a dataset of audio files was converted into a dataset of images.


## What is a Convolutional Neural Network?

![CNN](https://user-images.githubusercontent.com/89939151/132454209-9667744c-99cd-4575-bf75-b67d2253a3f4.PNG)

A convolutional neural network or CNN is a deep learning algorithm that takes an input image, assigns importance to certain objects in the input image through weights and biases, and then classifies the input image. In this particular project, CNNs are preferable to other model types like RNNs because they are able to deal with spatial data more efficiently. In converting the audio files, a 1-dimensional stream of bits, to an image, a 2-dimensional matrix of pixel values, the audio files are now converted into spatial data, so a CNN is appropriate. The CNN here will essentially reduce the parameters involved and apply relevant filters (which can be seen as key features of an image) to the image in order to reduce the amount of processing necessary to classify it and to retain spatial information. 

## How does a CNN work?

![Convolutional layer gif](/convlayer.gif)

In general, there are two important types of layers in a CNN: a convolution and pooling layer. A computer treats an image as a matrix of pixel values. This matrix then goes through the first layer which is convolution. In this layer, a process of matrix multiplication happens where the input matrix is multiplied by a filter that goes through the entire image. This filter outputs a convolved feature which is essentially the result of the model extracting the high-level information from the image (e.g. if the image was a dog, convolution could extract an eye). This convolved feature then goes through an additional processing layer called max pooling.

![Max pooling layer png](/Screen Shot 2021-09-08 at 11.22.00 AM.png)

Max pooling outputs the maximum value of a portion of the image covered by the filter. This process is similar to convolution where it reduces the necessary computational power but this also acts as a noise suppressant where by taking the maximum value, it discards all of the information that isn’t as important. These two processes repeat a number of times depending on the size of the model (we will talk about the importance of size later) but the matrix value it outputs is eventually flattened and fed into a bread-and-butter multilayer perceptron. Then, through a process of softmax classification, the input image is classified. In the case of our project, each spectrogram would go through this process to be identified as one of the eleven instruments.

## Results

### Baseline 6 layer Model

We started with a smaller baseline model that only had six layers. In other words, this model had three sets of convolution and max pooling. Since this model was quite small the results did not end up completely ideal. 

![Image of accuracy and loss plots](/IRMAS_3block_model_plot.png)

We trained our baseline model for around eighteen epochs, or iterations of the training dataset, and calculated the loss and accuracy for the training and validation sets in the graphs above. The training graph behaved normally where the loss decreased and the accuracy increased with the number of epochs. With the validation set, however, this started overfitting around the sixth epoch. At first, the validation loss and accuracy were similar to the training loss and accuracy, but at the sixth epoch, the model started to become too accustomed to the training data. This is overfitting—where the model begins to learn the training data too well and instead of identifying spatial patterns in the spectrograms, the model begins to just identify the training data. When trying to classify the validation data, the model is confused. In terms of validation loss, the model started to hover around 2.0 at the point of overfitting which is a considerably high value. The model’s overall accuracy peaked just below 40% at the point of overfitting. This model was definitely better than baseline guessing (which is around 9% accuracy for 11 instruments) however, it's not much better than that. An accuracy of 40% shows that this type of CNN is worth creating as it is  better than random guessing, but in comparison to other models, it doesn’t work effectively. 

![Confusion Matrix 1](https://user-images.githubusercontent.com/89939151/132530832-3a73b26c-c709-4d11-9023-24958b1bddac.PNG)

(cello, clarinet, flute, acoustic guitar, electric guitar, organ, piano, saxophone, trumpet, violin, and voice in order of abbreviations)

This is a confusion matrix of the baseline model that allows us to visualize which instruments the model had the most and least difficulty predicting. The Y-axis is the actual instruments and the X-axis is the model’s predictions. So a perfect model with 100% accuracy would have all the numbers lined up in a diagonal where the actual and predicted instruments meet, with every other box containing a 0. We can determine that the model did a poor job of predicting the instruments as there are a lot of large numbers that are outside of the diagonal; this is supported by its low accuracy of 40%. One way to evaluate the model's performance with individual instruments is recall: here, recall is the percentage of times the model correctly identified a given instrument out of the total number of that instrument. The instrument with the highest recall was the organ with 62.83%, while the lowest was the flute with 13.01%, with the violin coming in at a close second with 15.57%. It’s interesting that the model was able to differentiate the violin from a trumpet 100% of the time, but it wasn’t able to do the same with the trumpet; the model predicted the trumpet to be the violin 17 times. This could indicate that the spectrograms of the two instruments are somehow closely related.

Trumpet Spectrogram

![tru116](https://user-images.githubusercontent.com/89653898/132536117-68f6e373-7115-41ca-90f9-3af6c1fc7cdc.PNG)

Violin Spectrogram

![vio199](https://user-images.githubusercontent.com/89653898/132536253-a1b41f50-e7ad-4a9b-9cf3-6509bf7f2da7.PNG)

As seen in the confusion matrix, interestingly the model often confused the trumpet spectrograms as violin spectrograms, but not the other way around. The two spectrograms are shown above where there is a pattern of a large red bar on the bottom left of both of the images that began to thin out near the top. Perhaps this is why the model confused the trumpet for the violin. However, there is more space between the thin stripes in the violin spectrogram which may have allowed the model to differentiate the violin from the trumpet. 

### ResNet18 Model

![Original-ResNet-18](https://user-images.githubusercontent.com/89939151/132442868-598ad361-77bf-4ef8-b68c-192c2fab460f.png)

Afterward, we used a ResNet18 Model which has 18 layers compared to the 6 layers of the baseline model. Normally, in a convolutional neural network, each successive layer adds more in-depth analysis, with each one examining a certain aspect of the image (shapes, edges, etc.) Therefore, it makes sense that a CNN with more layers would be more accurate. However, this is not what we found. The most accurate models only had between 16-30 layers. Although the exact reason for this cause is unknown, one theory is due to the idea of the vanishing gradient problem. A neural network uses gradients to adjust its weight to reduce loss and maximize accuracy. However, sometimes in the beginning layers, the gradient is astronomically tiny (hence the name “vanishing gradient”), which means that our weights would not be adjusted properly throughout the network. If this is the case, adding more layers wouldn’t be helpful. 

![Graph](https://user-images.githubusercontent.com/89939151/132544543-e72872b6-42f9-4bee-8f37-a7d08d64eab2.PNG)

This is a graph of the training and validation accuracy over 18 epochs. By examining the graph, we can see that the ResNet18 model performed much better than the baseline model. Both the training and validation accuracies remain above 60%, hovering around 70%, and with high accuracy, the cross-entropy loss is low. In comparison, the highest validation accuracy from the baseline model was about 45%. Furthermore, the baseline model started to overfit around epoch 5 when the validation accuracy fell below the training accuracy, while the ResNet18 model started to overfit around epoch 8 as its validation accuracy gradually leveled off.

![Matrix](https://user-images.githubusercontent.com/89939151/132448414-0a822c84-0d4a-4816-840a-425d73dfcf60.PNG)

The preciseness of the ResNet18 model is also supported by its confusion matrix. Clearly visible along the diagonal line are the dark blue squares that contain the majority of the numbers. This indicates that the model predicted a majority of the instruments correctly as the numbers should appear where the predicted and actual instruments meet. In contrast, the baseline confusion matrix had light squares in its diagonal since it had a lower accuracy. There were many instruments that were previously misidentified by the 6 layer model that the ResNet18 model identifies accurately. For example, the flute had a recall of 13.01%, being commonly confused with the acoustic guitar in the baseline model. However, in the ResNet18 model, it had a recall of 85.18%, improving significantly.

The instrument with the highest recall was the flute with 85.18% and the instrument with the lowest was the violin with 59.82%. For example. the violin was  commonly mistaken for the human voice (a total of 6 times). Conversely, the voice was also commonly mistaken for the violin (a total of 11 times). From this information, we can assume that the spectrograms of the two instruments were similar in shape which might have confused the CNN!

Violin Spectrogram

![violon spectorgram](https://user-images.githubusercontent.com/89939151/132545429-0dcb6486-8ce3-4fe4-9f6b-d9846a062b77.png)

Voice Spectrogram

![Voice Spectrogram](https://user-images.githubusercontent.com/89939151/132545479-0e481f05-a3e6-44e1-87c0-31c9f02eb889.png)

As shown above, the spectrograms do indeed have some major similarities. They both start out with thick blocks at the bottom that gradually become thin stripes near the top. Although the violin has shorter stripes than the voice, the placement and shape are similar enough to where the model might have been confused.

## Implications

The model suggests that converting audio clips to visual spectrograms is an effective method to differentiate between sounds. This piece of information can have much more appropriate and useful applications in real life. For example, this tool could be used to identify and target different species of animals from their mating calls, which would be quite valuable in the environmental field. It could also be used to identify voices from an audio recording. Although unlikely, in the distant future, governments could potentially have a stored database of the voices of their citizens, similar to fingerprints, and use it to prosecute criminals using a similar, more advanced model.

## Future Work

If we were to continue this project, we could expand from the IRMAS dataset to include other instruments and more audio files. For example, never mind auditorially, many fail to visually distinguish a euphonium from a tuba. There are many more than 11 instruments in a symphony orchestra that could be included.
Also, our method of preprocessing could be changed. For example, the log transform served to emphasize patterns in lower frequencies, but this might not be necessary for a model to pick up on the patterns.

## Conclusion

Our method was as follows:
1. We used the IRMAS dataset for audio clips of 11 instruments.
2. Each of the audio clips was converted into spectrograms with a log transform applied to get a visual representation of our audio clips.
3. A 6 layer convolutional neural network and ResNet18 model were trained.
4. The 6 layer CNN showed that this method is viable as the model performed better than random guessing.
5. The ResNet18 model had better results as seen in the confusion matrices and plots of accuracy, making less mistakes than the 6 layer CNN.

## Acknowledgements

- IRMAS dataset for audio clips of instruments
  - Juan J. Bosch, Ferdinand Fuhrmann, & Perfecto Herrera. (2014). IRMAS: a dataset for instrument recognition in musical audio signals (1.0) [Data set]. 13th International Society for Music Information Retrieval Conference (ISMIR 2012), Porto, Portugal. Zenodo. https://doi.org/10.5281/zenodo.1290750
- ResNet18 Diagram
  - https://www.researchgate.net/figure/Original-ResNet-18-Architecture_fig1_336642248
- Other Convolutional Neural Network Diagrams
  - https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
- How to Create Basic Convolutional Neural Network
  - https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/
- Stack Overflow and Google Colaboratory for coding and resolving bugs
- PyTorch and Keras for machine learning frameworks
- Librosa for loading in and process audio clips
  - McFee, Brian, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto. “librosa: Audio and music signal analysis in python.” In Proceedings of the 14th python in science conference, pp. 18-25. 2015.
- Numpy, Sci-kit Learn, and Matplotlib for additional useful functions
- Professor Donald Smith for giving us the knowledge and guidance to present this project
