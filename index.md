## What’s that instrument?
### Identifying Instruments

Consider the following audio clip: [Guess the instrument!](/008__[cel][nod][cla]0058__1.wav)
What musical instrument are you listening to? 
To those of you who recognized the instrument, awesome! But to those of you who thought to yourself, “ah, that’s a violin,” this deeply offends the groupmate who is a cellist; he’ll have you know that that was an audio clip of the cello. Not a “big violin;” a cello.
Don’t you wish you had some sort of tool to distinguish between the different sounds of different instruments? Because to those of you who responded incorrectly, the cellist hopes you have a tool; this project is one such tool.

### Overview

1. Convert audio files into spectrogram images.
2. Create and train models.
3. Create visualizations of the models’ performances.

### Dataset and Data Preprocessing
The dataset used in this project was [IRMAS](https://zenodo.org/record/1290750#.YTfo455KhTa). The dataset contains 6,705 2-second audio clips of 11 instruments: cello, clarinet, flute, acoustic guitar, electric guitar, organ, piano, saxophone, trumpet, violin, and voice. 
The audio clips are of these instruments playing some short segment of music; all these clips have the instruments playing different sections in music with different pitches and volumes.
For each of these clips, we
- generated a waveform, which was not particularly due to the instruments being of varying volumes
- generated a spectrogram, which looks at the frequencies of each audio file, giving a better understanding of what the “shape” of the audio was, and
- applied a log transform to the spectrogram, spreading out the frequencies differently, which could make the patterns in lower frequencies easier to detect.
For the example audio, here are the aforementioned visualizations of the audio:

![Example of data preprocessing](/Screen Shot 2021-09-07 at 6.40.08 PM.png)

In this way, a dataset of audio files was converted into a dataset of images.


### What is a Convolutional Neural Network?

![CNN](https://user-images.githubusercontent.com/89939151/132454209-9667744c-99cd-4575-bf75-b67d2253a3f4.PNG)

A convolutional neural network or CNN is a deep learning algorithm that takes an input image, assigns importance to certain objects in the input image through weights and biases, and then differentiates an input image from other inputs. In this particular project, CNNs are preferable to RNNs compared to other model types because they are able to deal with spatial data more efficiently. In our case, we started with audio files which are not inputtable into a CNN so we had to go through an initial data preprocessing step that turned the audio files into inputtable images. The CNN here will essentially reduce the parameters involved and apply relevant filters (which can be seen as key features of an image) to the image in order to reduce the amount of processing necessary to classify it and to increase efficiency. 

### Even more neural network definition things

See definition about thing

### Results
#### Baseline 6 layer Model

#### ResNet18 Model
![Original-ResNet-18](https://user-images.githubusercontent.com/89939151/132442868-598ad361-77bf-4ef8-b68c-192c2fab460f.png)

Afterward, we used a ResNet18 Model which has 18 layers compared to the 6 layers of the baseline model. Normally, in a convolutional neural network, each successive layer adds more in-depth analysis, with each one examining a certain aspect of the image (shapes, edges, etc.) Therefore, it makes sense that a CNN with more layers would be more accurate. However, this is not what we found. The most accurate models only had between 16-30 layers. Although the exact reason for this cause is unknown, one theory is due to the idea of the vanishing gradient problem. A neural network uses gradients to adjust its weight to reduce loss and maximize accuracy. However, sometimes in the beginning layers, the gradient is astronomically tiny (hence the name “vanishing gradient”), which means that our weights would not be adjusted properly throughout the network. If this is the case, adding more layers wouldn’t be helpful. 

![Graph](https://user-images.githubusercontent.com/89939151/132444419-06370ff0-d23a-4062-8c07-9506b12692cf.PNG)

This is a graph of the training and validation accuracy over 18 epochs. By examining the graph, we can see that the ResNet18 model performed much better than the baseline model. Both the training and validation accuracies remain above 60%, hovering around 70, and with high accuracy, the cross-entropy loss is low. In comparison, the highest validation accuracy from the baseline model was about 45%. Furthermore, the baseline model started to overfit around epoch 5 when the validation accuracy fell below the training accuracy, while the ResNet18 model started to overfit around epoch 8 as its validation accuracy gradually leveled off.

![Matrix](https://user-images.githubusercontent.com/89939151/132448414-0a822c84-0d4a-4816-840a-425d73dfcf60.PNG)

The preciseness of the ResNet18 model is also supported by its confusion matrix. Clearly visible along the diagonal line are the dark blue squares that contain the majority of the numbers. This indicates that the model predicted a majority of the instruments correctly as the numbers should appear where the predicted and actual instruments meet. As a total average, the ResNet 18 model correctly predicted the instrument 68.04% of the time which is much higher than the baseline model. The instrument with the highest accuracy was the flute with 85.18% and the instrument with the lowest accuracy was the violin with 59.82%. Taking a closer look at the latter reveals that the violin was most commonly mistaken for an acoustic guitar (a total of 10 times). Conversely, the acoustic guitar was also mostly mistaken for the violin (a total of 7 times). From this information, we can assume that the spectrograms of the two instruments were similar in shape which might have confused the CNN!

Violin Spectrogram
![Similar violin spectrogram](/vio108.png)
Voice Spectrogram
![Similar voice spectrogram](/voi103.png)

As shown above, the spectrograms do indeed have some major similarities. They both start out with thick blocks at the bottom that gradually become thin stripes near the top. Although the violin has shorter stripes than the voice, the placement and shape are similar enough to where the model might have been confused.

### Implications

The model suggests that converting audio clips to visual spectrograms is an effective method to differentiate between sounds. This piece of information can have much more appropriate and useful applications in real life. For example, this tool can be used to identify and target different species of animals from their mating calls, which would be quite valuable in the environmental field. It could also be used to identify voices from an audio recording. Although unlikely, in the distant future, governments could potentially have a stored database of the voices of their citizens, similar to fingerprints, and use it to prosecute criminals using a similar, more advanced model.

### Future Work
If we were to continue this project, we could expand from the IRMAS dataset to include other instruments and more audio files. For example, never mind auditorially, many fail to visually distinguish a euphonium from a tuba. There are many more than 11 instruments in a symphony orchestra that could be included.
Also, our method of preprocessing could be changed. For example, the log transform served to emphasize patterns in lower frequencies, but this might not be necessary for a model to pick up on the patterns.

### Acknowledgements
- IRMAS dataset
  - Juan J. Bosch, Ferdinand Fuhrmann, & Perfecto Herrera. (2014). IRMAS: a dataset for instrument recognition in musical audio signals (1.0) [Data set]. 13th International Society for Music Information Retrieval Conference (ISMIR 2012), Porto, Portugal. Zenodo. https://doi.org/10.5281/zenodo.1290750
- ResNet18 Diagram
  - https://www.researchgate.net/figure/Original-ResNet-18-Architecture_fig1_336642248
- Other Convolutional Neural Network Diagrams
  - https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
- How to Create Basic Convolutional Neural Network
  - https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-photos-of-dogs-and-cats/
- Stack Overflow and Google Colaboratory
- PyTorch, Keras 
- Librosa
  - McFee, Brian, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto. “librosa: Audio and music signal analysis in python.” In Proceedings of the 14th python in science conference, pp. 18-25. 2015.
- Numpy, Sci-kit Learn, Matplotlib
- Professor Donald Smith
