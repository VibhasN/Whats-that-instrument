![Matrix](https://user-images.githubusercontent.com/89939151/132448414-0a822c84-0d4a-4816-840a-425d73dfcf60.PNG)

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


### Neural network definition things

See definition about thing

### Even more neural network definition things

See definition about thing

### Results
#### Baseline 6 layer Model

#### ResNet18 Model
![Original-ResNet-18](https://user-images.githubusercontent.com/89939151/132442868-598ad361-77bf-4ef8-b68c-192c2fab460f.png)

Afterwards, we used a ResNet18 Model which has 18 layers compared to the 6 layers of the baseline model. Normally, in a convolutional Neural Network, each successive layer adds more in-depth analysis, with each one examining a certain aspect of the image (shapes, edges, etc..) Therefore, it makes sense that a CNN with more layers would be more accurate. However, this is not what we found. The most accurate models only had between 16-30 layers. Although the exact reason for this cause is unknown, one theory is due to the idea of Vanishing Gradient. A neural network uses gradients to adjust its weight to reduce loss and maximize accuracy. However, sometimes in the beginning layers, the gradient is astronomically tiny (hence the name “vanishing gradient”), which means that our weights would not be adjusted properly throughout the network. If this is the case, adding more layers wouldn’t be helpful. 

![Graph](https://user-images.githubusercontent.com/89939151/132444419-06370ff0-d23a-4062-8c07-9506b12692cf.PNG)

This is a graph of the training and validation accuracy over 18 epochs. By examining the graph, we can see that the ResNet18 Model preformed much better than the baseline model. Both the training and validaiton accuracies remain above 60%, hovering around 70, and with a high accuracy, the cross-entrophy loss is low. In comparison, the highest validaiton accuracy from the baseline model was about 45%. Furthermore, the baseline model started to overfit around epoch 5 when the validation accuracy fell below the training accruacy, while the ResNet18 model overfitted after the 18 epochs as its validation accracuy gradully leveled off.
With diagram, no discussion

Include examples

### Discussion of results

Discussion

### Future Work
If we were to continue this project, we could expand from the IRMAS dataset to include other instruments and more audio files. For example, never mind auditorially, many fail to visually distinguish a euphonium from a tuba. There are many more than 11 instruments in a symphony orchestra that could be included.
Also, our method of preprocessing could be changed. For example, the log transform served to emphasize patterns in lower frequencies, but this might not be necessary for a model to pick up on the patterns.

### Acknowledgements

Acknowledge dataset, Don, relevant python libraries, Colab
