## What’s that instrument?
### Identifying Instruments

Consider the following audio clip: [Guess the instrument!] (/008__[cel][nod][cla]0058__1.wav)
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

With diagram, no discussion

Include examples

### Discussion of results

Discussion

### Future Work
If we were to continue this project, we could expand from the IRMAS dataset to include other instruments and more audio files. For example, never mind auditorially, many fail to visually distinguish a euphonium from a tuba. There are many more than 11 instruments in a symphony orchestra that could be included.
Also, our method of preprocessing could be changed. For example, the log transform served to emphasize patterns in lower frequencies, but this might not be necessary for a model to pick up on the patterns.

### Acknowledgements

Acknowledge dataset, Don, relevant python libraries, Colab
