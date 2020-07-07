# Gait Based User Recognition from Channel State Information(CSI) collected from WiFi Data
Deep learning methods for user recognition from Channel State Information(CSI) collected from WiFi Data. We used [Intel CSI Tool](https://dhalperi.github.io/linux-80211n-csitool/) and [Atheros CSI Tool](https://github.com/kdkalvik/Atheros-CSI-Tool) to collect our gait dataset(private). The methods in this repo convert CSI data to spectrograms and use deep learning methods for user recognition. Refer to the publications listed below for further details of the method.

## Directory Structure
- `models/`: Deep learning models/experiments
- `preprocessing/`: Scripts used to preprocess the CSI data
- `theoretical_derivation/`: Theoretical derivation of the expected spectrograms from human gait
- `tools/`: Arbitrary tools to plot results/spectrograms, debug CSI tool

## Citation
Please cite the following paper if you use this repository in your research.
```
@inproceedings{pokkunuru2018neuralwave,
  title={NeuralWave: Gait-based user identification through commodity WiFi and deep learning},
  author={Pokkunuru, Akarsh and Jakkala, Kalvik and Bhuyan, Arupjyoti and Wang, Pu and Sun, Zhi},
  booktitle={IECON 2018-44th Annual Conference of the IEEE Industrial Electronics Society},
  pages={758--765},
  year={2018},
  organization={IEEE}
}

@article{jakkala2019deep,
  title={Deep CSI learning for gait biometric sensing and recognition},
  author={Jakkala, Kalvik and Bhuya, Arupjyoti and Sun, Zhi and Wang, Pu and Cheng, Zhuo},
  journal={arXiv preprint arXiv:1902.02300},
  year={2019}
}
```
