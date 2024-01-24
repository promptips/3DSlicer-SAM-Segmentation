
# 3DSlicer-SAM-Segmentation

[![arXiv Paper](https://img.shields.io/badge/arXiv-2401.12974-orange.svg?style=flat)](https://arxiv.org/abs/2408.15224) [**`MIDL Paper`**](https://openreview.net/pdf?id=zDOZ0IhLFF)

SAM-Segmentation is designed to assist users in segmenting medical data on <a href="https://github.com/Slicer/Slicer">3D Slicer</a> by comprehensively integrating the <a href="https://github.com/facebookresearch/segment-anything">Segment Anything Model (SAM)</a> developed by Meta.

<img src="SAM-Segmentation/Resources/Icons/SAM-Segmentation.png" width=50% height=50%>

## How to Cite

For those who find this project useful for their research, following are the riveting papers about it:

```bibtex
@article{yildiz2024sam,
  title={SAM \& SAM 2 in 3D Slicer: SAM-Segmentation Extension for Annotating Medical Images},
  author={Yildiz, Zafer and Chen, Yuwen and Mazurowski, Maciej A},
  journal={arXiv preprint arXiv:2408.15224},
  year={2024}
}

@inproceedings{yildiz2024slicersam,
  title={SAM-Segmentation: 3D Slicer Extension for Segment Anything Model (SAM)},
  author={Yildiz, Zafer and Gu, Hanxue and Zhang, Jikai and Yang, Jichen and Mazurowski, Maciej A},
  booktitle={Medical Imaging with Deep Learning},
  year={2024}
}
```

## Installation via Extension Manager

If you wish to install the extension via 3D Slicer's Extension Manager, simply follow the steps below:

- Access the Extension Manager of 3D Slicer (Ctrl+4)
- Search for "SAM-Segmentation"
- Click the "Install" button
- Restart 3D Slicer

## Installation via GitHub Repository

Alternatively, you can install this extension by cloning this repository. Run the following command to do the same:

```
git clone https://github.com/promptips/3DSlicer-SAM-Segmentation.git
```

Before integrating the extension to 3D Slicer, get all the necessary dependencies installed in the 3D Slicer’s Python terminal using the given commands. Make sure to download the SAM <a href="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth">checkpoint</a> into the repository directory.

Once you've prepared the all necessary files, guide 3D Slicer to the extension. If no errors surface in Python's terminal, you'll be ready to put the extension to use!

## Usage

Before you dive in, ensure that you have loaded a file into 3D Slicer. If you've got the extension set up, it should now be listed under **Modules > Segmentation > SAM-Segmentation**.

Prior to beginning the segmentation process, create the required labels for your case by interacting with the "Configure labels in the segment editor" button. Once this is done, navigate through Modules > Segmentation > SAM-Segmentation path again. Now, you're all set to begin segmentation!

If you need further assistance, watch our [tutorial video](https://youtu.be/PAW2iIXMGvY) to learn more about how to use SAM-Segmentation.