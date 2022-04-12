# DocParser: Hierarchical Structure Parsing of Document Renderings
## Codes for the system presented in "DocParser: Hierarchical Structure Parsing of Document Renderings"
[paper](https://arxiv.org/abs/1911.01702)


### Installation and requirements

Tested for Ubuntu 18.04/20.04.

Use of a GPU significantly speeds up generation of detection outputs, but it is possible to run the inference demo code on CPU.

To setup via Anaconda, please follow these steps:

1. Install anaconda. Up-to-date instructions can be found at: https://docs.anaconda.com/anaconda/install/

2. Set up python 3.6 environment: 
`conda create -n docparser python=3.6`

3. Activate the environment:
`source activate docparser`

4. Install all requirements:
`pip install -r requirements.txt`
	- (for GPU-enabled installation: `pip install -r requirements_gpu.txt`)


5. Install Mask R-CNN library:
    - We used a slightly modified version of https://github.com/matterport/Mask_RCNN, though the original version should still be usable, possibly with minor adaptions.
    - Clone repository from https://github.com/j-rausch/Mask_RCNN 
	- Change into mask rcnn directory 
	- type `python setup.py install`

6. Install docparser:
	- Change into DocParser directory 
	- type `python setup.py develop`

7. Prepare the datasets:
	- Download arxivdocs-target from:
    https://drive.google.com/file/d/1qVR_mI2xBtgsUG0LWVJctygDzvkMYN7C/view?usp=sharing
	- To run the ICDAR demo, download the prepared files from:
    https://drive.google.com/file/d/1SdGTq80eUGqUJBA6kdVQBO9L6a_ijAcN/view?usp=sharing
	- Extract datasets to the `DocParser` subdirectory 
		- (resulting in structure: `DocParser/datasets`). 

8. Prepare the trained models:
	- Download from URL:
    https://drive.google.com/file/d/1Hi4-tg4Zmtx8zYiCg6IBi47R88PdmAW4/view?usp=sharing 
	- Extract the pretrained models to the `default_models` subdirectory in `DocParser/docparser/`
		- (resulting in structure `DocParser/docparser/default_models/`).
    - For convenience, we include the COCO pre-trained weights from from https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5 in the zip file

9. For running the ICDAR demo:
	- Please note that, in order to run the ICDAR 2013 evaluation script provided by the competition organizers, a Java installation is necessary. We used `openjdk 11.0.7 2020-04-14` in our experiments. 
	- If necessary, update permissions for the evaluation script (on linux systems):
		`chmod a+x DocParser/docparser/utils/dataset-tools-20180206.jar`


10. From the `DocParser` directory, execute:
`python demos/demo_inference.py` plus one or more of the following command line arguments:
	
	- `--page`
	- `--table`
	- `--icdar`
	- e.g. `python demos/demo_inference.py --page --table`


### Evaluations

#### arXivDocs
The results of our current system on arXivDocs-target is likely to perform better than the one evaluated in the last version of the paper, mostly due to further improvements to postprocessing.  

#### ICDAR 2013, Table Structure Recognition
Updated Results. We corrected a read-out error on the outputs of the provided evaluation script for documents with multiple tables.

| System            | F1*    | F1     |
|-------------------|--------|--------|
| DocParser Baselie | 0.8443 | 0.8209 |
| DocParser WS      | 0.8117 | 0.8056 |
| DocParser WS+FT   | 0.9292 | 0.9292 |

(PDF-based system F1: 0.9221)

### Credits
Parts of our code is based on:
https://github.com/rafaelpadilla/Object-Detection-Metrics

https://github.com/matterport/Mask_RCNN

### Reference
Rausch, J., Martinez, O., Bissig, F., Zhang, C., & Feuerriegel, S. - 35th AAAI Conference on Artificial Intelligence (AAAI-21)(virtual) 



