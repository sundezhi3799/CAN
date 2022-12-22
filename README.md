# CAN

User Guide of Chamber Attention Network (CAN)

##Introduction

This is the CAN toolkit developed for accurate diagnosis of pulmonary artery hypertension using echocardiography. You can set the suprameters according to your needs.

Dezhi Sun (sundezhi3799@163.com) 

##Install

```
python -m pip install -r requirements.txt
```
##Usage

```python .\classify_viewclass.py -d  "echo-jpg-2022" -g '0'```<br>
- -d, the input directory of images;
- -g, cuda device to use;

```python .\segment_a4c_plax.py -p 'I00610693534.dcm.jpg' -g '0' -v 'a4c'```<br>
- -p, the path of the input image;
- -g, cuda device to use;
- -v, the view of the image to segment.

```python detect_pah.py -d  "142297583" -g '0'```
- -d, the input directory of a subject consisting of images;
- -g, cuda device to use.
