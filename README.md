# High-Dynamic-Range-Imaging

## Usage

```bash
$ python main.py 
    <--img_path = string> 
    <--showPlot = bool> 
    <--mtb_level = int> 
    <--sample_mode = random/uniform> 
    <--lambda_ = int> 

 # example
$ python main.py --img_path=./night01 --showPlot=True --mtb_level=6 --sample_mode=uniform lambda_=30

The hdr image will output in the img_path. 
There are two file. (hdr_nm.jpg, hdr_tm.jpg)
The first one normalize radiance between 0 to 255
The second one uses the method by Photographic Tone Reproduction for Digital Images.
```
