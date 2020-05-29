# Project Title
Laplace based perceptual hashing for object tracking using OpenCV

## Motivation
This project is Assignment for Digital Image Processing (ENGR7761) at Flinders University

### Laplace Hash Technique
The technique purpose is comparing hash code generated from laplaced based image processing using hamming distance. The less the distance between two codes means the image is similar.

## Screenshot
![Laplace Hash Object Tracking](Result/result.gif)

## Getting Started
The project is developed using Python and OpenCV and tested under MacOS.

### Prequisites
- Python 
- OpenCV
- Virtual Environment

```
pip install opencv-python
```

## Running the App
To run the app head to hash.py file and download this [dataset](https://flinders-my.sharepoint.com/:f:/g/personal/pray0008_flinders_edu_au/Ejrv8BzD3RxIiOpqVr47Yu0B7guroObn_hfJlpLIeFvUJA?e=xZe1e5)

Simply run the app by typing:
```
python hash.py
```

## Tests
OpenCV window will pop up. Draw rectangle on object in the first frame by click on the top left point and right bottom point  around the object. Press "C" to continue the the frame. To track object in the next frame, press "P" to pause the video and draw rectangle around object. Mini window will pop up soon as the sliding window technique is finished track the object.

## Authors

* **Rezka Bunaiya Prayudha**



## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* pyimagesearch.com for the Sliding Window program
