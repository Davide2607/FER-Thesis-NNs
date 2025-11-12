1) yolo is actually wrong in terms of code since I now get 69%. Just look at Adele's code and try
4) Research "online" augmentations (with "online" as in "real time while running the training process")
    - to see if I can make the occlusions on the fly
    - or if I should be creating a shit ton of images
    > https://chatgpt.com/c/6900e9cf-960c-8328-83aa-645d2a999140
    > Try the various approaches (only augs, % of augs, growing % of augs)
    > Take into account storing maybe just the landmarks in case it's an overhead, although it shouldn't be. Based on my experience the patching was much heavier, so maybe optimize that code.


# Online Occlusion Task
1) Create a pipeline to get the list_of_landmark_sets given an emotion JUST LIKE you did in occlude.py from d3 Masks
2) 