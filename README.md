# Predictor
A room type predictor using tensorflow. Currently we predict based on the content type of each content exists in the room.
![alt tag](https://github.com/iratao/predictor/blob/master/screencast.gif)

### Setup & Run - Live Demo
To run the live demo, you need to execute the following command.

    $ cd server
    $ pip install -r requirements.txt
    $ python app.py
    

### Setup & Run - Classifier

    $ cd tensorf
    $ python roomtype-softmax.py
    

### Structure Introduction
##### /designjsons
This folder contains the input data for our predicting algorithm. Each file is a design json with the following format:

    {
      "1435" {                                # the room id
        "category": "LivingDiningRoom",       # room type of this room
        "id": "1435",
        "characters": [                       # list of contents that belongs to this room
          {                          
            "id": "e8ad5df2-2e4c-4d6e-9174-2b6bb55fdc38",     # content id/seekid
            "contenttype": "sofa/double seat sofa"            # content type
          },
          {
            "id": "48512a7c-b04d-4487-ae45-bf513e684556",
            "contenttype": "sofa/double seat sofa"
          },
          ...
     },
      ...
    }

##### /dwgjsons
This folder contains the original design jsons which are exported from floorplan.

##### /server
This folder contains the code for live demo.

##### /tensorf
Predictor algorithm.

##### /validators
Data used for validation. Same format as the ones in /designjsons.
