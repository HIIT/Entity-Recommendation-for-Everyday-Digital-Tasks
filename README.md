# Entity Recommendation for Everyday Digital Tasks

The starting point of the code is main.py file. However, please note that the user data is not available in this repository.
Please refer to DataLoader.py, DataProjector.py, and UserModelCoupled.py as the main recommendation system components.


To be added: 
 - Link to the paper and DOI


 main.py controls the following process:
 
      1. Load the experiment logs 
      2. Create (or load) the low dimensional representation of data
      3. Interaction loop:
          3.1. Receive new snapshots (real-time documents)
          3.2. Update the user model
          3.3. Recommend items from different views
          3.4. gather feedback for items
