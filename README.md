Entity Recommendation for Everyday Digital Tasks
==============================================

This repository contains code and data implementing the methods and experiments described in

Giulio Jacucci, Pedram Daee, Tung Vuong, Salvatore Andolina, Khalil Klouche, Mats SjÖberg, Tuukka Ruotsalo, and Samuel Kaski  
**Entity Recommendation for Everyday Digital Tasks**,  TOCHI 2021 
https://doi.org/10.1145/3458919 

**Abstract**: 

Recommender systems can support everyday digital tasks by retrieving and recommending useful information contextually. This is becoming increasingly relevant in services and operating systems. Previous research often focuses on specific recommendation tasks with data captured from interactions with an individual application. The quality of recommendations is also often evaluated addressing only computational measures of accuracy, without investigating the usefulness of recommendations in realistic tasks. The aim of this work is to synthesize the research in this area through a novel approach by (1) demonstrating comprehensive digital activity monitoring, (2) introducing entity-based computing and interaction, and (3) investigating the previously overlooked usefulness of entity recommendations and their actual impact on user behavior in real tasks. The methodology exploits context from screen frames recorded every 2 seconds to recommend information entities related to the current task. We embodied this methodology in an interactive system and investigated the relevance and influence of the recommended entities in a study with participants resuming their real-world tasks after a 14-day monitoring phase. Results show that the recommendations allowed participants to find more relevant entities than in a control without the system. In addition, the recommended entities were also used in the actual tasks. In the discussion, we reflect on a research agenda for entity recommendation in context, revisiting comprehensive monitoring to include the physical world, considering entities as actionable recommendations, capturing drifting intent and routines, and considering explainability and transparency of recommendations, ethics, and ownership of data.


# Code logic

The starting point of the code is main.py file. However, please note that the user data is not available in this repository.
Please refer to DataLoader.py, DataProjector.py, and UserModelCoupled.py as the main recommendation system components.

 main.py controls the following process:
 
      1. Load the experiment logs 
      2. Create (or load) the low dimensional representation of data
      3. Interaction loop:
          3.1. Receive new snapshots (real-time documents)
          3.2. Update the user model
          3.3. Recommend items from different views
          3.4. gather feedback for items
          
          
## Reference

If you are using this source code in your research please consider citing us:

 * Giulio Jacucci, Pedram Daee, Tung Vuong, Salvatore Andolina, Khalil Klouche, Mats SjÖberg, Tuukka Ruotsalo, and Samuel Kaski. 2021. **Entity Recommendation for Everyday Digital Tasks**. ACM Trans. Comput.-Hum. Interact. 28, 5, Article 29 (October 2021), 41 pages. DOI:https://doi.org/10.1145/3458919


## License

