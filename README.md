# GHD_RAG

This repo is an essential tool for conducting comprehensive tests and evaluations on a chatbot system based on basic RAG model. The primary purpose of this notebook is to facilitate the deployment of various configurations to assess the effectiveness and accuracy of chatbot responses. Below is a detailed overview of the contents and functionalities of this notebook:

## Key Functionalities:

* **Database Querying:** Functions are defined to interact with a database to retrieve relevant paragraphs based on user-submitted questions. This setup allows for a focused analysis of the chatbot's ability to understand and respond to queries effectively.
* **Chatbot Interaction:** The notebook includes several functions that manage interactions with the chatbot under different settings. These functions enable customization of parameters such as the model used, sensitivity thresholds, limits on the number of results returned, and the temperature setting that influences response variability.
* **Experiment Execution:** To systematically evaluate the chatbot's performance, the notebook provides functionality to execute experiments using varied configurations. Each experiment can be adjusted according to specific test conditions to analyze different aspects of the chatbot's functionality.
* **Results Management:** After running experiments, the notebook handles the output by organizing the results and saving them into Excel files. This feature is crucial for later analysis and review of the chatbot's performance across different settings.
* **Error Handling and Progress Tracking:** The notebook is equipped with error handling to address issues during execution, ensuring stability and reliability during tests. Progress indicators are also incorporated to monitor the duration of operations, especially useful during lengthy experiment runs.

## Neo4j Database Setup:

To set up the Neo4j Graph Database, use the dump files for each chunking strategy (paragraph or 200_chunk) to populate the database with relevant data. The database will be used to retrieve paragraphs based on user queries during chatbot interactions. To run experiments, install the Neo4j Desktop application and create a new database using the dump files provided. Make sure to have APOC and Graph Data Science libraries installed in the database.

While running the chat.py files, make sure to run the Neo4j Database in the background. The chat.py files will interact with the database to retrieve relevant paragraphs based on user queries. The chatbot will then generate responses based on the retrieved paragraphs and the specified model.

For more information on installing and setting up the Neo4j Database, refer to the official documentation: https://neo4j.com/docs/

## Credetials:

All credentials are stored in the cred.env file which we have provide a template for. Please fill in the necessary information and rename the file to .env before running the chat.py file.

## Chatbot Instructions:

Install all python dependencies by running the following command:

```
pip install -r requirements.txt
```
or 
```
pip3 install -r requirements. txt
```

To run the chatbot, navigate to the appropriate folder based on the chunking strategy (paragraph or 200_chunk) folder and run the chat.py file with the following command:


```
python chat.py model threshold input_prompt result_limit temperature output_config exp_mode
```

For paragraph chunking database: go into the 200_chunk folder and run:

```
python chat.py model threshold input_prompt result_limit temperature output_config exp_mode

```

exp_mode = 0 for normal chatbot interaction, 1 for experiment mode

You are welcomed to adjust the parameters based on your requirements to evaluate the chatbot's performance under different settings. The chatbot will generate responses based on the specified model and other parameters provided in the command. You can also adjust the code to include additional models or modify existing ones to enhance the chatbot's capabilities.