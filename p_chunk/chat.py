import os
import dotenv
import json
from openai import AzureOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import AzureOpenAIEmbeddings
from neo4j import GraphDatabase
import pprint
import pandas as pd
from tqdm import tqdm
import sys


# Load environment variables
env_path = os.path.join(os.path.abspath(".."), "cred.env")

if os.path.exists(env_path):
    dotenv.load_dotenv(env_path, override=True)
    print("Environment variables loaded successfully.")
else:
    print("Error: .env file not found.")

# Load Azure OpenAI credentials
resource_name = os.environ.get("AZURE_RESOURCE_NAME")
chat_deployment_name = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")
embedding_deployment_name = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
api_key = os.environ.get("AZURE_OPENAI_API_KEY")
api_version = "2023-12-01-preview"
endpoint = f"https://{resource_name}.openai.azure.com"
api_url = f"https://{resource_name}.openai.azure.com/openai/deployments/{chat_deployment_name}/chat/completions?api-version={api_version}"
# Load Neo4j credentials
username = os.environ.get("NEO4J_USERNAME")
password = os.environ.get("NEO4J_PASSWORD")
url = os.environ.get("NEO4J_URI")

print(api_key)

# Creating AzureOpneAI client
client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=endpoint,
)

# Creating Neo4jGraph

graph = Neo4jGraph(url=url, username=username, password=password)

graph_driver = GraphDatabase.driver(
    url, auth=(username, password), max_connection_lifetime=200
)

# Creating vector indexes
vector_index = Neo4jVector.from_existing_graph(
    AzureOpenAIEmbeddings(
        azure_deployment=embedding_deployment_name,
        openai_api_version=api_version,
        azure_endpoint=endpoint,
    ),
    url=url,
    username=username,
    password=password,
    node_label="Part",
    text_node_properties=["title"],
    embedding_node_property="embedding",
)

vector_index = Neo4jVector.from_existing_graph(
    AzureOpenAIEmbeddings(
        azure_deployment=embedding_deployment_name,
        openai_api_version=api_version,
        azure_endpoint=endpoint,
    ),
    url=url,
    username=username,
    password=password,
    node_label="SubPart",
    text_node_properties=["title"],
    embedding_node_property="embedding",
)

vector_index = Neo4jVector.from_existing_graph(
    AzureOpenAIEmbeddings(
        azure_deployment=embedding_deployment_name,
        openai_api_version=api_version,
        azure_endpoint=endpoint,
    ),
    url=url,
    username=username,
    password=password,
    node_label="Section",
    text_node_properties=["title"],
    embedding_node_property="embedding",
)

vector_index = Neo4jVector.from_existing_graph(
    AzureOpenAIEmbeddings(
        azure_deployment=embedding_deployment_name,
        openai_api_version=api_version,
        azure_endpoint=endpoint,
    ),
    url=url,
    username=username,
    password=password,
    node_label="Section_P",
    text_node_properties=["text"],
    embedding_node_property="embedding",
)


entity_types = {
    "Part": "Segment of a subchapter, detailing more specific topics, regulations, agencies and guidelines.",
    "Subpart": "Further division of a part, detailing very specific aspects or regulations and topic of interest.",
    "Section": "The most granular division, often representing specific area in individual regulations or guidelines.",
    "Section_Formula": "Include formulas related to enviornmental regulations mentioned in the corresponding section and includes their explanation in extraction.",
    "Section_P": "Paragraph of texts of the corresponding section, provide most detailed information in regards regulation or guidelines.",
    "Table": "A table of data that is related to the corresponding section.",
}

relation_types = {
    "HAS_SUBPART": "A part contains one or more subparts.",
    "HAS_SECTION": "A subpart contains one or more sections.",
    "HAS_FORMULA": "A section contains one or more formulas.",
    "HAS_P": "A section contains one or more chunks of texts.",
    "HAS_TABLE": "A SubPart contains one or more tables.",
}

entity_relationship_match = {
    "Part": "HAS_SUBPART",
    "Subpart": ["HAS_SECTION", "HAS_TABLE"],
    "Section": ["HAS_FORMULA", "HAS_P"],
}


one_shot_input_prompt = f"""
    You are a helpful agent designed to fetch information from a graph database structured around environmental regulations.

    The graph database organizes regulations into the following entity types:
    {json.dumps(entity_types, indent=0)}

    Each entity is connected through one of the following hierarchical relationships:
    {json.dumps(relation_types, indent=0)}

    Depending on the user prompt, determine if it is possible to answer with the graph database.

    The graph database can navigate through multiple layers of hierarchy to find specific sections of regulations.

    Example user input:
    "We have a continuous emission monitoring system for our acid gas units but no vent meter, what calculation method should we use?"

    There are multiple layers to analyse:
    1. The mention of "continuous emission monitoring system" indicates what subject matter the prompt is asking for.
    2. The mention of "what calculation method" indicates the action we want to perform on the subject matter.
    3. The mention of "acid gas units no vent meter" provides additional conditions to the subject matter to consider.


    Return a json object following these rules:
    For each layer of the hierarchy or specific query parameter mentioned, add a key-value pair with the key being a match for one of the entity types provided, and the value being the relevant detail from the user query.

    For the example provided above, the expected output would be:
    {{
        "subject" : "continuous emission monitoring system",
        "to_do" : "what calculation method",
        "clarification" : "acid gas units no vent meter"
    }}

    If there are no relevant entities or layers in the user prompt, return an empty json object.
"""

three_shot_input_prompt = f"""
    You're assisting users in understanding environmental regulations by querying a structured graph database. 

    The graph database organizes regulations into the following entity types:
    {json.dumps(entity_types, indent=0)}

    Each entity is connected through one of the following hierarchical relationships:
    {json.dumps(relation_types, indent=0)}

    Depending on the user input, determine if it is possible to answer with the graph database.

    The graph database can navigate through multiple layers of hierarchy to find specific sections of regulations.

    We have 3 user inputs. Let's break them down.

    Example user input 1:
    "How does EPA define a 'facility' for petroleum and natural gas systems?"

    Analysis breakdown:
    1. The phrase "for petroleum and natural gas systems" specifies the subject matter of the inquiry.
    2. The question "How does EPA define a 'facility'" indicates the desired action related to the subject matter.
    3. The term "EPA define" provides additional context or conditions to consider.

    Your task is to generate a JSON object structured as follows:
    - "subject": Specify the subject matter identified in the user input.
    - "action_requested": Describe the action the user wants to perform.
    - "clarification": Provide any additional context or conditions mentioned in the user input to narrow down search space and return specific and relevant answer.

    For the given example, the expected output would be:
    {{
        "subject": "for petroleum and natural gas systems",
        "action_requested": "How does EPA define a 'facility'",
        "clarification": "EPA define"
    }}

    Example user input 2:
    "What are my pneumatic device emissions? I have 100 high bleed devices."

    There are multiple layers to analyze:
    1. The mention of "pneumatic device emissions" indicates what subject matter the prompt is asking for.
    2. The mention of "What are" indicates the action we want to perform on the subject matter.
    3. The mention of "100 high bleed devices" provides additional conditions to the subject matter to consider.

    Return a JSON object following these rules:
    For each layer of the hierarchy or specific query parameter mentioned, add a key-value pair with the key being a match for one of the entity types provided, and the value being the relevant detail from the user query.

    For the example provided above, the expected output would be:
    {{
        "subject": "pneumatic device emissions",
        "action_requested": "What are",
        "clarification": "100 high bleed devices"
    }}

    Example user input 3:
    "Which calculation from 98.233(o) should be used for centrifugal compressor venting at onshore petroleum and natural gas production facilities?"

    There are multiple layers to analyze:
    1. The mention of "centrifugal compressor venting at onshore petroleum and natural gas production facilities" indicates what subject matter the prompt is asking for.
    2. The mention of "Which calculation" indicates the action we want to perform on the subject matter.
    3. The mention of " from 98.233(o)" provides additional conditions to the subject matter to consider.

    Return a JSON object following these rules:
    For each layer of the hierarchy or specific query parameter mentioned, add a key-value pair with the key being a match for one of the entity types provided, and the value being the relevant detail from the user query.

    For the example provided above, the expected output would be:
    {{
        "subject": "centrifugal compressor venting at onshore petroleum and natural gas production facilities",
        "action_requested": "Which calculation",
        "clarification": "from 98.233(o)"
    }}

    If there are no relevant entities or layers in the user prompt, return an empty JSON object.
"""


def LLM_input_result(prompt, model, input_prompt, temperature):
    """
    This function defines a query to the Azure OpenAI chat model
    and return its interpretation of the prompt with desired output format.
    """

    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": input_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content


def create_embedding(text):
    """
    This function creates an embedding for a given text using the Azure OpenAI Embedding model.
    """

    result = client.embeddings.create(model=embedding_deployment_name, input=text)
    return result.data[0].embedding


def create_input_prompt_query(text, threshold=0.8):
    """
    This function creates a Cypher query to find matching sections in the graph database
    """

    query_data = json.loads(text)

    # Creating embeddings
    embeddings_data = []
    for key, val in query_data.items():
        embeddings_data.append(f"${key}Embedding AS {key}Embedding")
    query = "WITH " + ",\n".join(e for e in embeddings_data)
    query += "\nMATCH (p:Section_P)"
    # Find matching
    similarity_data = []
    for key, val in query_data.items():
        similarity_data.append(
            f"gds.similarity.cosine(p.embedding, {key}Embedding) > {threshold}"
        )
    query += "\nWHERE "
    query += " OR ".join(e for e in similarity_data)
    query += "\nRETURN p.text, ID(p), p.p_id," + ", ".join(
        f"gds.similarity.cosine(p.embedding, {key}Embedding) AS similarity_score_{key}"
        for key in query_data.keys()
    )
    # print(query)
    return query


def query_database_result(
    prompt,
    model="chat35",
    input_prompt=one_shot_input_prompt,
    temperature=0,
    threshold=0.8,
):
    """
    This function queries the graph database to find matching sections based on the user prompt.
    """

    response = LLM_input_result(
        prompt, model, input_prompt=input_prompt, temperature=temperature
    )
    embeddingsParams = {}
    query = create_input_prompt_query(response, threshold=threshold)
    query_data = json.loads(response)

    for key, val in query_data.items():
        embeddingsParams[f"{key}Embedding"] = create_embedding(val)
    result = graph.query(query, params=embeddingsParams)

    # Sort the chunks based on similarity scores in descending order
    # Dynamically generate the key for sorting based on query_data keys
    result = sorted(
        result,
        key=lambda x: x[f"similarity_score_{list(query_data.keys())[0]}"],
        reverse=True,
    )
    return result, response


def LLM_output_result_config_1(
    result, question, model="chat35", result_limit=80, temperature=0
):
    """
    This function inputs the matched section texts and return its interpretation of the prompt with desired output format.
    """

    """
    Concatenate the chunked text
    """

    result_text = ""
    for res in result[:result_limit]:
        result_text += res["p.text"] + " "

    """
    Getting the formulas mentioned in the sections
    """

    section_ids = [res["ID(p)"] for res in result[:result_limit]]
    section_query = f" MATCH (p:Section_P) WHERE id(p) = {section_ids[0]}"
    for id in section_ids[1:]:
        section_query += f" OR id(p) = {id}"

    section_query += f" MATCH (s:Section)-[:HAS_P]->(p) RETURN ID(s),s.title"
    section_result = graph.query(section_query)
    section_id_search = [res["ID(s)"] for res in section_result]
    section_id_search = list(set(section_id_search))

    formula_query = f"MATCH (s:Section)-[:HAS_FORMULA]->(f:Formula) WHERE ID(s) = {section_id_search[0]}"
    for id in section_id_search[1:]:
        formula_query += f" OR ID(s) = {id}"
    formula_query += " RETURN f.extraction, f.content"
    formula_result = graph.query(formula_query)

    formula_results_final = []
    for res in formula_result[:result_limit]:
        formula_results_final.append(res["f.content"])
        formula_results_final.append(res["f.extraction"])

    formula_results_final = str(formula_results_final)

    section_title_search = [res["s.title"] for res in section_result]
    section_title_search = list(set(section_title_search))
    # print( section_title_search)

    """
    Generating summaries and analysis of the tables
    """

    table_query = f"MATCH (s:SubPart)-[:HAS_TABLE]->(t:Table) WHERE ID(s) = 3156 RETURN t.id,t.content"
    table_result = graph.query(table_query)

    table_results_final = []
    for res in table_result:
        table_results_final.append(res["t.id"])
        table_results_final.append(res["t.content"])

    table_results_final = str(table_results_final)

    """
    Generating summaries and anlysis of the chunked texts
    """

    text_analysis_system_prompt = f"""
    You are an intelligent agent tasked with analyzing information from a graph database on environmental regulations.
    A question has been posed: "{question}"
    Based on the following text extracted from the database, analyze and summarize the content to answer the question with the following steps:
    1. Look for any numbers that are mentioned in the content result_text, they may be key to answering the question.
    2. Look for any enviornmental terms in the result text that closely resembles the question.
    3. Pay attention to any calculation, measures, methods mentioned in result_text if asked in the question, return or calculate with them. 
    If information about the any concept or key points is missing or not clear, do not infer any information.
    Your response should be structured as a JSON object, encapsulating the summary and analysis relevant to the question:
    Expected example output format:
    {{
        "result": "The document mentioned several calucuation method which explains how to derive the emission from the data."
    }}
    """

    text_analysis = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": text_analysis_system_prompt},
            {
                "role": "user",
                "content": result_text,
            },
        ],
    )

    text_analysis_result = text_analysis.choices[0].message.content

    table_analysis_system_prompt = f"""
    You are a helpful agent designed to fetch information from a graph database structured around environmental regulations. 
    Here is the question asked: {question}
    Here is the related analyzed text extracted from the database: {text_analysis_result}
    Given the following input, which includes tables in json format. Understand the content of the table. Find any tables that are helpful to answering the question or to calculating in the question and reconstruct them into a helpful format.
    If information about the table is missing or not clear, do not infer any information.
    Present your analysis in a structured JSON object format under the key "result".
    The expected example output format would be:
    {{
        "result" : "The table describes how many devices should be registered for various categories."
    }}
    """

    table_analysis = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": table_analysis_system_prompt},
            {
                "role": "user",
                "content": table_results_final,
            },
        ],
    )

    table_analysis_result = table_analysis.choices[0].message.content

    """
    Generating summaries and anlysis of the formulas
    """

    formulas_analysis_system_prompt = f"""
    You are a helpful agent designed to fetch information from a graph database structured around environmental regulations. 
    Here is the question asked: {question}
    Here is the related analyzed text extracted from the database: {text_analysis_result}
    Given the following input, which includes LaTeX equations and its extraction or explanation, provide a comprehensive explanation of the LaTeX formulas. 
    If information about the formulas is missing or not clear, do not infer any information.
    Present your analysis in a structured JSON object format under the key "result".
    The expected example output format would be:
    {{
        "result" : "The equation describes E = COUNT * EF ....... The extraction explains that COUNT is the number of units and EF is the emission factor."
    }}
    """

    formulas_analysis = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": formulas_analysis_system_prompt},
            {
                "role": "user",
                "content": formula_results_final,
            },
        ],
    )

    formulas_analysis_result = formulas_analysis.choices[0].message.content

    """
    Ingeration of the results
    """

    intergate_text = (
        json.loads(text_analysis_result)["result"]
        + formulas_analysis_result
        + table_analysis_result
    )

    final_analysis_system_prompt = f"""
    You are a helpful agent designed to fetch information from a graph database structured around environmental regulations. 
    Here is the question asked: {question}
    Given the following input, which includes the analysis of the text, formulas and relevant tables, provide a comprehensive explanation of the givien text to answer the question. 
    Here are some steps to follow:
    1. Look for any numbers that are mentioned in the content result_text, they may be key to answering the question.
    2. Look for any enviornmental terms in the result text that closely resembles the question.
    3. Pay attention to any calculation, measures, methods mentioned in result_text if asked in the question, find relavent formulas or table and return or calculate with them.
    4. If a numerical value is mentioned in the question, try to find the same in the text and calculate with it.

    If there are any information missing or not clear, do not infer any information.
    Output the final analysis in a structured JSON object format under the key "result".
    {{
        "result" : "The document mentioned several calucuation method which explains how to derive the emission from the data. The equation describes E = COUNT * EF ....... The extraction explains that COUNT is the number of units and EF is the emission factor."
    }}
    """

    final_analysis = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": final_analysis_system_prompt},
            {
                "role": "user",
                "content": intergate_text,
            },
        ],
    )

    integrate_response = final_analysis.choices[0].message.content

    return integrate_response, section_title_search, intergate_text


def LLM_output_result_config_2(
    result, question, model="chat35", result_limit=80, temperature=0
):
    """
    This function inputs the matched section texts and return its interpretation of the prompt with desired output format.
    """

    """
    Concatenate the chunked text
    """

    result_text = ""
    for res in result[:result_limit]:
        result_text += res["p.text"] + " "

    """
    Getting the formulas mentioned in the sections
    """

    section_ids = [res["ID(p)"] for res in result[:result_limit]]
    section_query = f" MATCH (p:Section_P) WHERE id(p) = {section_ids[0]}"
    for id in section_ids[1:]:
        section_query += f" OR id(p) = {id}"

    section_query += f" MATCH (s:Section)-[:HAS_P]->(p) RETURN ID(s),s.title"
    section_result = graph.query(section_query)
    section_id_search = [res["ID(s)"] for res in section_result]
    section_id_search = list(set(section_id_search))

    formula_query = f"MATCH (s:Section)-[:HAS_FORMULA]->(f:Formula) WHERE ID(s) = {section_id_search[0]}"
    for id in section_id_search[1:]:
        formula_query += f" OR ID(s) = {id}"
    formula_query += " RETURN f.extraction, f.content"
    formula_result = graph.query(formula_query)

    formula_results_final = []
    for res in formula_result[:result_limit]:
        formula_results_final.append(res["f.content"])
        formula_results_final.append(res["f.extraction"])

    formula_results_final = str(formula_results_final)

    section_title_search = [res["s.title"] for res in section_result]
    section_title_search = list(set(section_title_search))
    # print( section_title_search)

    """
    Generating summaries and analysis of the tables
    """

    table_query = f"MATCH (s:SubPart)-[:HAS_TABLE]->(t:Table) WHERE ID(s) = 3156 RETURN t.id,t.content"
    table_result = graph.query(table_query)

    table_results_final = []
    for res in table_result:
        table_results_final.append(res["t.id"])
        table_results_final.append(res["t.content"])

    table_results_final = str(table_results_final)

    """
    Generating summaries and anlysis of the chunked texts
    """

    text_analysis_system_prompt = f"""
    You are an intelligent agent tasked with analyzing result_text extracted from a graph database on environmental regulations.
    A question has been posed "{question}"
    Analyze the result_text and directly give human-like answer to the question based on the relevant information with the following steps:
    1. Identify any environmental terms or keywords within the content result_text that closely correspond to the question. These terms can guide you towards the relevant information needed for the analysis.
    2. Identify any numbers mentioned in the content result_text. These numbers could be crucial for answering the question. If they are pertinent to the question, ensure to incorporate them into your answer.
    3. Please pay attention to all relevant factors when making your decision. Consider all available options before providing your response. Ensure you analyze all the related information, before making a decision.
    4. If the question involves any calculations, measures, or methods, ensure to address them in your answer.
    5. Additionally, you are required to directly provide a clear and definitive yes or no response based on your analysis to answer the question, if necessary. 

 
    Your response should be structured as a JSON object

    First example question would be:
    "What gases must be reported by oil and natural gas system facilities?"
    
    The answer you are expected to generate should look like:
    {{
        "result": "Summary of Source Types by Industry Segment. Each facility must report:• Carbon Dioxide (CO2) and methane (CH4) emissions from equipment leaks and vented emissions. The table below identifies each source type that industry segments are required to report. For example, natural gas processing facilities must report emissions from seven specific source types, and underground storage must report for five source types.• CO2, CH4, and nitrous oxide (N2O) emissions from gas flares by following the requirements of subpart W.• CO2, CH4, and N2O emissions from stationary and portable fuel combustion sources in the onshore production industry segment following the requirements in subpart W.• CO2, CH4, and N2O emissions from stationary combustion sources in the natural gas distribution industry segment following the requirements in subpart W.• CO2, CH4, and N2O emissions from all other applicable stationary combustion sources following the requirements of 40 CFR 98 subpart C (General Stationary Fuel Combustion Sources)."
    }}
    
    Second example question would be:
    "My question concerns the calculation of standard temperature and pressure.  The rule stipulates what standard temperature and pressure are, but how, for an annual average, is actual temperature and pressure defined?"
    
    The answer you are expected to generate should look like:
    {{
        "result": "Actual temperature and pressure as defined for §98.233 is the “average atmospheric conditions or typical operating conditions." Therefore, the average temperature and pressure at a given location based on annual averages can be used for actual temperature and actual pressure.
    }}
    """

    text_analysis = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": text_analysis_system_prompt},
            {
                "role": "user",
                "content": result_text,
            },
        ],
    )

    text_analysis_result = text_analysis.choices[0].message.content

    """
    Generating summaries and anlysis of the tables
    """

    table_analysis_system_prompt = f"""
    You are a helpful agent designed to analyze the table information provided.
    Here is the question asked: {question}
    Based on your analysis, directly give a human-like answer to the question based on the relevant information.
    If the question involves any calculations, measures, or methods, ensure to address them in your answer.

    Present your analysis in a structured JSON object format under the key "result".

    The example question would be:
    "Are the emissions factors listed in Table W-1A for both leaking components and non-leaking components? How do you calculate emissions from leaking components if onshore petroleum and natural gas source are not required to monitor components?"

    The answer you are expected to generate should look like:
    {{
        "result" : "Equipment leak emissions in onshore production are to be estimated using methods provided in 98.233(r)(2). Hence, no leak detection of emissions is required for onshore production.  Table W-1A provides population emission factors, which represent the emissions on an average from the entire population of components – both leaking and non-leaking; please see section 6(d) of the Technical Support Document (http://www.epa.gov/ghgreporting/documents/pdf/2010/Subpart-W_TSD.pdf) for further details on the concept of population emission factors."
    }}
    """

    table_analysis = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": table_analysis_system_prompt},
            {
                "role": "user",
                "content": table_results_final,
            },
        ],
    )

    table_analysis_result = table_analysis.choices[0].message.content

    """
    Generating summaries and anlysis of the formulas
    """

    formulas_analysis_system_prompt = f"""
    You are a helpful agent designed to analyze the Latex math formula information provided.
    Here is the question asked: {question}
    Based on your analysis, directly give a human-like answer to the question based on the relevant information.
    If the question involves any calculations, measures, or use of math formulas, ensure to address them in your answer.

    Present your analysis in a structured JSON object format under the key "result".

    The example question would be:
    "What are my pneumatic device emissions? I have 100 high bleed devices."

    The answer you are expected to generate should look like:
    {{
        "result" : "To calculate CH4 and CO2 volumetric emissions from natural gas driven pneumatic pump venting, we need to use Equation W-2 of §98.233 where Es,i = Annual total volumetric GHG emissions at standard conditions in standard cubic feet per year from all natural gas driven pneumatic pump venting, for GHGi. Count = Total number of natural gas driven pneumatic pumps. EF = Population emissions factors for natural gas driven pneumatic pumps (in standard cubic feet per hour per pump) listed in Table W-1A of this subpart for onshore petroleum and natural gas production and onshore petroleum and natural gas gathering and boosting facilities. GHGi = Concentration of GHGi, CH4, or CO2, in produced natural gas as defined in paragraph (u)(2)(i) of this section. T = Average estimated number of hours in the operating year the pumps were operational using engineering estimates based on best available data. Default is 8,760 hours.The the following assumptions will be made to complete the calculation: 1) T will be the default value 86 hours. 2) The concentration of GHGi is 95% for CH4 and 1% for CO2. EF for High Continuous Bleed Pneumatic Device Vents is 37.3 scf/hour/component. Your pneumatic device emissions would be 31,041,060 scf CH4 which is calculated as 100 * 37.3 scf / hr / device * 0.95 * 8760 hours and 326,748 scf CO2 which is calculated as 100 * 37.3 scf / hr / device * 0.01 * 8760 hours."
    }}
    """

    formulas_analysis = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": formulas_analysis_system_prompt},
            {
                "role": "user",
                "content": formula_results_final,
            },
        ],
    )

    formulas_analysis_result = formulas_analysis.choices[0].message.content

    """
    Ingeration of the results
    """

    intergate_text = (
        json.loads(text_analysis_result)["result"]
        + formulas_analysis_result
        + table_analysis_result
    )

    final_analysis_system_prompt = f"""
    You are an intelligent agent tasked with integrating the answers from text analysis, table analysis, and formula analysis to generate a final cohesive answer.
    The question posed is: {question}
    Below are the individual analyses:

    Text Analysis Result:
    {text_analysis_result}

    Table Analysis Result:
    {table_analysis_result}
    
    Formula Analysis Result:
    {formulas_analysis_result}

    Please generate a final cohesive answer integrating the information from all the analyses performed.
    Ensure that your response is structured as a JSON object under the key "result".
    
    """

    final_analysis = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": final_analysis_system_prompt},
            {
                "role": "user",
                "content": intergate_text,
            },
        ],
    )

    integrate_response = final_analysis.choices[0].message.content

    return integrate_response, section_title_search, intergate_text


def LLM_output_result_config_3(
    result, question, model="chat35", result_limit=80, temperature=0
):
    """
    This function inputs the matched section texts and return its interpretation of the prompt with desired output format.
    """

    """
    Concatenate the chunked text
    """

    result_text = ""
    for res in result[:result_limit]:
        result_text += res["p.text"] + " "

    """
    Getting the formulas mentioned in the sections
    """

    section_ids = [res["ID(p)"] for res in result[:result_limit]]
    section_query = f" MATCH (p:Section_P) WHERE id(p) = {section_ids[0]}"
    for id in section_ids[1:]:
        section_query += f" OR id(p) = {id}"

    section_query += f" MATCH (s:Section)-[:HAS_P]->(p) RETURN ID(s),s.title"
    section_result = graph.query(section_query)
    section_id_search = [res["ID(s)"] for res in section_result]
    section_id_search = list(set(section_id_search))

    formula_query = f"MATCH (s:Section)-[:HAS_FORMULA]->(f:Formula) WHERE ID(s) = {section_id_search[0]}"
    for id in section_id_search[1:]:
        formula_query += f" OR ID(s) = {id}"
    formula_query += " RETURN f.extraction, f.content"
    formula_result = graph.query(formula_query)

    formula_results_final = []
    for res in formula_result[:result_limit]:
        formula_results_final.append(res["f.content"])
        formula_results_final.append(res["f.extraction"])

    formula_results_final = str(formula_results_final)

    section_title_search = [res["s.title"] for res in section_result]
    section_title_search = list(set(section_title_search))
    # print( section_title_search)

    """
    Generating summaries and analysis of the tables
    """

    table_query = f"MATCH (s:SubPart)-[:HAS_TABLE]->(t:Table) WHERE ID(s) = 3156 RETURN t.id,t.content"
    table_result = graph.query(table_query)

    table_results_final = []
    for res in table_result:
        table_results_final.append(res["t.id"])
        table_results_final.append(res["t.content"])

    table_results_final = str(table_results_final)

    # System prompt for defining analysis
    system_prompt = f"""
    You are an intelligent agent tasked with determining the type of analysis to perform based on the user question.
    There are three analysis types: Text, Table and Formula. Text analysis is the default analysis that must be performed. If the question involves calculations, equations or directly mention which tables or equations to look at, it may require referencing tables and formulas provided.
    Determine the appropriate analysis type(s) to perform based on {question}. 
    
    Your task is to generate a JSON object structured as follows:
    - "analysis_types": Specify the type(s) of analysis to be performed.
    
    Example user input 1:
    "How does EPA define a 'facility' for petroleum and natural gas systems?"

    For the given example, the expected output would be:
    {{
        "analysis_types": ["Text"]
    }}

    Example user input 2:
    "For liquefied natural gas facility equipment that is in gas service, is only the equipment listed in Table W-5 (Vapor Recovery Compressor) required to be reported if it is found to be leaking as defined in the rule?"

    For the given example, the expected output would be:
    {{
        "analysis_types": ["Text", "Table"]
    }}


    Example user input 3:
    "We have a CEMS for our acid gas units but no vent meter, what calculation method should we use?"

    For the given example, the expected output would be:
    {{
        "analysis_types": ["Text", "Formula"]
    }}

    If there are no relevant entities in the user prompt, return an empty json object.    
    """

    # Create completion to determine analysis type
    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )

    # Extract the determined analysis type
    analysis_type = completion.choices[0].message.content
    # print(analysis_type)

    """
    Generating summaries and anlysis of the chunked texts
    """

    text_analysis_system_prompt = f"""
    You are an intelligent agent tasked with analyzing "{result_text}" extracted from a graph database on environmental regulations.
    A question has been posed "{question}"
    Analyze the "{result_text}" and directly give human-like answer to the question based on the relevant information with the following steps:
    1. Identify any environmental terms or keywords within the content "{result_text}" that closely correspond to the question. These terms can guide you towards the relevant information needed for the analysis.
    2. Identify any numbers mentioned in the content "{result_text}". These numbers could be crucial for answering the question. If they are pertinent to the question, ensure to incorporate them into your answer.
    3. Please pay attention to all relevant factors when making your decision. Consider all available options before providing your response. Ensure you analyze all the related information, before making a decision.
    4. If the question involves any calculations, measures, or methods, ensure to address them in your answer.
    5. Additionally, you are required to directly provide a clear and definitive yes or no response based on your analysis to answer the question, if necessary. 

 
    Your response should be structured as a JSON object

    First example question would be:
    "What gases must be reported by oil and natural gas system facilities?"
    
    The answer you are expected to generate should look like:
    {{
        "result": "Summary of Source Types by Industry Segment. Each facility must report:• Carbon Dioxide (CO2) and methane (CH4) emissions from equipment leaks and vented emissions. The table below identifies each source type that industry segments are required to report. For example, natural gas processing facilities must report emissions from seven specific source types, and underground storage must report for five source types.• CO2, CH4, and nitrous oxide (N2O) emissions from gas flares by following the requirements of subpart W.• CO2, CH4, and N2O emissions from stationary and portable fuel combustion sources in the onshore production industry segment following the requirements in subpart W.• CO2, CH4, and N2O emissions from stationary combustion sources in the natural gas distribution industry segment following the requirements in subpart W.• CO2, CH4, and N2O emissions from all other applicable stationary combustion sources following the requirements of 40 CFR 98 subpart C (General Stationary Fuel Combustion Sources)."
    }}
    
    Second example question would be:
    "My question concerns the calculation of standard temperature and pressure.  The rule stipulates what standard temperature and pressure are, but how, for an annual average, is actual temperature and pressure defined?"
    
    The answer you are expected to generate should look like:
    {{
        "result": "Actual temperature and pressure as defined for §98.233 is the “average atmospheric conditions or typical operating conditions.”  Therefore, the average temperature and pressure at a given location based on annual averages can be used for actual temperature and actual pressure.
    }}
    """

    text_analysis = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": text_analysis_system_prompt},
            {
                "role": "user",
                "content": result_text,
            },
        ],
    )

    text_analysis_result = text_analysis.choices[0].message.content

    """
    Generating summaries and anlysis of the tables
    """

    table_analysis_result = ""  # initialize table result
    if "Table" in analysis_type:
        table_analysis_system_prompt = f"""
        You are a helpful agent designed to analyze the table information provided.
        Here is the question asked: {question}
        Based on your analysis, directly give a human-like answer to the question based on the relevant information.
        If the question involves any calculations, measures, or methods, ensure to address them in your answer.

        Present your analysis in a structured JSON object format under the key "result".

        The example question would be:
        "Are the emissions factors listed in Table W-1A for both leaking components and non-leaking components? How do you calculate emissions from leaking components if onshore petroleum and natural gas source are not required to monitor components?"

        The answer you are expected to generate should look like:
        {{
            "result" : "Equipment leak emissions in onshore production are to be estimated using methods provided in 98.233(r)(2).  Hence, no leak detection of emissions is required for onshore production.  Table W-1A provides population emission factors, which represent the emissions on an average from the entire population of components – both leaking and non-leaking; please see section 6(d) of the Technical Support Document (http://www.epa.gov/ghgreporting/documents/pdf/2010/Subpart-W_TSD.pdf) for further details on the concept of population emission factors."
        }}
        """

        table_analysis = client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": table_analysis_system_prompt},
                {
                    "role": "user",
                    "content": table_results_final,
                },
            ],
        )

        table_analysis_result = table_analysis.choices[0].message.content

    """
    Generating summaries and anlysis of the formulas
    """

    formulas_analysis_result = ""  # initialize formula analysis result
    if "Formula" in analysis_type:
        formulas_analysis_system_prompt = f"""
        You are a helpful agent designed to analyze the Latex math formula information provided.
        Here is the question asked: {question}
        Based on your analysis, directly give a human-like answer to the question based on the relevant information.
        If the question involves any calculations, measures, or use of math formulas, ensure to address them in your answer.

        Present your analysis in a structured JSON object format under the key "result".

        The example question would be:
        "What are my pneumatic device emissions? I have 100 high bleed devices."

        The answer you are expected to generate should look like:
        {{
            "result" : "To calculate CH4 and CO2 volumetric emissions from natural gas driven pneumatic pump venting , we need to use Equation W-2 of §98.233 where Es,i = Annual total volumetric GHG emissions at standard conditions in standard cubic feet per year from all natural gas driven pneumatic pump venting, for GHGi. Count = Total number of natural gas driven pneumatic pumps. EF = Population emissions factors for natural gas driven pneumatic pumps (in standard cubic feet per hour per pump) listed in Table W-1A of this subpart for onshore petroleum and natural gas production and onshore petroleum and natural gas gathering and boosting facilities. GHGi = Concentration of GHGi, CH4, or CO2, in produced natural gas as defined in paragraph (u)(2)(i) of this section. T = Average estimated number of hours in the operating year the pumps were operational using engineering estimates based on best available data. Default is 8,760 hours.The the following assumptions will be made to complete the calculation: 1) T will be the default value 86 hours. 2) The concentration of GHGi is 95% for CH4 and 1% for CO2. EF for High Continuous Bleed Pneumatic Device Vents is 37.3 scf/hour/component. Your pneumatic device emissions would be 31,041,060 scf CH4 which is calculated as 100 * 37.3 scf / hr / device * 0.95 * 8760 hours and 326,748 scf CO2 which is calculated as 100 * 37.3 scf / hr / device * 0.01 * 8760 hours."
        }}
        """

        formulas_analysis = client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": formulas_analysis_system_prompt},
                {
                    "role": "user",
                    "content": formula_results_final,
                },
            ],
        )

        formulas_analysis_result = formulas_analysis.choices[0].message.content

    """
    Ingeration of the results
    """

    intergate_text = (
        json.loads(text_analysis_result)["result"]
        + formulas_analysis_result
        + table_analysis_result
    )

    final_analysis_system_prompt = f"""
    You are an intelligent agent tasked with integrating the answers from text analysis, table analysis, and formula analysis to generate a final cohesive answer.
    The question posed is: {question}
    Below are the individual analyses:

    Text Analysis Result:
    {text_analysis_result}

    Table Analysis Result:
    {table_analysis_result}
    
    Formula Analysis Result:
    {formulas_analysis_result}

    Please generate a final cohesive answer integrating the information from all the analyses performed.
    Ensure that your response is structured as a JSON object under the key "result".
    
    """

    final_analysis = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": final_analysis_system_prompt},
        ],
    )

    final_analysis_result = final_analysis.choices[0].message.content

    return final_analysis_result, section_title_search, intergate_text


def chat_main(
    question,
    model="chat35",
    threshold=0.8,
    input_prompt=3,
    result_limit=50,
    temperature=0,
    output_config=2,
    exp_mode=0,
):
    """
    This function is the main function to run the chatbot.
    """
    if exp_mode == 0:
        if input_prompt == 1:
            result, response = query_database_result(
                question,
                model=model,
                input_prompt=one_shot_input_prompt,
                temperature=temperature,
            )
            print(
                f"\nParameter: \nQuestion: {question}\nModel: {model}, Threshold: {threshold}, Result_limit: {result_limit}, Temperature: {temperature}, output_config: {output_config}\n"
            )
            print(f"Found {len(result)} matching paragraphs\n")
            if output_config == 1:
                inter_result, section_titles, context = LLM_output_result_config_1(
                    result,
                    question,
                    model=model,
                    result_limit=result_limit,
                    temperature=temperature,
                )
            if output_config == 2:
                inter_result, section_titles, context = LLM_output_result_config_2(
                    result,
                    question,
                    model=model,
                    result_limit=result_limit,
                    temperature=temperature,
                )
            if output_config == 3:
                inter_result, section_titles, context = LLM_output_result_config_3(
                    result,
                    question,
                    model=model,
                    result_limit=result_limit,
                    temperature=temperature,
                )
            pprint.pp(("Analyzed Results:", json.loads(inter_result)["result"]))
            print("\nFollowing Sections are referenced in the analysis:")
            for title in section_titles:
                print(title.strip())
            # return json.loads(inter_result)['result'],section_titles,context
            return json.loads(inter_result)
        if input_prompt == 3:
            result, response = query_database_result(
                question,
                model=model,
                input_prompt=three_shot_input_prompt,
                temperature=temperature,
            )
            print(
                f"\nParameter: \nQuestion: {question}\nModel: {model}, Threshold: {threshold}, Result_limit: {result_limit}, Temperature: {temperature}, output_config: {output_config}\n"
            )
            print(f"Found {len(result)} matching paragraphs\n")
            if output_config == 1:
                inter_result, section_titles, context = LLM_output_result_config_1(
                    result,
                    question,
                    model=model,
                    result_limit=result_limit,
                    temperature=temperature,
                )
            if output_config == 2:
                inter_result, section_titles, context = LLM_output_result_config_2(
                    result,
                    question,
                    model=model,
                    result_limit=result_limit,
                    temperature=temperature,
                )
            if output_config == 3:
                inter_result, section_titles, context = LLM_output_result_config_3(
                    result,
                    question,
                    model=model,
                    result_limit=result_limit,
                    temperature=temperature,
                )
            pprint.pp(("Analyzed Results:", json.loads(inter_result)["result"]))
            print("\nFollowing Sections are referenced in the analysis:")
            for title in section_titles:
                print(title.strip())
            # return json.loads(inter_result)['result'],section_titles,context
            return json.loads(inter_result)
    if exp_mode == 1:
        if input_prompt == 1:
            result, response = query_database_result(
                question,
                model=model,
                input_prompt=one_shot_input_prompt,
                temperature=temperature,
            )
            if output_config == 1:
                inter_result, section_titles, context = LLM_output_result_config_1(
                    result,
                    question,
                    model=model,
                    result_limit=result_limit,
                    temperature=temperature,
                )
            if output_config == 2:
                inter_result, section_titles, context = LLM_output_result_config_2(
                    result,
                    question,
                    model=model,
                    result_limit=result_limit,
                    temperature=temperature,
                )
            if output_config == 3:
                inter_result, section_titles, context = LLM_output_result_config_3(
                    result,
                    question,
                    model=model,
                    result_limit=result_limit,
                    temperature=temperature,
                )
            return json.loads(inter_result)["result"], section_titles, context
        if input_prompt == 3:
            result, response = query_database_result(
                question,
                model=model,
                input_prompt=three_shot_input_prompt,
                temperature=temperature,
            )
            if output_config == 1:
                inter_result, section_titles, context = LLM_output_result_config_1(
                    result,
                    question,
                    model=model,
                    result_limit=result_limit,
                    temperature=temperature,
                )
            if output_config == 2:
                inter_result, section_titles, context = LLM_output_result_config_2(
                    result,
                    question,
                    model=model,
                    result_limit=result_limit,
                    temperature=temperature,
                )
            if output_config == 3:
                inter_result, section_titles, context = LLM_output_result_config_3(
                    result,
                    question,
                    model=model,
                    result_limit=result_limit,
                    temperature=temperature,
                )
            return json.loads(inter_result)["result"], section_titles, context


# Input the example prompt to test the chatbot

## Example prompt:
## Do I have to count the number of pneumatic devices we have?
## What are my pneumatic device emissions? I have 100 high bleed devices.
## What is the emission reporting limit?
## What is the emission factor for an intermittent bleed device?
## We have a CEMS for our acid gas units but no vent meter, what calculation method should we use?
## How many calculation methods are available for G&B storage tanks?


model = sys.argv[1]
threshold = float(sys.argv[2])
input_prompt = int(sys.argv[3])
result_limit = int(sys.argv[4])
temperature = float(sys.argv[5])
output_config = int(sys.argv[6])
exp_mode = int(sys.argv[7])


try:
    model = sys.argv[1]
    threshold = float(sys.argv[2])
    input_prompt = int(sys.argv[3])
    result_limit = int(sys.argv[4])
    temperature = float(sys.argv[5])
    output_config = int(sys.argv[6])
    exp_mode = int(sys.argv[7])

    with open("questions.txt") as f:
        questions = f.readlines()
        for question in questions:
            question = question.strip()
            chat_result = chat_main(
                question,
                model=model,
                threshold=threshold,
                input_prompt=input_prompt,
                result_limit=result_limit,
                temperature=temperature,
                output_config=output_config,
                exp_mode=exp_mode,
            )

except:
    print(
        " Example Usage: python chat.py model threshold input_prompt result_limit temperature output_config exp_mode"
    )
