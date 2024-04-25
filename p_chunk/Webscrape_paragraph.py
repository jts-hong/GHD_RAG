import pandas as pd
import requests
import re
from bs4 import BeautifulSoup as bs
from neo4j import GraphDatabase


from PIL import Image
from pix2tex.cli import LatexOCR
import requests
from io import BytesIO

import os
import dotenv
import json
from openai import AzureOpenAI
from tqdm import tqdm

#Use env to get your username and password
username = "your username here"
password = "your password here"

#Data import
xml_bs = requests.get('https://www.govinfo.gov/bulkdata/ECFR/title-40/ECFR-title40.xml')
soup = bs(xml_bs.content, 'xml')

#Set up function for formula translation
def get_formula(formula):
    url_name = 'https://img.federalregister.gov/' + formula + '/' + formula + '_original_size.png'
    response = requests.get(url_name)
    img = Image.open(BytesIO(response.content))
    model = LatexOCR()
    inter = model(img)
    return(str(inter))

#Scarpe texts s
list_of_dicts_bs = []  #creating an empty list that I will iteratively append results to               

parts_bs = soup.find_all('DIV5')                #finding & saving all DIV5 elements in subchapter_bs
for part_bs in parts_bs:                                 #looping through the parts_bs object
    if part_bs.attrs['N']=="98":     
        part_num_bs = part_bs.attrs['N']                     #finding & saving all DIV5 attributes that are N
        part_title_bs = part_bs.find('HEAD').text            #finding & saving the text of the HEAD tags

        subparts_bs = part_bs.find_all('DIV6')             #finding & saving all DIV6 elements in subchapter_bs
        for subpart_bs in subparts_bs:                                 #looping through the parts_bs object
            subpart_num_bs = subpart_bs.attrs['N']                     #finding & saving all DIV6 attributes that are N
            subpart_title_bs = subpart_bs.find('HEAD').text            #finding & saving the text of the HEAD tags
            
            sections_bs = subpart_bs.find_all('DIV8')               #finding & saving all DIV8 elements in the part_bs object 
            for section_bs in sections_bs:                       #looping through the sections_bs object
                section_num_bs = section_bs.attrs['N'][2:]       #finding & saving all DIV8 attributes that are N
                section_title_bs = section_bs.find('HEAD').text  #finding & saving the text of the HEAD tags
                section_text_bs = section_bs.find_all('P')       #finding & saving all the content of P tags
                
                j = 1
                for section_text in section_text_bs:
                    section_text_str = str(section_text)
                    section_text_str = section_text_str.strip() #looping through my columns to strip any leading/trailing whitespace
                    regex_bs = "\[+|\]+|<[A-Z]+>+|<\/[A-Z]+>+|\\n+"  #regular expression that matches on xml tags/ASCII characters
                    section_text_str = section_text_str.replace(regex_bs, '')  #replacing any matches on regex_bs with nothing
                    
                    list_of_dicts_bs.append({ 
                                      'part': part_num_bs, 'part_title': part_title_bs, 
                        'subpart': subpart_num_bs, 'subpart_title': subpart_title_bs,  
                        'section': section_num_bs, 'section_title': section_title_bs,
                        'section_p':str(j),'section_text':section_text_str
                    })
                    j+=1
                    #appending my results to a dictionary at the paragraph-level
                    
#Set up text dataframe and preprocess using regex
df_bs = pd.DataFrame(data=list_of_dicts_bs, 
                     columns=[
                              'part', 'part_title','subpart','subpart_title',
                              'section', 'section_title', 
                              'section_p', 'section_text'])

regex_bs = "\[+|\]+|<[A-Z]+>+|<\/[A-Z]+>+|\\n+"  #regular expression that matches on xml tags/ASCII characters
df_bs.section_text = df_bs.section_text.str.replace(regex_bs, '')  #replacing any matches on regex_bs with nothing 
df_bs.section_title = df_bs.section_title.str.replace('§', '')

#get tables
lists_of_tables = []
parts_bs = soup.find_all('DIV5')                #finding & saving all DIV5 elements in subchapter_bs
for part_bs in parts_bs:                                 #looping through the parts_bs object
    if part_bs.attrs['N']=="98":     
        part_num_bs = part_bs.attrs['N']                     #finding & saving all DIV5 attributes that are N
        part_title_bs = part_bs.find('HEAD').text            #finding & saving the text of the HEAD tags
        
        subparts_bs = part_bs.find_all('DIV6')             #finding & saving all DIV6 elements in subchapter_bs
        for subpart_bs in subparts_bs:                                 #looping through the parts_bs object
            subpart_num_bs = subpart_bs.attrs['N']                     #finding & saving all DIV6 attributes that are N
            subpart_title_bs = subpart_bs.find('HEAD').text            #finding & saving the text of the HEAD tags
            
            subpart_tables = subpart_bs.find_all('DIV9')  #Finding ad saving all the tables in subpart DIV9
            for subpart_table in subpart_tables:
                table_title = subpart_table.find('HEAD').text
                content = str(subpart_table.find('TABLE'))
                
                lists_of_tables.append({ 
                                      'part': part_num_bs, 'part_title': part_title_bs, 
                        'subpart': subpart_num_bs, 'subpart_title': subpart_title_bs, 'table_title':table_title, 'content':content
                    })

df_table = pd.DataFrame(data=lists_of_tables, 
                        columns = [
                              'part', 'part_title','subpart','subpart_title', 'table_title','content'])

#Set up GPT call to translate table from XML format to Json
# Load environment variables
dotenv.load_dotenv("GHD_cred.env", override=True)

# Load Azure OpenAI credentials
resource_name = os.environ.get("AZURE_RESOURCE_NAME")
chat_deployment_name = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")
embedding_deployment_name = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
api_key = os.environ.get("AZURE_OPENAI_API_KEY")
api_version = "2023-12-01-preview"
endpoint = f"https://{resource_name}.openai.azure.com"
api_url = f"https://{resource_name}.openai.azure.com/openai/deployments/{chat_deployment_name}/chat/completions?api-version={api_version}"

# Creating AzureOpneAI client
client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=endpoint,
)

df_table_clean = df_table.copy()
table_clean = list()
for i, item in tqdm(df_table.iterrows()):
    
    #print(item['content'], len(item['content']))
    table_analysis_system_prompt = f"""
    You are a helpful agent to reconstruct tables from XML 
    Given the following input, which includes XML of tables, provide a hollistic reconstruction of the table.
    Present your result in a structured JSON object format under the key "result".
    The expected example output format would be:
    {{
        "result" : results lies here
    }}
    """
    try:
        table_analysis = client.chat.completions.create(
            model="chat35",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": table_analysis_system_prompt},
                {
                    "role": "user",
                    "content": item['content'],
                },
            ],
        )
        table_result = table_analysis.choices[0].message.content
    except:
        table_result = item['content']
    
    table_clean.append(table_result)

df_table_clean['content'] = table_clean
df_table_clean.drop(0, inplace = True)

#get formulas
list_of_formulas = []  #creating an empty list that I will iteratively append results to               

parts_bs = soup.find_all('DIV5')                #finding & saving all DIV5 elements in subchapter_bs
for part_bs in parts_bs:                                 #looping through the parts_bs object
    if part_bs.attrs['N']=="98":     
        subparts_bs = part_bs.find_all('DIV6')             #finding & saving all DIV6 elements in subchapter_bs
        for subpart_bs in subparts_bs:                                 #looping through the parts_bs object
            sections_bs = subpart_bs.find_all('DIV8')               #finding & saving all DIV8 elements in the part_bs object 
            for section_bs in sections_bs:                       #looping through the sections_bs object
                section_num_bs = section_bs.attrs['N'][2:]       #finding & saving all DIV8 attributes that are N
                section_title_bs = section_bs.find('HEAD').text  #finding & saving the text of the HEAD tags
                section_text_bs = section_bs.find_all('P')       #finding & saving all the content of P tags
                
                section_img_bs = section_bs.find_all('img')     #finding images 
                img_list = []
                
                for img in section_img_bs:
                    i = img.get('src')
                    i = i.replace('/graphics/', '')
                    i = i.replace('.img', '')
                    i = i.strip("'")
                    #https://img.federalregister.gov/ER25NO14.060/ER25NO14.060_original_size.png
                    img_temp = get_formula(i)
                    img_list.append(img_temp)
                
                section_img_ex = section_bs.find_all('EXTRACT') #finding extracts
                extract_list = []
                for ex in section_img_ex:
                    section_FP2 = ex.find_all('FP-2')
                    for fp2 in section_FP2:
                        # Use .get_text() to extract all text content, strip() to remove leading/trailing whitespaces
                        text = fp2.get_text(strip=True)
                        # Replace occurrences of 'C' and 'T' with 'Prod C' and 'Prod T' if needed
                        text = re.sub(r'<E T="52">(.*?)</E>', r'`\1`', text)
                        extract_list.append(text)
                img_ex_pair = list(zip(img_list, extract_list))
                
                for i,j in img_ex_pair:
                    list_of_formulas.append({  
                        'section': section_num_bs, 'section_title': section_title_bs,
                        'section_img':i, 'section_ex':j
                    })

df_formula = pd.DataFrame(data=list_of_formulas, 
                        columns = [
                              'section','section_title','section_img','section_ex'])

df_formula.section_title = df_bs.section_title.str.replace('§', '')

#Upload to Neo4j 
#For each P tag in section
# Assuming df is your DataFrame

# Connect to Neo4j
uri = "your online neo4j database uri" 
driver = GraphDatabase.driver(uri, auth=(username, password))

def create_graph(tx, part_title, subpart_title, section_title, section_p, section_text):
    query = (
        "MERGE (pt:Part {title: $part_title}) "
        "MERGE (subpt:SubPart {title: $subpart_title})"
        "MERGE (sec:Section {title: $section_title})"
        "MERGE (sec_p:Section_P {p_id: $section_p, text: $section_text})"
        "MERGE (pt)-[:HAS_SUBPART]->(subpt)"
        "MERGE (subpt)-[:HAS_SECTION]->(sec)"
        "MERGE (sec)-[:HAS_P]->(sec_p)"
    )
    tx.run(query, part_title=part_title, subpart_title=subpart_title, 
            section_title=section_title, section_p=section_p, section_text=section_text)


#with driver.session() as session:
#    for index, row in tqdm(df_bs.iterrows()):
#        session.write_transaction(create_graph, row['chapter_title'], row['subchapter_title'], row['part_title'], row['subpart_title']
#                                row['section'], row['section_title'], row['section_text'], row['section_img'], row['section_ex'])
with driver.session() as session:
    # Wrap df_bs.iterrows() with tqdm for a progress bar
    for index, row in tqdm(df_bs.iterrows(), total=df_bs.shape[0]):
        session.write_transaction(create_graph, row['part_title'], row['subpart_title'],
                                 row['section_title'], row['section_p'], row['section_text'])

driver.close()

driver = GraphDatabase.driver(uri, auth=(username, password))

def create_and_link_manufacturer(tx, subpart_title, table_title, content):
        query = (
        "MERGE (table:Table {id: $table_title}) "  # Use table_title as the unique identifier
        "ON CREATE SET table.content = $content "
        "WITH table "
        "MATCH (subpart:SubPart {title: $subpart_title}) "  # Match SubPart using subpart_title
        "MERGE (subpart)-[:HAS_TABLE]->(table)"  # Create the relationship
    )
        tx.run(query, subpart_title=subpart_title, table_title=table_title, content=content)

with driver.session() as session:
    # Wrap df_bs.iterrows() with tqdm for a progress bar
    for index, row in tqdm(df_table_clean.iterrows(), total=df_table_clean.shape[0]):
        session.write_transaction(create_and_link_manufacturer, row['subpart_title'],
                                 row['table_title'], row['content'])

driver.close()

driver = GraphDatabase.driver(uri, auth=(username, password))

def create_and_link_manufacturer(tx, section_title, section_img, section_ex):
        query = (
        "MERGE (formula:Formula {extraction: $section_ex}) "  # Use table_title as the unique identifier
        "ON CREATE SET formula.content = $section_img "
        "WITH formula "
        "MATCH (section:Section {title: $section_title}) "  # Match SubPart using subpart_title
        "MERGE (section)-[:HAS_FORMULA]->(formula)"  # Create the relationship
    )
        tx.run(query, section_title=section_title, section_img=section_img, section_ex=section_ex)

with driver.session() as session:
    # Wrap df_bs.iterrows() with tqdm for a progress bar
    for index, row in tqdm(df_formula.iterrows(), total=df_formula.shape[0]):
        session.write_transaction(create_and_link_manufacturer, row['section_title'],
                                 row['section_img'], row['section_ex'])

driver.close()
