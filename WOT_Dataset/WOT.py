import datasets
import re
import requests
import json
import pandas as pd
import random
from bs4 import BeautifulSoup

class WizardOfTasksConfig(datasets.BuilderConfig):
    """BuilderConfig for Wizard of Tasks."""

    def __init__(self, dataset_type='general', prev_utterances=4, description='',  **kwargs):
        super(WizardOfTasksConfig, self).__init__(**kwargs)
        
        if dataset_type == 'general' and prev_utterances==0:
            self.name = dataset_type
        else:
            self.name = f'{dataset_type}_{prev_utterances}'
        self.dataset_type = dataset_type
        self.prev_utterances = 4
        self.description = description
        
class WOT(datasets.GeneratorBasedBuilder):
    DEFAULT_WRITER_BATCH_SIZE = 256
    BUILDER_CONFIGS = [WizardOfTasksConfig(dataset_type="general", prev_utterances=0, description="General Dataset"),
                       WizardOfTasksConfig(dataset_type="qa", prev_utterances=0, description="Dataset used for QA, without previous utterances"),
                       WizardOfTasksConfig(dataset_type="qa", prev_utterances=4, description="Dataset used for QA, with previous 4 utterances"),
                       WizardOfTasksConfig(dataset_type="qa_simple", prev_utterances=0, description="Dataset used for QA, with generate ingredient questions")]


    def _info(self):
        description = f'Wizard of Tasks dataset for {self.config.dataset_type} purpose',
        if self.config.dataset_type == 'general':
            return datasets.DatasetInfo(
            description = description,
            features=datasets.Features(
                {"conversation": datasets.Value("string"),
                 "text": datasets.Value("string"),
                 "turn": datasets.Value("int32"),
                 "dangerous_tools": datasets.Value("string"),
                 "shared_data": datasets.Value("string"),
                 "relevant": datasets.Value("string"),
                 "useful": datasets.Value("string"),
                 "role": datasets.Value("string"),
                 "theme":datasets.Value("string"),
                 "intent":datasets.Value("string"),
                 "context_title": datasets.Value("string"),
                 "context_description": datasets.Value("string"),
                 "context_steps": datasets.Value("string")}
            ),
            supervised_keys=("file", "label")
        )
        else:
            return datasets.DatasetInfo(
            description = description,
            features=datasets.Features(
                {"input": datasets.Value("string"),
                 "output": datasets.Value("string"),
                 "theme":datasets.Value("string")}
            ),
            supervised_keys=("input", "output")
        )
            
    def _split_generators(self, dl_manager: datasets.utils.download_manager.DownloadManager):
        self.prepare_data()
        
        urls = {'train': f'.train.json',
                'test': f'.test.json',
                'dev': f'.dev.json'}
        
        downloaded_files = dl_manager.download(urls)
        
        return [
        datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
        datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]})
        ]
        
    
    def _generate_examples(self, filepath):
        """Generate examples from a Crema unzipped directory."""
        id = 0
        with open(filepath) as f:
            for i, conversation in enumerate(f):
                
                conv = json.loads(conversation)
                
                conversation_id = conv['conversation_id']
                
                if 'wikihow' in conv['document_url']:
                    theme = 'diy'
                else:
                    theme = 'cooking'
            
                context = self.get_context(conv['document_url'])
                if context == None:
                    continue
                
                if self.config.dataset_type == 'qa_simple':
                    if theme == 'cooking':
                        for ingredient in context['ingredients']:
                            ingre = ingredient.split(',')[0].split(' ')
                            
                            q = ''
                            a = ''
                            if len(ingre) == 2:
                                
                                r = random.randint(0, 2)
                                if r == 0:
                                    q = f'How many  {ingre[1]} should I get?'
                                elif r == 1:
                                    q = f'How many {ingre[1]} do I need?'
                                else:
                                    q = f'How many {ingre[1]} should I use?'
                                a = ingre[0]
                            
                            elif len(ingre) > 1 and ingre[1] in ['(7.0-ounce)', '(8.0-ounce)','(14.0-ounce)','(15.0-ounce)', '(20.0-ounce)' ,'(12.0-ounce)', '(9-inch)', '(28.0-ounce)', 'piece', '(6-ounce)', 'whole', 'ounce', 'ounces', 'cloves', 'small', 'large', 'cup', 'cups', 'tablespoon', 'tablespoons', 'teaspoon', 'teaspoons', 'pound','pounds', 'gram','grams', 'g', 'gr', '(750.0-ml)', '(500.0-ml)', '(26.0-ounce)','(8.0-ounce)', '(6.0-ounce)', '(9.0-ounce)']:
                                r = random.randint(0,1)
                                if r == 0:
                                    q = f'How much {" ".join(ingre[2:])}, do I need?'
                                else:
                                    q = f'How much {" ".join(ingre[2:])}, should I use?'
                                a = ' '.join(ingre[:2])
                                
                            inp = self.generate_input_qa(q, context, '')
                            if q != '' and a != '' and len(inp.split(' ')) <=675:
                                yield id, {
                                    'input': inp,
                                    'output': a,
                                    'theme': theme
                                } 
                            id += 1 
                else:   
                    for i,turn in enumerate(conv['turns']):
                        if self.config.dataset_type == 'general':
                            
                            yield id, {
                                "conversation": conversation_id,
                                "theme": theme,
                                "text": turn['text'],
                                "turn": turn['turn_counter'],
                                "dangerous_tools": turn['dangerous_tools'],
                                "shared_data": turn['shared_data'],
                                "relevant": turn['relevant'],
                                "useful": turn['useful'],
                                "role": turn['role'],
                                "intent": turn['intent'],
                                "context_title": context['title'],
                                "context_description": context['description'],
                                "context_steps": context['steps']
                            }
                        else:    
                            if turn['intent'] in ['ask_question_ingredients_tools', 'ask_question_recipe_steps'] and conv['turns'][i+1] != None and turn['text']!=None:
                                history = []
                                for j in range(max(0,i-self.config.prev_utterances)-1, i-1):
                                    history.append(conv['turns'][j]['text'])
                                answer_turn = conv['turns'][i+1]
                                inp = self.generate_input_qa(turn['text'], context, history)
                                if len(inp.split(' ')) <=675:
                                    yield id, {
                                        'input': inp,
                                        'output': self.generate_output_qa(answer_turn['text'], answer_turn['shared_data']),
                                        'theme': theme
                                    } 
                        id += 1
        
    def generate_input_qa(self, text, context, history):
        # question format: Text | Recipe | History
        question = self.process_text(text)
        history_combined = ''
        role = 'teacher'
        for i in range(len(history)-1,-1,-1):
            history_combined = f"{role}: {history[i]}; {history_combined}"
            if role == 'teacher':
                role = 'student'
            else:
                role = 'teacher'
        context = f" | recipe title: {context['title']} | recipe description: {context['description']} | recipe ingredients: {context['ingredients']} | recipe steps: {context['steps']} | history: {history_combined}"
        context = self.process_text(context)
        # print(f'Question: {question}')
        # print(f'Context: {context}')
        return f"{question} SPLIT{context}"
        
        
    def generate_output_qa(self, text, shared_data):
        # answer format: Text | Shared data
        answer = f"{text} | shared data: {', '.join(shared_data)}"
        # print(f'Answer: {answer}')
        return self.process_text(answer)
    
    
    def process_text(self, text):
        text = text.lower()
        text = text.replace('"', '')
        text = text.replace("'", '')
        text = text.replace('\n', ';')
        text = text.replace('{', '')
        text = text.replace('}', '')
        return text
    
    
    def prepare_data(self):
        df_cooking = pd.read_json('https://wizard-of-tasks.s3.us-west-2.amazonaws.com/wizard_of_tasks_cooking_v1.0.json').transpose()
        df_diy = pd.read_json('https://wizard-of-tasks.s3.us-west-2.amazonaws.com/wizard_of_tasks_diy_v1.0.json').transpose()
        df = pd.concat([df_cooking, df_diy])
        df['conversation_id'] = df.index

        df[df['data_split']=='train'].to_json('.train.json', orient='records', lines=True)
        df[df['data_split']=='test'].to_json('.test.json', orient='records', lines=True)
        df[df['data_split']=='validation'].to_json('.dev.json', orient='records', lines=True)

    def get_context(self, url):
        recipe = {}

        request = requests.get(url)
        html = request.text
        status = request.status_code
        if 'wholefoodsmarket' in url and status == 200:
            soup = BeautifulSoup(html, 'html.parser')
            schema = soup.find_all('script', attrs={"type": "application/ld+json"})
            schema = json.loads(schema[0].string)
           
            recipe['title'] = soup.find(class_='w-header-title').text
            recipe['description'] = schema['description']

            ingredients = []
            for r in schema.get('recipeIngredient'):
                ingredients.append(r)
            recipe['ingredients'] = ingredients

            steps = []
            for r in schema.get('recipeInstructions'):
                steps.append(r.get('text'))
            recipe['steps'] = steps

        elif 'wikihow' in url and status == 200:
            soup = BeautifulSoup(html, 'html.parser')
            
            recipe['title'] = soup.find(id='section_0').text
            recipe['description'] = soup.find(id='mf-section-0').text
            recipe['steps'] = self.get_steps(soup)
            
            requirements = []
            tun_object = soup.find(id='thingsyoullneed')
            if tun_object:
                # Loop of ingredient objects to add full list.
                for child in tun_object.find_all('li'):
                    requirements.append(child.text.strip())
            recipe['ingredients'] = requirements


        else:
            recipe = None
        return recipe
        

    def get_method_number(self, soup):
        """ Scrape how many methods the DIY article contains and save this number in DIYDocument.number_of_parallel_
        methods
        """
        steps_lists = soup.find_all(class_='steps')
        if steps_lists != []:
            header = steps_lists[0].find('h3')
            if header:
                if "Part" in header.text:
                    # DIY article has parts
                    return 1
                else:
                    # DIY article has methods
                    number = len(steps_lists)
                    return number
            else:
                # DIY article has just one method, no parts
                return 1

    def get_steps(self, soup):
        """ Parse steps. """
        # article just has one method
        list_of_steps_list = soup.find_all('ol', class_='steps_list_2')
        method_number = self.get_method_number(soup)
        if list_of_steps_list != []:
            # loop through all parts (which include an array of steps) of the article
            if method_number > 1:
                list_of_steps_list = list_of_steps_list[:1]
            for part in list_of_steps_list:
                list_of_steps = self.get_steps_list(soup, part)
        return list_of_steps

    def get_steps_list(self, soup, steps_list):
        """ Helper function to scrape all steps from list object """
        # Loop over steps object to access associated text, images, or video.
        steps = []
        for step_tag in steps_list.find_all('li'):
            if step_tag.get('id') is not None:
                # Build Step object.
                step_object = step_tag.find(class_='step')
                if step_object:
                    # scrape bold step header
                    header = step_object.find('b')
                    if header:
                        header = header.text

                    # scrape all the remaining text after the bold step header
                    text = step_object.find_all(text=True)
                    if len(text) >= 3:
                        text = text[2]
                        if text[0] == '[':
                            text = ''
                        else:
                            text += ';'
                    elif text != []:
                        text = text[0]

                    # scrape all step text that is stored in a bulleted list
                    step_list_text = step_object.find('ul')
                    if step_list_text:
                        for bullet_point in step_list_text.find_all('li'):
                            text += bullet_point.find_all(text=True)[0].strip() + ';'
                    steps.append(f'{header}: {text}')
        return steps
